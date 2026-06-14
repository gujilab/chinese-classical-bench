"""LLM-as-judge re-scoring for chinese-classical-bench translate + char-gloss.

Uses claude-opus-4-7 (via kcli-gw) to assign a 0-5 semantic score to each
prediction in `results/<model>.json`, then writes `judge_scores.jsonl` and
prints a quick correlation summary.

Usage:
  python judge.py [--limit N] [--tasks translate char-gloss] [--models ...]

Notes:
- Concurrency capped at 2 (shared gateway).
- Cache keyed by (model, task, id) — re-running skips already-judged items.
- If judge returns no digit, we log and skip.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
RESULTS_DIR = REPO / "results"
OUT_DIR = Path(__file__).resolve().parent
CACHE_PATH = OUT_DIR / "judge_scores.jsonl"

DEFAULT_MODELS = [
    "claude-opus-4-7",
    "claude-opus-4-7-thinking",
    "claude-sonnet-4-6",
    "deepseek-3.2",
    "glm-5",
]
DEFAULT_TASKS = ["translate", "char-gloss"]

BASE_URL = os.environ.get("KCLI_BASE_URL", "http://localhost:8990/v1")
API_KEY = os.environ.get("KCLI_API_KEY")
if not API_KEY:
    sys.exit("KCLI_API_KEY not set — export it before running (no default key).")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-4-7")
MAX_CONCURRENCY = 8  # gateway raised to 6000 rpm + burst 1000; 5 is safe with sibling agents
TIMEOUT_S = 90
MAX_TOKENS = 8

TASK_FILES = {
    "translate": "translate.jsonl",
    "char-gloss": "char_gloss.jsonl",
}

PROMPT_TRANSLATE = """你是古文翻译评分专家。请阅读原文、参考翻译、模型译文，给模型译文打分 0-5 整数:
0 = 完全错误或胡说
1 = 大部分错误
2 = 部分正确，关键信息错
3 = 基本正确，细节有损
4 = 准确流畅，与参考几乎等价
5 = 比参考更精准或同样优秀

原文: {input}
参考译文: {reference}
模型译文: {prediction}

直接输出整数 0-5，不要解释。"""

PROMPT_CHAR_GLOSS = """你是古汉语字义评分专家。请评分模型对古文中某字含义的解释 0-5 整数:
0 = 字义完全错
1 = 含义偏离
2 = 部分正确
3 = 基本正确
4 = 准确
5 = 准确且包含语境

问题: {input}
参考释义: {reference}
模型释义: {prediction}

直接输出整数 0-5，不要解释。"""

PROMPTS = {"translate": PROMPT_TRANSLATE, "char-gloss": PROMPT_CHAR_GLOSS}


# ----------------- IO helpers -----------------

def load_data(task: str) -> dict[str, dict]:
    """Return {id: record} for a task."""
    path = DATA_DIR / TASK_FILES[task]
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def load_results(model: str) -> dict:
    path = RESULTS_DIR / f"{model}.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_cache() -> dict[tuple[str, str, str], dict]:
    """Read existing judge_scores.jsonl. Last entry per key wins."""
    if not CACHE_PATH.exists():
        return {}
    cache: dict[tuple[str, str, str], dict] = {}
    with CACHE_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec["model"], rec["task"], rec["id"])
            cache[key] = rec
    return cache


# ----------------- judge call -----------------

def chat_once(prompt: str) -> str:
    url = BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": "你是评分助手。只输出一个 0-5 之间的整数。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


_DIGIT_RE = re.compile(r"[0-5]")


def parse_score(raw: str) -> int | None:
    """Extract first digit 0-5 from response. Returns None if none found."""
    if not raw:
        return None
    m = _DIGIT_RE.search(raw)
    if not m:
        return None
    return int(m.group(0))


def judge_one(model: str, task: str, qid: str, rec: dict, prediction: str,
              chrf: float, max_retries: int = 3) -> dict | None:
    prompt = PROMPTS[task].format(
        input=rec.get("input", ""),
        reference=rec.get("reference", ""),
        prediction=prediction,
    )
    last_err = None
    raw = ""
    for attempt in range(max_retries):
        try:
            raw = chat_once(prompt)
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ConnectionError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    else:
        print(f"  [err] {model} {task} {qid}: {last_err}", file=sys.stderr)
        return None

    score = parse_score(raw)
    if score is None:
        print(f"  [skip] {model} {task} {qid}: no digit in {raw!r}",
              file=sys.stderr)
        return None
    return {
        "model": model,
        "task": task,
        "id": qid,
        "chrf": chrf,
        "judge": score,
        "judge_raw": raw.strip(),
        "prediction": prediction,
        "reference": rec.get("reference", ""),
        "input": rec.get("input", ""),
    }


# ----------------- main loop -----------------

def gather_jobs(models: list[str], tasks: list[str], data_by_task: dict[str, dict],
                cache: dict, limit: int | None) -> list[tuple]:
    """Build list of (model, task, qid, rec, prediction, chrf) jobs not in cache."""
    jobs = []
    for model in models:
        try:
            res = load_results(model)
        except FileNotFoundError:
            print(f"  [warn] missing result file for {model}", file=sys.stderr)
            continue
        for task in tasks:
            tdata = res.get("tasks", {}).get(task)
            if not tdata:
                continue
            items = tdata.get("items", [])
            if limit:
                items = items[:limit]
            for it in items:
                qid = it["id"]
                pred = it.get("prediction", "")
                if not pred:
                    continue
                rec = data_by_task[task].get(qid)
                if rec is None:
                    continue
                chrf = (it.get("scores") or {}).get("chrf", 0.0)
                if (model, task, qid) in cache:
                    continue
                jobs.append((model, task, qid, rec, pred, chrf))
    return jobs


def run(models: list[str], tasks: list[str], limit: int | None,
        concurrency: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data_by_task = {t: load_data(t) for t in tasks}
    cache = load_cache()
    print(f"[cache] {len(cache)} existing judge scores loaded")
    jobs = gather_jobs(models, tasks, data_by_task, cache, limit)
    print(f"[jobs] {len(jobs)} new judge calls to make "
          f"(models={len(models)}, tasks={tasks}, concurrency={concurrency})")
    if not jobs:
        return

    t0 = time.time()
    done = 0
    skipped = 0
    with CACHE_PATH.open("a", encoding="utf-8") as fout:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = {
                pool.submit(judge_one, m, t, qid, rec, pred, chrf): (m, t, qid)
                for (m, t, qid, rec, pred, chrf) in jobs
            }
            for fut in cf.as_completed(futs):
                result = fut.result()
                done += 1
                if result is None:
                    skipped += 1
                else:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                if done % 25 == 0 or done == len(jobs):
                    rate = done / max(time.time() - t0, 0.1)
                    print(f"  {done}/{len(jobs)}  "
                          f"({time.time()-t0:.0f}s, {rate:.1f} req/s, "
                          f"{skipped} skipped)")


# ----------------- CLI -----------------

def main() -> None:
    global JUDGE_MODEL, CACHE_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                    choices=list(TASK_FILES.keys()))
    ap.add_argument("--limit", type=int, default=None,
                    help="limit per-task questions (for smoke test)")
    ap.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)
    ap.add_argument("--judge-model", default=JUDGE_MODEL,
                    help=f"judge model (default {JUDGE_MODEL}). Use claude-sonnet-4-6 for cross-judge.")
    ap.add_argument("--cache-path", type=Path, default=CACHE_PATH,
                    help=f"override cache file (default {CACHE_PATH.name})")
    args = ap.parse_args()
    if args.concurrency > MAX_CONCURRENCY:
        print(f"[note] capping concurrency at {MAX_CONCURRENCY}")
        args.concurrency = MAX_CONCURRENCY
    JUDGE_MODEL = args.judge_model
    CACHE_PATH = args.cache_path
    print(f"[config] judge_model={JUDGE_MODEL} cache={CACHE_PATH.name}")
    run(args.models, args.tasks, args.limit, args.concurrency)


if __name__ == "__main__":
    main()
