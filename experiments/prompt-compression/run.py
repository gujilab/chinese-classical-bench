#!/usr/bin/env python3
"""
Prompt compression experiment: 文言文 vs 现代中文 vs English on C-Eval humanities.

End-to-end:
 1. Load 50 C-Eval val questions (humanities)
 2. Generate 3 prompt versions (modern_cn / english / classical_cn) via Claude Sonnet 4.6
 3. Eval each (version × question) on 3 models via kcli-gw
 4. Analyze: accuracy + token cost per (model, version)

Concurrency capped at 2 (kcli-gw shared).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Local OpenAI SDK (pip install --user openai)
sys.path.insert(0, os.path.expanduser("~/Library/Python/3.13/lib/python/site-packages"))
sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.13/site-packages"))
sys.path.insert(0, os.path.expanduser("~/Library/Python/3.12/lib/python/site-packages"))

from openai import AsyncOpenAI  # noqa: E402
from datasets import load_dataset  # noqa: E402

# ---- Config ---------------------------------------------------------------

ROOT = Path(__file__).parent
BASE_URL = os.environ.get("KCLI_BASE_URL", "http://localhost:8990/v1")
API_KEY = os.environ.get("KCLI_API_KEY")
if not API_KEY:
    sys.exit("KCLI_API_KEY not set — export it before running (no default key).")
TRANSLATOR = os.environ.get("TRANSLATOR_MODEL", "claude-sonnet-4-6")
EVAL_MODELS = os.environ.get(
    "EVAL_MODELS", "claude-opus-4-7,deepseek-3.2,qwen3-coder-next"
).split(",")
MAX_CONC = int(os.environ.get("MAX_CONCURRENCY", "5"))

PROMPTS_FILE = ROOT / "prompts.jsonl"
RESULTS_FILE = ROOT / "results.jsonl"
REPORT_FILE = ROOT / "report.md"
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

VERSIONS = ["modern_cn", "english", "classical_cn"]

# ---- Dataset --------------------------------------------------------------

# (ceval config, n_to_take)
CEVAL_CONFIGS = [
    ("chinese_language_and_literature", 23),
    ("high_school_chinese", 19),
    ("middle_school_history", 8),
]


def load_questions() -> list[dict[str, Any]]:
    """Load 50 questions from C-Eval val splits."""
    out: list[dict[str, Any]] = []
    for cfg, n in CEVAL_CONFIGS:
        ds = load_dataset("ceval/ceval-exam", cfg, split="val")
        for i, row in enumerate(ds):
            if i >= n:
                break
            if not row.get("answer") or row["answer"] not in "ABCD":
                continue
            out.append(
                {
                    "question_id": f"{cfg}-{row['id']}",
                    "source_config": cfg,
                    "question": row["question"].strip(),
                    "A": row["A"].strip(),
                    "B": row["B"].strip(),
                    "C": row["C"].strip(),
                    "D": row["D"].strip(),
                    "correct_letter": row["answer"].strip(),
                }
            )
    return out


# ---- Prompt builders ------------------------------------------------------


def fmt_modern_cn(q: dict[str, Any]) -> str:
    return (
        f"{q['question']}\n"
        f"A. {q['A']}\n"
        f"B. {q['B']}\n"
        f"C. {q['C']}\n"
        f"D. {q['D']}"
    )


SYS_PROMPT = {
    "modern_cn": "你是一位答题助手。请阅读以下多选题，只输出正确答案的字母（A、B、C 或 D），不要任何其他内容。",
    "english": "You are an answering assistant. Read the multiple-choice question and output only the letter of the correct answer (A, B, C, or D). No other text.",
    "classical_cn": "君为答题之士。阅下问，仅出一字以答（A、B、C 或 D），勿赘他言。",
}


# ---- Async client ---------------------------------------------------------


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        timeout=120.0,
        max_retries=0,  # we'll retry ourselves
    )


SEMAPHORE = asyncio.Semaphore(MAX_CONC)


async def chat_call(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1024,
    retries: int = 3,
) -> dict[str, Any]:
    """One chat call through the semaphore with retry."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            async with SEMAPHORE:
                t0 = time.monotonic()
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                latency = time.monotonic() - t0
                content = (resp.choices[0].message.content or "").strip()
                usage = resp.usage
                return {
                    "content": content,
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                    "latency_s": round(latency, 3),
                }
        except Exception as e:  # noqa: BLE001
            last_err = e
            # gentle backoff; kcli-gw is shared
            await asyncio.sleep(2 + attempt * 2)
    raise RuntimeError(f"chat_call failed after {retries} tries: {last_err}")


# ---- Translation ----------------------------------------------------------


TRANSLATE_INSTR = {
    "english": (
        "Translate the following Chinese multiple-choice question and its four options "
        "into natural English. Keep the option letters A/B/C/D. Keep meaning faithful. "
        "Output ONLY the translated question + options, no preamble.\n\n---\n{src}"
    ),
    "classical_cn": (
        "把下面的多选题问题和选项改写为文言文，保持准确性和可解性。"
        "Choice 字母 A/B/C/D 保留。只输出改写后的题目和四个选项，不要任何前后说明。\n\n---\n{src}"
    ),
}


async def translate_one(
    client: AsyncOpenAI, q: dict[str, Any], target: str
) -> str:
    src = fmt_modern_cn(q)
    cache_key = CACHE_DIR / f"trans-{target}-{q['question_id']}.txt"
    if cache_key.exists():
        return cache_key.read_text(encoding="utf-8")
    prompt = TRANSLATE_INSTR[target].format(src=src)
    resp = await chat_call(
        client,
        model=TRANSLATOR,
        system="You rewrite text precisely as instructed. Output only the rewritten text.",
        user=prompt,
        max_tokens=2048,
    )
    text = resp["content"].strip()
    # strip code-fence accidents
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    cache_key.write_text(text, encoding="utf-8")
    return text


# ---- Eval -----------------------------------------------------------------


LETTER_RE = re.compile(r"\b([ABCD])\b")


def parse_answer(text: str) -> str | None:
    if not text:
        return None
    s = text.strip()
    # quick path: first non-whitespace char is a letter
    head = s.lstrip("`'\"*# \t\n")
    if head and head[0].upper() in "ABCD":
        return head[0].upper()
    m = LETTER_RE.search(s.upper())
    return m.group(1) if m else None


async def eval_one(
    client: AsyncOpenAI,
    model: str,
    version: str,
    prompt_text: str,
    q: dict[str, Any],
) -> dict[str, Any]:
    resp = await chat_call(
        client,
        model=model,
        system=SYS_PROMPT[version],
        user=prompt_text,
        max_tokens=16,  # only need the letter
    )
    pred = parse_answer(resp["content"])
    return {
        "question_id": q["question_id"],
        "source_config": q["source_config"],
        "version": version,
        "model": model,
        "predicted": pred,
        "raw_response": resp["content"],
        "correct_letter": q["correct_letter"],
        "is_correct": pred == q["correct_letter"],
        "prompt_chars": len(prompt_text),
        "output_chars": len(resp["content"]),
        "prompt_tokens": resp["prompt_tokens"],
        "completion_tokens": resp["completion_tokens"],
        "latency_s": resp["latency_s"],
    }


# ---- Main stages ----------------------------------------------------------


async def stage_translate(client: AsyncOpenAI, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build prompts.jsonl: 3 versions per question."""
    prompts: list[dict[str, Any]] = []
    # All modern first (no API calls)
    for q in questions:
        prompts.append(
            {
                "question_id": q["question_id"],
                "source_config": q["source_config"],
                "version": "modern_cn",
                "prompt": fmt_modern_cn(q),
                "correct_letter": q["correct_letter"],
            }
        )
    # Translate to english + classical, gather
    tasks = []
    meta = []
    for q in questions:
        for v in ("english", "classical_cn"):
            tasks.append(translate_one(client, q, v))
            meta.append((q, v))
    print(f"[translate] {len(tasks)} translation tasks (cached on disk)…", flush=True)
    done = 0
    results = [None] * len(tasks)
    # run in semaphore-controlled batches via asyncio.gather (semaphore already caps)
    async def runner(i: int, coro):
        nonlocal done
        try:
            results[i] = await coro
        except Exception as e:  # noqa: BLE001
            results[i] = f"__ERR__: {e}"
        done += 1
        if done % 10 == 0 or done == len(tasks):
            print(f"  translated {done}/{len(tasks)}", flush=True)

    await asyncio.gather(*[runner(i, t) for i, t in enumerate(tasks)])
    for (q, v), text in zip(meta, results):
        prompts.append(
            {
                "question_id": q["question_id"],
                "source_config": q["source_config"],
                "version": v,
                "prompt": text,
                "correct_letter": q["correct_letter"],
            }
        )

    with PROMPTS_FILE.open("w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[translate] wrote {len(prompts)} prompts → {PROMPTS_FILE.name}", flush=True)
    return prompts


async def stage_eval(
    client: AsyncOpenAI,
    prompts: list[dict[str, Any]],
    questions_idx: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    tasks = []
    meta = []
    for p in prompts:
        if isinstance(p["prompt"], str) and p["prompt"].startswith("__ERR__"):
            continue
        q = questions_idx[p["question_id"]]
        for m in EVAL_MODELS:
            tasks.append(eval_one(client, m, p["version"], p["prompt"], q))
            meta.append((p["question_id"], p["version"], m))

    print(f"[eval] {len(tasks)} eval calls (3 models × 3 versions × N)…", flush=True)
    results: list[Any] = [None] * len(tasks)
    done = 0
    failed = 0

    async def runner(i: int, coro):
        nonlocal done, failed
        try:
            results[i] = await coro
        except Exception as e:  # noqa: BLE001
            qid, v, m = meta[i]
            results[i] = {
                "question_id": qid,
                "version": v,
                "model": m,
                "predicted": None,
                "is_correct": False,
                "error": str(e),
                "prompt_chars": 0,
                "output_chars": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_s": 0,
            }
            failed += 1
        done += 1
        if done % 25 == 0 or done == len(tasks):
            print(f"  eval {done}/{len(tasks)}  (failed: {failed})", flush=True)

    await asyncio.gather(*[runner(i, t) for i, t in enumerate(tasks)])

    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[eval] wrote {len(results)} rows → {RESULTS_FILE.name}", flush=True)
    return results


# ---- Analysis -------------------------------------------------------------


def analyze() -> None:
    rows = [json.loads(l) for l in RESULTS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    # aggregate
    agg: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "correct": 0,
            "prompt_chars_sum": 0,
            "output_chars_sum": 0,
            "prompt_tokens_sum": 0,
            "errors": 0,
        }
    )
    for r in rows:
        key = (r["model"], r["version"])
        a = agg[key]
        a["n"] += 1
        if r.get("error"):
            a["errors"] += 1
            continue
        if r.get("is_correct"):
            a["correct"] += 1
        a["prompt_chars_sum"] += r.get("prompt_chars", 0)
        a["output_chars_sum"] += r.get("output_chars", 0)
        a["prompt_tokens_sum"] += r.get("prompt_tokens", 0)

    # Format report
    lines: list[str] = []
    lines.append("# Prompt Compression Experiment — Report\n")
    lines.append(
        "**Setup:** 50 C-Eval humanities multiple-choice questions × 3 prompt versions "
        "(modern_cn baseline, english translation, classical_cn rewrite) × 3 evaluator models "
        f"({', '.join(EVAL_MODELS)}). Translations by `{TRANSLATOR}` via kcli-gw at `{BASE_URL}`. "
        f"Concurrency cap = {MAX_CONC}.\n"
    )
    lines.append("## Per-(model, version) summary\n")
    lines.append(
        "model | version | n | accuracy | mean prompt chars | mean output chars | mean prompt_tokens (kcli-reported)\n"
    )
    lines.append("--- | --- | --- | --- | --- | --- | ---\n")

    # also collect for headline
    by_model: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)

    for model in EVAL_MODELS:
        for version in VERSIONS:
            key = (model, version)
            if key not in agg:
                continue
            a = agg[key]
            n_ok = a["n"] - a["errors"]
            acc = (a["correct"] / n_ok * 100) if n_ok else 0
            mc = (a["prompt_chars_sum"] / n_ok) if n_ok else 0
            oc = (a["output_chars_sum"] / n_ok) if n_ok else 0
            pt = (a["prompt_tokens_sum"] / n_ok) if n_ok else 0
            lines.append(
                f"{model} | {version} | {a['n']} (err {a['errors']}) | {acc:.1f}% | "
                f"{mc:.0f} | {oc:.0f} | {pt:.0f}\n"
            )
            by_model[model][version] = {
                "accuracy": acc,
                "prompt_chars": mc,
                "prompt_tokens": pt,
            }

    lines.append("\n## Compression vs accuracy (classical_cn relative to baselines)\n")
    lines.append("model | acc Δ vs modern_cn | char savings vs modern_cn | acc Δ vs english | char savings vs english\n")
    lines.append("--- | --- | --- | --- | ---\n")
    for model in EVAL_MODELS:
        d = by_model.get(model, {})
        m = d.get("modern_cn"); e = d.get("english"); c = d.get("classical_cn")
        if not (m and e and c):
            lines.append(f"{model} | (missing data)\n")
            continue
        d_acc_m = c["accuracy"] - m["accuracy"]
        d_acc_e = c["accuracy"] - e["accuracy"]
        chr_save_m = (1 - c["prompt_chars"] / m["prompt_chars"]) * 100 if m["prompt_chars"] else 0
        chr_save_e = (1 - c["prompt_chars"] / e["prompt_chars"]) * 100 if e["prompt_chars"] else 0
        lines.append(
            f"{model} | {d_acc_m:+.1f}pp | {chr_save_m:+.1f}% | {d_acc_e:+.1f}pp | {chr_save_e:+.1f}%\n"
        )

    # headline
    avg_acc_loss_vs_modern = []
    avg_char_save_vs_modern = []
    avg_acc_loss_vs_english = []
    avg_char_save_vs_english = []
    for model in EVAL_MODELS:
        d = by_model.get(model, {})
        m = d.get("modern_cn"); e = d.get("english"); c = d.get("classical_cn")
        if not (m and e and c):
            continue
        avg_acc_loss_vs_modern.append(c["accuracy"] - m["accuracy"])
        avg_char_save_vs_modern.append((1 - c["prompt_chars"] / m["prompt_chars"]) * 100 if m["prompt_chars"] else 0)
        avg_acc_loss_vs_english.append(c["accuracy"] - e["accuracy"])
        avg_char_save_vs_english.append((1 - c["prompt_chars"] / e["prompt_chars"]) * 100 if e["prompt_chars"] else 0)

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0

    lines.append("\n## Headline\n")
    lines.append(
        f"Averaged across the 3 models, switching prompts from **现代中文 → 文言文** changes accuracy by "
        f"**{mean(avg_acc_loss_vs_modern):+.1f}pp** while saving **{mean(avg_char_save_vs_modern):+.1f}%** of input characters. "
        f"vs **English → 文言文**: accuracy **{mean(avg_acc_loss_vs_english):+.1f}pp**, char savings **{mean(avg_char_save_vs_english):+.1f}%**.\n\n"
    )
    lines.append(
        "_Caveat:_ kcli-gw's reported `prompt_tokens` includes a per-model system overhead (≈1.2k–4.5k tokens) that dwarfs the question payload, "
        "so character count is the cleanest version-comparison metric here. The tokenizer-level study (`../../tokenizer_study/`) gives the per-language token ratios.\n"
    )

    REPORT_FILE.write_text("".join(lines), encoding="utf-8")
    print(f"[analyze] wrote {REPORT_FILE.name}", flush=True)
    print("".join(lines))


# ---- Orchestration --------------------------------------------------------


async def main_async(skip_translate: bool, skip_eval: bool) -> None:
    client = make_client()
    questions = load_questions()
    print(f"[load] {len(questions)} questions", flush=True)
    questions_idx = {q["question_id"]: q for q in questions}

    if not skip_translate or not PROMPTS_FILE.exists():
        prompts = await stage_translate(client, questions)
    else:
        prompts = [json.loads(l) for l in PROMPTS_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"[translate] reused existing {PROMPTS_FILE.name} ({len(prompts)} rows)", flush=True)

    if not skip_eval or not RESULTS_FILE.exists():
        await stage_eval(client, prompts, questions_idx)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--analyze", action="store_true", help="Only run analysis on existing results.jsonl")
    p.add_argument("--skip-translate", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    if args.analyze:
        analyze()
        return

    asyncio.run(main_async(args.skip_translate, args.skip_eval))
    analyze()


if __name__ == "__main__":
    main()
