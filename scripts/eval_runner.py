"""Eval runner: query an OpenAI-compatible endpoint (vLLM / Anthropic / OpenAI)
on all 6 benchmark tasks and compute per-task metrics.

Usage:
  python eval_runner.py --model Qwen/Qwen3-7B-Instruct \
      --base-url http://localhost:8000/v1 --api-key EMPTY

  python eval_runner.py --model claude-sonnet-4-5 \
      --base-url https://api.anthropic.com/v1 --api-key $ANTHROPIC_API_KEY \
      --tasks translate fill-in
"""

import argparse
import concurrent.futures as cf
import json
import random
import statistics
import sys
import time
from pathlib import Path

# Make stdout line-buffered so progress lines flush when redirected
sys.stdout.reconfigure(line_buffering=True)

import urllib.request
import urllib.error

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from scorers import score  # noqa: E402

DATA_DIR = REPO / "data"
RESULTS_DIR = REPO / "results"

# Bounded retry with exponential backoff for transient API failures.
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # seconds; attempt n waits ~base * 2**n + jitter

TASK_FILES = {
    "translate": "translate.jsonl",
    "punctuate": "punctuate.jsonl",
    "char-gloss": "char_gloss.jsonl",
    "idiom-source": "idiom_source.jsonl",
    "fill-in": "fill_in.jsonl",
    "compress": "compress.jsonl",
}

SYSTEM_PROMPT = (
    "你是中国古典文献专家。回答力求简洁准确，不要解释，不要附加多余文字。"
)


def load_task(task: str) -> list[dict]:
    path = DATA_DIR / TASK_FILES[task]
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_prompt(rec: dict) -> str:
    return f"{rec['instruction']}\n\n{rec['input']}"


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: int = 120,
    max_tokens: int = 1024,
    extra_headers: dict | None = None,
    extra_body: dict | None = None,
) -> str:
    """Call OpenAI-compatible /chat/completions, return assistant content."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if extra_body:
        payload.update(extra_body)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def run_task(
    task: str,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int,
    limit: int | None,
    extra_headers: dict | None = None,
    extra_body: dict | None = None,
) -> dict:
    records = load_task(task)
    if limit:
        records = records[:limit]
    print(f"[{task}] {len(records)} questions, concurrency={concurrency}")

    preds = [None] * len(records)
    errs = [None] * len(records)  # per-item error string (None = succeeded)
    errors = 0
    t0 = time.time()

    def worker(i: int, rec: dict):
        """Call the endpoint with bounded retries + exponential backoff.

        Returns (i, prediction, error). On total failure prediction is None
        and error is the last exception string — callers MUST treat such items
        as "no data" rather than a genuine score of 0.
        """
        prompt = make_prompt(rec)
        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                pred = chat_completion(
                    base_url, api_key, model, prompt,
                    extra_headers=extra_headers,
                    extra_body=extra_body,
                )
                return i, pred, None
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < MAX_RETRIES - 1:
                    # exponential backoff with small jitter
                    sleep_s = RETRY_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5)
                    time.sleep(sleep_s)
        return i, None, str(last_err)

    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(worker, i, r) for i, r in enumerate(records)]
        done = 0
        for fut in cf.as_completed(futs):
            i, pred, err = fut.result()
            preds[i] = pred
            errs[i] = err
            if err:
                errors += 1
                if errors <= 3:
                    print(f"  [err] q{i}: {err}", file=sys.stderr)
            done += 1
            if done % 25 == 0 or done == len(records):
                print(f"  {done}/{len(records)}  ({time.time()-t0:.0f}s, {errors} err)")

    # score
    #
    # IMPORTANT: items that ended in an API error (pred is None) are NOT scored
    # and NOT folded into the summary metrics — an API failure is missing data,
    # not a model output worth 0. They are still emitted in `items` with an
    # `error` field so downstream tooling can see exactly which questions failed.
    items = []
    metric_acc: dict[str, list[float]] = {}
    n_scored = 0
    for rec, pred, err in zip(records, preds, errs):
        item = {
            "id": rec["id"],
            "input": rec["input"],
            "reference": rec["reference"],
            "prediction": pred,
        }
        if err is not None or pred is None:
            # API error / no response: record the error, leave scores absent.
            item["error"] = err if err is not None else "no response"
            item["scores"] = None
        else:
            s = score(rec, pred)
            for k, v in s.items():
                metric_acc.setdefault(k, []).append(v)
            item["scores"] = s
            n_scored += 1
        items.append(item)

    summary = {
        m: round(statistics.fmean(v), 4) for m, v in metric_acc.items()
    }
    return {
        "task": task,
        "n": len(records),
        # number of items actually scored (excludes API-error items)
        "n_scored": n_scored,
        "errors": errors,
        # fraction of items that failed with an API error (0..1)
        "error_rate": round(errors / len(records), 4) if records else 0.0,
        "elapsed_sec": round(time.time() - t0, 1),
        "summary": summary,
        "items": items,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model id (e.g., Qwen/Qwen3-7B-Instruct)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument(
        "--tasks",
        nargs="*",
        default=list(TASK_FILES.keys()),
        choices=list(TASK_FILES.keys()),
    )
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="limit questions per task (debug)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--header", action="append", default=[],
                    help="extra header K:V (repeat). e.g. --header 'x-skip-sanitize:true'")
    ap.add_argument("--extra-body", type=str, default=None,
                    help="JSON merged into request payload, e.g. "
                    "'{\"chat_template_kwargs\": {\"enable_thinking\": false}}'")
    args = ap.parse_args()

    extra_headers = {}
    for h in args.header:
        k, _, v = h.partition(":")
        if k.strip() and v.strip():
            extra_headers[k.strip()] = v.strip()

    extra_body = json.loads(args.extra_body) if args.extra_body else None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = args.model.replace("/", "_")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{safe_name}.json"

    # Merge with existing file so partial-task runs don't drop earlier results.
    if out_path.exists():
        try:
            all_results = json.loads(out_path.read_text(encoding="utf-8"))
            all_results.setdefault("tasks", {})
            all_results["model"] = args.model
            all_results["base_url"] = args.base_url
            all_results["last_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            all_results = None
    else:
        all_results = None
    if all_results is None:
        all_results = {
            "model": args.model,
            "base_url": args.base_url,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tasks": {},
        }

    for task in args.tasks:
        result = run_task(
            task=task,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            concurrency=args.concurrency,
            limit=args.limit,
            extra_headers=extra_headers or None,
            extra_body=extra_body,
        )
        all_results["tasks"][task] = result
        print(f"  ⇒ {task} summary: {result['summary']}")

    out_path.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote → {out_path.relative_to(REPO)}")
    print("\n=== summary ===")
    for t, r in all_results["tasks"].items():
        er = r.get("error_rate", 0.0)
        flag = "  ⚠️ HIGH ERROR RATE — results unreliable" if er > 0.05 else ""
        print(
            f"  {t:<14}  {r['summary']}  "
            f"(errors={r['errors']}, error_rate={er:.1%}){flag}"
        )


if __name__ == "__main__":
    main()
