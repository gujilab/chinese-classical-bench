"""Retroactively flag empty-prediction items (API failures) as errors.

Some early result files stored API-failed questions as `prediction: ""` with
NO error marker, and the scorer then gave them an all-zero score that polluted
the task summary. For the affected (model, task) cells below, the count of
empty predictions equals the recorded `errors` count for that task — confirming
that an empty prediction == an API failure (verified, not assumed).

This script, for exactly those cells:
  1. marks every item whose prediction is "" (or None) as an error:
       item["error"]  = "empty output (API failure, retroactively flagged)"
       item["scores"] = None
  2. re-scores every other item via scorers.score (no API calls)
  3. recomputes the task-level summary / n_scored / errors / error_rate on
     the real (successful) items only.

Other (model, task) cells are left untouched.

Usage:
  python scripts/flag_empty_errors.py --dry-run
  python scripts/flag_empty_errors.py
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from scorers import score  # noqa: E402

DATA_DIR = REPO / "data"
RESULTS = REPO / "results"

TASK_FILES = {
    "translate":    "translate.jsonl",
    "punctuate":    "punctuate.jsonl",
    "char-gloss":   "char_gloss.jsonl",
    "idiom-source": "idiom_source.jsonl",
    "fill-in":      "fill_in.jsonl",
    "compress":     "compress.jsonl",
}

# Only these (file, task) cells were polluted by the un-flagged API failures.
TARGETS = {
    "claude-opus-4-7.json":          ["compress"],
    "claude-opus-4-7-thinking.json": ["compress"],
    "minimax-m2.5.json":             ["char-gloss", "idiom-source"],
}

EMPTY_ERROR = "empty output (API failure, retroactively flagged)"

JUDGE_KEYS = {"judge", "judge_norm", "judge_sonnet", "judge_sonnet_norm"}


def load_records() -> dict[str, dict]:
    out = {}
    for fname in TASK_FILES.values():
        with (DATA_DIR / fname).open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["id"]] = r
    return out


def is_empty(it: dict) -> bool:
    p = it.get("prediction")
    return p is None or (isinstance(p, str) and p.strip() == "")


def process_cell(tdata: dict, recs: dict[str, dict]) -> dict:
    """Flag empties + rescore the rest. Returns before/after numbers."""
    items = tdata.get("items", [])
    n = len(items)
    flagged = 0
    all_scores: dict[str, list[float]] = {}
    for it in items:
        # Already an error (e.g. a real prior error) -> leave it, count it.
        if it.get("error") is not None:
            continue
        if is_empty(it):
            it["error"] = EMPTY_ERROR
            it["scores"] = None
            flagged += 1
            continue
        rec = recs.get(it["id"])
        if rec is None:
            continue
        sc = score(rec, it.get("prediction", ""))
        preserved = {k: v for k, v in (it.get("scores") or {}).items()
                     if k in JUDGE_KEYS}
        it["scores"] = {**sc, **preserved}
        for k, v in it["scores"].items():
            all_scores.setdefault(k, []).append(v)

    errors = sum(1 for it in items if it.get("error") is not None)
    n_scored = sum(1 for it in items
                   if it.get("error") is None and it.get("scores"))
    new_summary = {k: round(statistics.fmean(v), 4)
                   for k, v in all_scores.items() if v}
    for k, v in tdata.get("summary", {}).items():
        if k.endswith("_n") and k not in new_summary:
            new_summary[k] = v

    before = {
        "summary": dict(tdata.get("summary", {})),
        "errors": tdata.get("errors"),
        "n_scored": tdata.get("n_scored"),
        "error_rate": tdata.get("error_rate"),
    }
    tdata["summary"] = new_summary
    tdata["errors"] = errors
    tdata["n_scored"] = n_scored
    tdata["error_rate"] = round(errors / n, 4) if n else None
    after = {
        "summary": new_summary,
        "errors": errors,
        "n_scored": n_scored,
        "error_rate": tdata["error_rate"],
    }
    return {"flagged": flagged, "n": n, "before": before, "after": after}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    recs = load_records()
    for fname, tasks in TARGETS.items():
        fp = RESULTS / fname
        d = json.loads(fp.read_text(encoding="utf-8"))
        print(f"\n=== {fname} ===")
        for task in tasks:
            tdata = d["tasks"][task]
            r = process_cell(tdata, recs)
            print(f"  [{task}]  n={r['n']}  newly_flagged_empty={r['flagged']}")
            b, a = r["before"], r["after"]
            print(f"    errors:     {b['errors']} -> {a['errors']}")
            print(f"    n_scored:   {b['n_scored']} -> {a['n_scored']}")
            print(f"    error_rate: {b['error_rate']} -> {a['error_rate']}")
            keys = sorted(set(b["summary"]) | set(a["summary"]))
            for k in keys:
                ov, nv = b["summary"].get(k), a["summary"].get(k)
                if ov != nv:
                    print(f"    summary.{k}: {ov} -> {nv}")
        if not args.dry_run:
            fp.write_text(
                json.dumps(d, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    if args.dry_run:
        print("\n(dry-run: no files modified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
