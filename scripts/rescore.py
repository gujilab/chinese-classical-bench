"""Re-apply current scorers to every results/*.json in place.

Result files store the raw per-question prediction in
`tasks.<task>.items[].prediction`. When the scorers in `scorers.py` change
(e.g. better idiom-source book matcher, future LLM judge), this script
recomputes `scores` per item and the `summary` aggregate for every task —
no API calls, no re-running the models.

Usage:
  python scripts/rescore.py             # rescore all results/*.json
  python scripts/rescore.py --dry-run   # show diffs without writing
  python scripts/rescore.py --files results/glm-5.json
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


def load_records() -> dict[str, dict]:
    """id → full benchmark record (across all 5 tasks)."""
    out = {}
    for task, fname in TASK_FILES.items():
        with (DATA_DIR / fname).open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["id"]] = r
    return out


def rescore_file(fp: Path, recs: dict[str, dict]) -> tuple[dict, dict]:
    """Returns (new_doc, summary_diff). summary_diff: {task: {metric: (old, new)}}."""
    d = json.loads(fp.read_text(encoding="utf-8"))
    diff: dict[str, dict] = {}
    # Keys produced by the LLM judge (backfill_judge.py), not by score().
    # rescore must NOT drop them — merge scorer output, preserve the rest.
    JUDGE_KEYS = {"judge", "judge_norm", "judge_sonnet", "judge_sonnet_norm"}
    for task, tdata in d.get("tasks", {}).items():
        items = tdata.get("items", [])
        all_scores: dict[str, list[float]] = {}
        for it in items:
            rec = recs.get(it["id"])
            if rec is None:
                continue
            # Skip API-error items (prediction is None / has an `error` field):
            # they carry no data and must not be re-scored to a fake 0.
            if it.get("error") is not None or it.get("prediction") is None:
                continue
            sc = score(rec, it.get("prediction", ""))
            preserved = {k: v for k, v in (it.get("scores") or {}).items()
                         if k in JUDGE_KEYS}
            it["scores"] = {**sc, **preserved}
            for k, v in it["scores"].items():
                all_scores.setdefault(k, []).append(v)
        new_summary = {k: round(statistics.fmean(v), 4)
                       for k, v in all_scores.items() if v}
        # carry forward judge_n / judge_sonnet_n (counts, not per-item scores)
        for k, v in tdata.get("summary", {}).items():
            if k.endswith("_n") and k not in new_summary:
                new_summary[k] = v
        old_summary = tdata.get("summary", {})
        if new_summary != old_summary:
            diff[task] = {
                k: (old_summary.get(k), new_summary.get(k))
                for k in sorted(set(old_summary) | set(new_summary))
                if old_summary.get(k) != new_summary.get(k)
            }
        tdata["summary"] = new_summary
    return d, diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", type=Path, default=None,
                    help="specific files (default: results/*.json)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = args.files or [f for f in sorted(RESULTS.glob("*.json"))
                           if not f.name.startswith("_")]
    recs = load_records()

    any_change = False
    for fp in files:
        new_doc, diff = rescore_file(fp, recs)
        if not diff:
            print(f"{fp.name}: no change")
            continue
        any_change = True
        print(f"{fp.name}: changed")
        for task, metrics in diff.items():
            for m, (o, n) in metrics.items():
                o_s = f"{o:.4f}" if isinstance(o, (int, float)) else str(o)
                n_s = f"{n:.4f}" if isinstance(n, (int, float)) else str(n)
                print(f"  {task}.{m}: {o_s} → {n_s}")
        if not args.dry_run:
            fp.write_text(
                json.dumps(new_doc, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    if args.dry_run and any_change:
        print("\n(dry-run: no files modified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
