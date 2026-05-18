"""Bootstrap 95% confidence intervals for the per-task and average scores.

Reads results/*.json, resamples item-level scores per (model, task) and
reports the percentile-based CI. Used by aggregate.py to show "0.482 ±0.021"
style numbers and to mark statistical ties on the leaderboard.

Usage:
  python scripts/bootstrap_ci.py                # write results/_bootstrap.json
  python scripts/bootstrap_ci.py --iters 5000   # more iters for tighter CI
  python scripts/bootstrap_ci.py --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = RESULTS / "_bootstrap.json"

# PRIMARY metric per task. translate/char-gloss use the LLM judge
# (judge_norm, Opus 0-1) — chrF systematically under-rates synonymous
# paraphrase (see findings.md §2 / experiments/llm-judge). chrF is kept as a
# labelled reproducible floor in aggregate.py, not as the headline.
HEADLINE = {
    "translate":     "judge_norm",
    "punctuate":     "punct_f1",
    "char-gloss":    "judge_norm",
    "idiom-source":  "book_em",
    "fill-in":       "exact_match",
    "compress":      "efficiency",
}


def per_item_scores(doc: dict) -> dict[str, list[float]]:
    """{task: [headline_metric per item]}."""
    out: dict[str, list[float]] = {}
    for task, tdata in doc.get("tasks", {}).items():
        metric = HEADLINE.get(task)
        if metric is None:
            continue
        vals = []
        for it in tdata.get("items", []):
            sc = it.get("scores") or {}
            v = sc.get(metric)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            out[task] = vals
    return out


def bootstrap_ci(values: list[float], iters: int, rng: random.Random,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for given alpha (default 95% CI)."""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = statistics.fmean(values)
    if n == 1:
        return (mean, mean, mean)
    means = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return (mean, lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    files = sorted(RESULTS.glob("*.json"))
    files = [f for f in files if not f.name.startswith("_")]

    summary: dict[str, dict] = {}
    for fp in files:
        doc = json.loads(fp.read_text(encoding="utf-8"))
        model = doc.get("model", fp.stem)
        scores_by_task = per_item_scores(doc)
        model_summary: dict[str, dict] = {}

        # per-task CI on headline metric
        task_means: list[float] = []
        for task, vals in scores_by_task.items():
            mean, lo, hi = bootstrap_ci(vals, args.iters, rng)
            model_summary[task] = {
                "metric": HEADLINE[task],
                "n": len(vals),
                "mean": round(mean, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "half_width": round((hi - lo) / 2, 4),
            }
            task_means.append(mean)

        # avg-of-task-means CI: bootstrap by resampling tasks themselves
        # (this is conservative — actual leaderboard avg is mean of per-task means)
        if task_means:
            avg = statistics.fmean(task_means)
            # For avg CI, resample item-level jointly per task.
            iters = args.iters
            avg_samples = []
            for _ in range(iters):
                per_task_resampled = []
                for task, vals in scores_by_task.items():
                    n = len(vals)
                    s = sum(vals[rng.randrange(n)] for _ in range(n)) / n
                    per_task_resampled.append(s)
                avg_samples.append(sum(per_task_resampled) / len(per_task_resampled))
            avg_samples.sort()
            lo = avg_samples[int(0.025 * iters)]
            hi = avg_samples[int(0.975 * iters) - 1]
            model_summary["_avg"] = {
                "mean": round(avg, 4),
                "ci_lo": round(lo, 4),
                "ci_hi": round(hi, 4),
                "half_width": round((hi - lo) / 2, 4),
            }
        summary[model] = model_summary

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({
            "iters": args.iters,
            "seed": args.seed,
            "alpha": 0.05,
            "models": summary,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote → {args.out.relative_to(REPO)}  ({len(summary)} models, {args.iters} iters)")

    # quick stdout preview: ranked by avg, with CI
    print()
    rows = []
    for model, ms in summary.items():
        avg = ms.get("_avg")
        if avg:
            rows.append((model, avg["mean"], avg["ci_lo"], avg["ci_hi"]))
    rows.sort(key=lambda r: -r[1])
    print(f"{'model':<32} {'avg':>7}  95% CI")
    for m, mean, lo, hi in rows:
        print(f"{m:<32} {mean:>7.3f}  [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
