"""Aggregate per-model results files into the leaderboard.

Reads results/*.json (per-item predictions + scores) and results/_bootstrap.json
(precomputed 95% CIs) and renders:
  1. Primary leaderboard — PRIMARY metric per task ±CI. translate/char-gloss
     use the LLM judge (chrF under-rates paraphrase, findings.md §2);
     punctuate also shows char_preserved (the additive fidelity diagnostic,
     task-redundancy.md §7).
  2. Canonicity-stratified — Avg on core-canon (T3) vs obscure (T1) sources
     + recall gap, sorted by the contamination-robust T1 ranking
     (contamination.md / findings.md §6).
  3. Transparency — chrF reproducible floor + two-judge cross-check for
     translate/char-gloss (shows the judge headline is not cherry-picked).

Usage:
  python scripts/aggregate.py                  # print to stdout
  python scripts/aggregate.py --out leaderboard.md
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
BOOTSTRAP = RESULTS / "_bootstrap.json"
sys.path.insert(0, str(REPO / "scripts"))
from canonicity import id_to_tier  # noqa: E402

# PRIMARY metric per task (translate/char-gloss promoted to LLM judge).
HEADLINE = {
    "translate":     ("judge_norm",  "Judge"),
    "punctuate":     ("punct_f1",    "Punct F1"),
    "char-gloss":    ("judge_norm",  "Judge"),
    "idiom-source":  ("book_em",     "Book EM"),
    "fill-in":       ("exact_match", "Exact"),
    "compress":      ("efficiency",  "Compress Eff"),
}
TASK_ORDER = list(HEADLINE.keys())
JUDGE_TASKS = ["translate", "char-gloss"]


def load_bootstrap() -> dict:
    if not BOOTSTRAP.exists():
        return {}
    return json.loads(BOOTSTRAP.read_text(encoding="utf-8")).get("models", {})


def fmt_with_ci(mean: float | None, ci: dict | None) -> str:
    if mean is None:
        return "—"
    if ci and ci.get("half_width") is not None:
        return f"{mean:.3f} ±{ci['half_width']:.3f}"
    return f"{mean:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(args.results_dir.glob("*.json"))
    files = [f for f in files if not f.name.startswith("_")]
    if not files:
        print(f"no result files in {args.results_dir}")
        return

    bootstrap = load_bootstrap()
    tiers = id_to_tier(REPO / "data")

    rows = []
    for fp in files:
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"warning: skipping {fp.name} (invalid JSON: {e})")
            continue
        model = d.get("model", fp.stem)
        row: dict = {"model": model}
        boot = bootstrap.get(model, {})
        for t in TASK_ORDER:
            metric_key, _ = HEADLINE[t]
            tr = d.get("tasks", {}).get(t, {})
            summ = tr.get("summary", {})
            row[t] = summ.get(metric_key)
            row[f"{t}_ci"] = boot.get(t)
            row[f"{t}_chrf"] = summ.get("chrf")          # floor for judge tasks
            row[f"{t}_judge_sonnet"] = summ.get("judge_sonnet_norm")
        row["char_preserved"] = (
            d.get("tasks", {}).get("punctuate", {})
            .get("summary", {}).get("char_preserved"))
        row["_missing"] = [t for t in TASK_ORDER if row[t] is None]
        row["_avg"] = (boot.get("_avg") or {}).get("mean")
        if row["_avg"] is None:
            vals = [row[t] for t in TASK_ORDER if row[t] is not None]
            row["_avg"] = round(sum(vals) / len(vals), 4) if vals else None
        row["_avg_ci"] = boot.get("_avg")

        # canonicity-stratified avg: per task, mean PRIMARY over items of
        # tier k, then mean across tasks (same shape as the headline Avg).
        by_tier: dict[int, list[float]] = {1: [], 2: [], 3: []}
        for t in TASK_ORDER:
            mk = HEADLINE[t][0]
            buckets: dict[int, list[float]] = {1: [], 2: [], 3: []}
            for it in d.get("tasks", {}).get(t, {}).get("items", []):
                tk = tiers.get(it["id"])
                v = (it.get("scores") or {}).get(mk)
                if tk in buckets and isinstance(v, (int, float)):
                    buckets[tk].append(float(v))
            for k in (1, 2, 3):
                if buckets[k]:
                    by_tier[k].append(sum(buckets[k]) / len(buckets[k]))
        for k in (1, 2, 3):
            row[f"_t{k}"] = (round(sum(by_tier[k]) / len(by_tier[k]), 4)
                             if by_tier[k] else None)
        rows.append(row)

    rows.sort(key=lambda r: -(r["_avg"] or 0))

    # Significance tiers: walking down by mean, a model stays in the current
    # tier while its 95% Avg CI still overlaps the tier *leader*'s CI
    # (leader = highest-mean member). It starts a new, lower tier only when
    # its CI upper bound falls below the leader's CI lower bound — i.e. it is
    # statistically distinguishable from the tier leader. Models sharing a
    # letter are not significantly different at 95%.
    tier_letter, leader_lo = "A", None
    for r in rows:
        ci = r.get("_avg_ci") or {}
        lo, hi = ci.get("ci_lo"), ci.get("ci_hi")
        if leader_lo is None:
            r["_sig"] = tier_letter
            leader_lo = lo
        elif lo is None or hi is None:
            r["_sig"] = "—"
        elif hi < leader_lo:                       # clearly worse than leader
            tier_letter = chr(ord(tier_letter) + 1)
            r["_sig"] = tier_letter
            leader_lo = lo
        else:
            r["_sig"] = tier_letter

    lines: list[str] = []

    # ---- 1. Primary leaderboard --------------------------------------------
    lines.append("## Leaderboard (primary, 95% CI from item bootstrap)")
    lines.append("")
    lines.append("`translate`/`char-gloss` headline = **Claude Opus 4.7 LLM "
                 "judge** (0–1); chrF systematically under-rates synonymous "
                 "paraphrase and is reported as a labelled floor in the "
                 "transparency table below. `Preserve` = `punctuate` "
                 "char-preservation rate (fraction of items where the model "
                 "did not rewrite the text — a fidelity diagnostic "
                 "`translate` cannot express; see `docs/task-redundancy.md`).")
    lines.append("")
    lines.append("`Tier` = statistical significance band: models sharing a "
                 "letter have **overlapping 95% Avg CIs** (not significantly "
                 "different); a lower letter is significantly worse than its "
                 "tier leader. Treat within-tier order as noise.")
    lines.append("")
    headers = ["Tier", "Model"] + \
              [f"{t} ({HEADLINE[t][1]})" for t in TASK_ORDER] + \
              ["Preserve", "Avg"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        cells = [r.get("_sig", "—"), r["model"]]
        for t in TASK_ORDER:
            cells.append(fmt_with_ci(r[t], r[f"{t}_ci"]))
        cp = r["char_preserved"]
        cells.append(f"{cp:.3f}" if cp is not None else "—")
        avg_str = fmt_with_ci(r["_avg"], r["_avg_ci"])
        mark = " ⚠" if r["_missing"] else ""
        cells.append(f"**{avg_str}{mark}**" if r["_avg"] is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    incomplete = [(r["model"], r["_missing"]) for r in rows if r["_missing"]]
    if incomplete:
        note = "; ".join(f"`{m}` missing {','.join(ts)}"
                         for m, ts in incomplete)
        lines.append(f"⚠ Avg is the mean of *available* task headlines — "
                     f"not strictly comparable for: {note}. Excluding a "
                     f"low-scoring task (e.g. `compress`) inflates that "
                     f"model's Avg.")
        lines.append("")

    # ---- 2. Canonicity-stratified ------------------------------------------
    lines.append("## Canonicity-stratified — recall vs. competence")
    lines.append("")
    lines.append("Avg restricted to items whose source is core canon "
                 "(**T3**: 论语/史记/诗经 … memorized verbatim by every LLM), "
                 "well-known (**T2**), or obscure (**T1**: minor dynastic "
                 "histories / specialized 子). **Gap = T3 − T1** is the "
                 "recall-reliance signal: a large positive gap means the "
                 "model leans on having seen the famous text. Ranked by "
                 "**T1** — the contamination-robust ranking. Point estimates "
                 "(no CI); tier definitions in `scripts/canonicity.py`. "
                 "`idiom-source` ρ=0.68 dominates this effect "
                 "(`docs/contamination.md`).")
    lines.append("")
    ch = ["Model", "T3 canon", "T2", "T1 obscure", "Gap (T3−T1)"]
    lines.append("| " + " | ".join(ch) + " |")
    lines.append("|" + "|".join(["---"] * len(ch)) + "|")
    for r in sorted(rows, key=lambda r: -(r["_t1"] if r["_t1"] is not None
                                          else -1)):
        t1, t2, t3 = r["_t1"], r["_t2"], r["_t3"]
        gap = (f"{t3 - t1:+.3f}" if (t3 is not None and t1 is not None)
               else "—")
        lines.append("| " + " | ".join([
            r["model"],
            f"{t3:.3f}" if t3 is not None else "—",
            f"{t2:.3f}" if t2 is not None else "—",
            f"{t1:.3f}" if t1 is not None else "—",
            gap]) + " |")
    lines.append("")

    # ---- 3. Transparency: chrF floor + two-judge cross-check ---------------
    lines.append("## Transparency — chrF floor & two-judge cross-check "
                 "(translate / char-gloss)")
    lines.append("")
    lines.append("The judge headline above is **Opus**. Here it is shown "
                 "next to the reproducible **chrF** floor and the "
                 "independent **Sonnet** judge. Opus and Sonnet agreeing "
                 "(and both far above chrF) is the evidence that the judge "
                 "promotion is sound, not cherry-picked. See "
                 "`experiments/llm-judge/`.")
    lines.append("")
    th = ["Model",
          "translate chrF", "translate Opus", "translate Sonnet",
          "char-gloss chrF", "char-gloss Opus", "char-gloss Sonnet"]
    lines.append("| " + " | ".join(th) + " |")
    lines.append("|" + "|".join(["---"] * len(th)) + "|")
    for r in sorted(rows, key=lambda r: -(r["_avg"] or 0)):
        cells = [r["model"]]
        for t in JUDGE_TASKS:
            f = r.get(f"{t}_chrf")
            o = r.get(t)                       # PRIMARY = Opus judge_norm
            s = r.get(f"{t}_judge_sonnet")
            cells += [f"{f:.3f}" if f is not None else "—",
                      f"{o:.3f}" if o is not None else "—",
                      f"{s:.3f}" if s is not None else "—"]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    md = "\n".join(lines)
    print(md)
    if args.out:
        out_path = args.out if args.out.is_absolute() else (Path.cwd() / args.out)
        out_path.write_text(md + "\n", encoding="utf-8")
        try:
            shown = out_path.relative_to(REPO)
        except ValueError:
            shown = out_path
        print(f"\nwrote → {shown}")


if __name__ == "__main__":
    main()
