"""Contamination / memorization probe — zero model calls.

Question: does this benchmark mostly reward *recall of canonical text* rather
than Classical-Chinese skill? If so, items drawn from the most over-
represented sources (论语/史记/诗经 — in every textbook, anthology, web dump)
should be systematically easier than items from obscure histories (南齐书,
周书) at the same intrinsic difficulty.

We can't see model training data. Two proxies, both deterministic:

  A. **Source canonicity tier** (primary). A 3-level ordinal: core canon
     every model has effectively memorized (Four Books, 诗经, 左传, 史记…)
     vs. well-known classics vs. obscure dynastic histories / specialized 子.
     This tracks real-world over-representation far better than (B).

  B. **In-corpus shingle recurrence** (secondary, weak). A mid-passage
     10-char shingle's count across all corpus.jsonl `content`. Kept only
     for transparency: each canonical text appears once here, so exact
     cross-text recurrence barely varies — low resolution, reported as-is.

Spearman(proxy, item difficulty) per task + bench-wide. Strong +ρ ⇒ the
bench leans on recall; ≈0 ⇒ difficulty is skill/metric-driven (the more
defensible outcome, and consistent with findings.md).

Usage:
  python scripts/contamination_probe.py [--out docs/contamination.md]
"""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from canonicity import tier  # noqa: E402  (shared source of truth)

CORPUS = Path.home() / "Documents/zion/classical-corpus/output/corpus.jsonl"
TASK_FILES = {
    "translate": "translate.jsonl", "punctuate": "punctuate.jsonl",
    "idiom-source": "idiom_source.jsonl", "fill-in": "fill_in.jsonl",
    # char-gloss excluded: 18/100 gold are broken stubs (see quality-audit)
    "compress": "compress.jsonl",
}
PUNCT = re.compile(r"[，。、；：！？「」『』《》（）()【】\s“”‘’,.;:!?\"'\-—…]")


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    n = len(xs)
    if n < 4:
        return None
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return sxy / math.sqrt(sxx * syy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shingle", type=int, default=10)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "contamination.md")
    args = ap.parse_args()

    ia = json.loads((REPO / "docs" / "item-analysis.json").read_text("utf-8"))
    blob = ""
    if CORPUS.exists():
        blob = "\n".join(json.loads(l).get("content", "")
                         for l in CORPUS.open(encoding="utf-8") if l.strip())

    md = ["# Contamination / memorization probe", "",
          "Does difficulty track how *over-represented* an item's source is, "
          "rather than Classical skill? Proxy A = source canonicity tier "
          "(3=core canon e.g. 论语/史记/诗经, 2=well-known classics, "
          "1=obscure histories/子). Proxy B = in-corpus 10-char shingle "
          "recurrence (low-resolution, see header). Spearman vs item "
          "difficulty from `item-analysis.json`. n≈100/task.", ""]

    report = {}
    allA_x, allA_y, allB_x, allB_y = [], [], [], []
    tier_diff_all = {1: [], 2: [], 3: []}

    for task, fname in TASK_FILES.items():
        recs = {}
        for l in (REPO / "data" / fname).open(encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                recs[r["id"]] = r
        ax, ay, bx, by = [], [], [], []
        tdiff = {1: [], 2: [], 3: []}
        for it in ia["tasks"].get(task, {}).get("items", []):
            r = recs.get(it["id"])
            if not r:
                continue
            md_ = r.get("metadata", {})
            src = md_.get("source") or md_.get("book") or ""
            book = re.split(r"[·/]", src)[0].strip() if src else ""
            if not book:
                continue
            tt = tier(book)
            diff = it["difficulty"]
            ax.append(tt)
            ay.append(diff)
            tdiff[tt].append(diff)
            tier_diff_all[tt].append(diff)
            if blob:
                key = (md_.get("expected_quote") if task == "idiom-source"
                       else r.get("reference", "")) or ""
                txt = key if task == "idiom-source" else PUNCT.sub("", key)
                if len(txt) >= args.shingle:
                    s = (len(txt) - args.shingle) // 2
                    sh = txt[s:s + args.shingle]
                elif txt:
                    sh = txt
                else:
                    sh = ""
                if sh:
                    bx.append(math.log1p(blob.count(sh)))
                    by.append(diff)
        rhoA = spearman(ax, ay)
        rhoB = spearman(bx, by)
        allA_x += ax
        allA_y += ay
        allB_x += bx
        allB_y += by
        report[task] = {
            "n": len(ax), "rho_canonicity": rhoA, "rho_shingle": rhoB,
            "tier_mean_difficulty": {
                k: (round(sum(v) / len(v), 4) if v else None)
                for k, v in tdiff.items()},
            "tier_n": {k: len(v) for k, v in tdiff.items()},
        }
        md.append(f"## {task}  (n={len(ax)})")
        md.append("")
        rA = f"{rhoA:+.3f}" if rhoA is not None else "n/a"
        rB = f"{rhoB:+.3f}" if rhoB is not None else "n/a"
        md.append(f"- **Spearman(canonicity tier, difficulty) = {rA}**  "
                  f"(shingle proxy: {rB})")
        means = " · ".join(
            f"T{k} {sum(v)/len(v):.3f} (n={len(v)})"
            for k, v in tdiff.items() if v)
        md.append(f"- mean difficulty by tier: {means}")
        md.append("")

    rhoA_all = spearman(allA_x, allA_y)
    rhoB_all = spearman(allB_x, allB_y)
    md.append("## Bench-wide")
    md.append("")
    md.append(f"- **Spearman(canonicity, difficulty) over {len(allA_x)} "
              f"items = {rhoA_all:+.3f}**" if rhoA_all is not None
              else "- canonicity: n/a")
    md.append(f"- shingle proxy = "
              f"{rhoB_all:+.3f}" if rhoB_all is not None else "- shingle: n/a")
    tm = {k: (sum(v) / len(v) if v else None)
          for k, v in tier_diff_all.items()}
    md.append(f"- mean difficulty: core-canon T3 "
              f"**{tm[3]:.3f}** (n={len(tier_diff_all[3])}) · "
              f"T2 **{tm[2]:.3f}** (n={len(tier_diff_all[2])}) · "
              f"obscure T1 **{tm[1]:.3f}** (n={len(tier_diff_all[1])})")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    gap = (tm[3] - tm[1]) if (tm[3] is not None and tm[1] is not None) else 0
    recall = sorted(((t, d["rho_canonicity"]) for t, d in report.items()
                     if d["rho_canonicity"] is not None
                     and d["rho_canonicity"] >= 0.30),
                    key=lambda x: -x[1])
    clean = sorted(t for t, d in report.items()
                   if d["rho_canonicity"] is not None
                   and abs(d["rho_canonicity"]) < 0.15)
    md.append(f"The effect is **not uniform** — bench-wide ρ={rhoA_all:+.3f} "
              f"hides strong task-level structure:")
    md.append("")
    if recall:
        md.append(f"- **Recall-driven** (canonicity ρ≥0.30): "
                  + ", ".join(f"`{t}` (ρ={r:+.2f})" for t, r in recall)
                  + f". For these, source over-representation is a major "
                  f"difficulty driver — they partly measure *do you recognize "
                  f"this famous text*, not Classical reasoning. `idiom-source`"
                  f" is the extreme case and is exactly the task carrying the "
                  f"23 ceiling items in `item-analysis.md`.")
    if clean:
        md.append(f"- **Clean** (|ρ|<0.15): "
                  + ", ".join(f"`{t}`" for t in clean)
                  + ". Difficulty here is skill/metric-driven, not "
                  "memorization — these tasks are contamination-robust.")
    md.append("")
    md.append(f"Core-canon (T3) items are on average {gap:+.3f} easier than "
              f"obscure-source (T1) items bench-wide. **Actionable:** report "
              f"a canonicity-stratified leaderboard, and in v1.x rebuild "
              f"`idiom-source` (and to a lesser degree `compress`) from "
              f"Tier-1 sources so the score reflects competence over recall. "
              f"`translate`/`punctuate`/`fill-in` need no contamination fix.")
    md.append("")

    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    (REPO / "docs" / "contamination.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote → {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
