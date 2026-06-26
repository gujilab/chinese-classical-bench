# Findings — Recall-gap: a contamination-robust ranking for Classical Chinese LLMs

*Companion to [`leaderboard.md`](../leaderboard.md). Source data: 10 models × 6 tasks × 100 items each, deterministic re-scoring (no new model calls). 2026-05.*

## TL;DR

Most "Classical Chinese LLM" benchmarks measure two things that pretend to be one — actual ability to handle pre-modern Chinese, and prior exposure to canonical pre-modern text in pretraining. We separate them by **stratifying every benchmark item by canonicity tier** (T3 = core canon like 论语/史记/诗经 — virtually memorized by every modern LLM; T1 = obscure dynastic histories like 北齐书/陈书 / minor 子) and tracking **Gap = T3 − T1** as a per-model recall-reliance score.

Three things fall out:

1. **The ranking is not invariant.** Sorted by T1 (the contamination-robust axis), the open Chinese models lead: Qwen3.5-35B-A3B `0.546`, glm-5 `0.514` — above every Claude.
2. **Gap is itself a model property, not a benchmark property.** It ranges from `+0.027` (Haiku 4.5) to `+0.220` (Opus 4.7) — the same family — and is not predicted by Avg score. Haiku is the most competence-aligned ranker we tested even though it's the weakest model.
3. **`idiom-source` (the "name the book this allusion is from" task) is doing 70%+ of the contamination work.** ρ(canonicity-tier, item difficulty) = +0.68 on `idiom-source` and ≈+0.06–0.08 on the other tasks. Without `idiom-source`, the bench is contamination-robust by construction.

## Method

For every item in the benchmark we tag its source book with a canonicity tier:

- **T3** — core canon: 论语, 史记, 诗经, 大学, 中庸, 等。Assumed memorized verbatim by every multi-trillion-token pretraining corpus.
- **T2** — well-known but not omnipresent: 后汉书, 三国志, 春秋三传, parts of 资治通鉴, 等。
- **T1** — obscure dynastic histories or minor 子: 北齐书, 周书, 陈书, 隋书 部分章节, 等。Less likely to have been heavily upsampled.

Definitions are mechanical from `_source` and `book` metadata; see `scripts/canonicity.py`. We then report per-model **Avg restricted to T3 items**, **Avg restricted to T1 items**, and the **Gap**.

## Result

Sorted by **T1 obscure** (the contamination-robust column). Point estimates, n≈25–40 items per tier per model — directional, not significant.

- Qwen3.5-35B-A3B: T3 0.656 / T1 0.546 / **Gap +0.110**
- glm-5: T3 0.633 / T1 0.514 / **Gap +0.119**
- claude-opus-4-7: T3 0.732 / T1 0.512 / **Gap +0.220** ← #1 overall, but #3 obscure
- claude-sonnet-4-6: T3 0.663 / T1 0.507 / **Gap +0.155**
- claude-opus-4-7-thinking: T3 0.696 / T1 0.507 / **Gap +0.188**
- deepseek-3.2: T3 0.643 / T1 0.491 / **Gap +0.151**
- minimax-m2.1: T3 0.616 / T1 0.466 / **Gap +0.150**
- qwen3-coder-next: T3 0.599 / T1 0.455 / **Gap +0.144**
- minimax-m2.5: T3 0.566 / T1 0.445 / **Gap +0.121**
- claude-haiku-4-5: T3 0.461 / T1 0.434 / **Gap +0.027**

The top-overall model has the largest Gap. The smallest Gap belongs to the worst overall model — but Haiku's T1 (0.434) is within 0.012 of MiniMax m2.5's T1 (0.445), the difference at the canon-heavy end is dramatic (0.566 vs 0.461). Haiku catches up most when the task no longer rewards memorization.

## Interpretation

"Contamination" in LLM evaluation is usually treated as a binary defect — either the model has seen the test set or it hasn't. The data suggest a more useful framing: **every model has seen the canon; what differs is how much they lean on having seen it.**

- A model with high Gap is shifting weight onto retrieval of memorized text whenever it can. Useful in practice (a knowledgeable assistant), but it inflates apparent ability on any benchmark dominated by famous quotations.
- A model with low Gap performs about as well on obscure-source items as on canon — its score reflects classical-text *competence* more than *recall*. This is the regime you want a benchmark to measure.

The corollary is that **picking the headline differently picks the ranking differently** — both rankings are defensible, but they're answering different questions:

- *Which model knows the most about pre-modern Chinese?* → sort by **overall Avg** (Opus 4.7).
- *Which model is best at pre-modern Chinese in domains it hasn't been overtrained on?* → sort by **T1 Avg** (Qwen3.5-35B-A3B).

Most benchmarks publish only the first. We publish both.

## What this doesn't claim

- **Canonicity-tier is a proxy, not a measurement of training-corpus overlap.** It correlates with overlap but can't prove it. The honest claim is "items from obscure sources are harder to memorize" — verifying this against any specific model's training data is impossible from the outside.
- **The Gap signal lives on `idiom-source`.** On the other five tasks ρ(canonicity, difficulty) ≈ 0.06–0.08 — they are contamination-robust by construction. If you exclude `idiom-source` from the Avg, Gap collapses across all models. The finding above is best read as "this benchmark has one recall-heavy task and five competence-heavy tasks, and that asymmetry reveals the recall-reliance of each model."
- **n is small.** 10 models, 100 items/task. Tiers have 25–40 items in each. Gap differences below ~0.05 are noise.

## Why this matters

Canonicity stratification is a free, retroactive analysis: no new model calls, no human annotation, mechanical from a corpus's metadata. Any benchmark sampled from public literature with a clear canonicity gradient can do this. The cheapest contamination defense isn't building a private holdout (expensive, never reproducible) — it's reporting the score gradient across an exposure proxy you already have.

## Reproducing

```bash
git clone https://github.com/gujilab/chinese-classical-bench
cd chinese-classical-bench
python scripts/canonicity.py     # regenerates the stratified table
```

Definitions and item tagging: `scripts/canonicity.py`. Detailed numbers and the inter-task ρ matrix: [`findings.md`](findings.md) §6 and [`contamination.md`](contamination.md).
