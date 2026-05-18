## Leaderboard (primary, 95% CI from item bootstrap)

`translate`/`char-gloss` headline = **Claude Opus 4.7 LLM judge** (0–1); chrF systematically under-rates synonymous paraphrase and is reported as a labelled floor in the transparency table below. `Preserve` = `punctuate` char-preservation rate (fraction of items where the model did not rewrite the text — a fidelity diagnostic `translate` cannot express; see `docs/task-redundancy.md`).

`Tier` = statistical significance band: models sharing a letter have **overlapping 95% Avg CIs** (not significantly different); a lower letter is significantly worse than its tier leader. Treat within-tier order as noise.

| Tier | Model | translate (Judge) | punctuate (Punct F1) | char-gloss (Judge) | idiom-source (Book EM) | fill-in (Exact) | compress (Compress Eff) | Preserve | Avg |
|---|---|---|---|---|---|---|---|---|---|
| A | claude-opus-4-7 | 0.800 ±0.041 | 0.800 ±0.063 | 0.716 ±0.050 | 0.650 ±0.090 | 0.860 ±0.065 | 0.147 ±0.034 | 0.820 | **0.662 ±0.024** |
| A | claude-opus-4-7-thinking | 0.802 ±0.043 | 0.790 ±0.065 | 0.736 ±0.044 | 0.630 ±0.095 | 0.880 ±0.065 | 0.091 ±0.026 | 0.820 | **0.655 ±0.025** |
| A | claude-sonnet-4-6 | 0.776 ±0.047 | 0.785 ±0.063 | 0.694 ±0.049 | 0.560 ±0.100 | 0.700 ±0.090 | 0.163 ±0.022 | 0.740 | **0.613 ±0.027** |
| B | Qwen3.5-35B-A3B | 0.728 ±0.048 | 0.753 ±0.062 | 0.620 ±0.052 | 0.500 ±0.100 | 0.380 ±0.090 | — | 0.690 | **0.596 ±0.032 ⚠** |
| B | glm-5 | 0.748 ±0.045 | 0.799 ±0.063 | 0.638 ±0.055 | 0.740 ±0.085 | 0.450 ±0.100 | 0.153 ±0.018 | 0.790 | **0.588 ±0.026** |
| B | minimax-m2.1 | 0.704 ±0.052 | 0.709 ±0.072 | 0.695 ±0.060 | 0.660 ±0.090 | 0.630 ±0.095 | 0.094 ±0.010 | 0.620 | **0.582 ±0.028** |
| B | deepseek-3.2 | 0.754 ±0.045 | 0.745 ±0.071 | 0.538 ±0.060 | 0.740 ±0.085 | 0.550 ±0.095 | 0.163 ±0.019 | 0.770 | **0.582 ±0.029** |
| B | minimax-m2.5 | 0.704 ±0.047 | 0.709 ±0.068 | 0.654 ±0.057 | 0.550 ±0.095 | 0.590 ±0.095 | 0.092 ±0.009 | 0.620 | **0.550 ±0.029** |
| B | qwen3-coder-next | 0.746 ±0.046 | 0.767 ±0.063 | 0.602 ±0.055 | 0.540 ±0.100 | 0.520 ±0.095 | 0.113 ±0.011 | 0.660 | **0.548 ±0.028** |
| C | claude-haiku-4-5-20251001 | 0.675 ±0.049 | 0.729 ±0.062 | 0.578 ±0.058 | 0.340 ±0.090 | 0.350 ±0.090 | 0.087 ±0.009 | 0.720 | **0.460 ±0.028** |

⚠ Avg is the mean of *available* task headlines — not strictly comparable for: `Qwen3.5-35B-A3B` missing compress. Excluding a low-scoring task (e.g. `compress`) inflates that model's Avg.

## Canonicity-stratified — recall vs. competence

Avg restricted to items whose source is core canon (**T3**: 论语/史记/诗经 … memorized verbatim by every LLM), well-known (**T2**), or obscure (**T1**: minor dynastic histories / specialized 子). **Gap = T3 − T1** is the recall-reliance signal: a large positive gap means the model leans on having seen the famous text. Ranked by **T1** — the contamination-robust ranking. Point estimates (no CI); tier definitions in `scripts/canonicity.py`. `idiom-source` ρ=0.68 dominates this effect (`docs/contamination.md`).

| Model | T3 canon | T2 | T1 obscure | Gap (T3−T1) |
|---|---|---|---|---|
| Qwen3.5-35B-A3B | 0.656 | 0.393 | 0.546 | +0.110 |
| glm-5 | 0.633 | 0.426 | 0.514 | +0.119 |
| claude-opus-4-7 | 0.732 | 0.607 | 0.512 | +0.220 |
| claude-sonnet-4-6 | 0.663 | 0.579 | 0.507 | +0.155 |
| claude-opus-4-7-thinking | 0.696 | 0.598 | 0.507 | +0.188 |
| deepseek-3.2 | 0.643 | 0.435 | 0.491 | +0.151 |
| minimax-m2.1 | 0.616 | 0.370 | 0.466 | +0.150 |
| qwen3-coder-next | 0.599 | 0.561 | 0.455 | +0.144 |
| minimax-m2.5 | 0.566 | 0.364 | 0.445 | +0.121 |
| claude-haiku-4-5-20251001 | 0.461 | 0.304 | 0.434 | +0.027 |

## Transparency — chrF floor & two-judge cross-check (translate / char-gloss)

The judge headline above is **Opus**. Here it is shown next to the reproducible **chrF** floor and the independent **Sonnet** judge. Opus and Sonnet agreeing (and both far above chrF) is the evidence that the judge promotion is sound, not cherry-picked. See `experiments/llm-judge/`.

| Model | translate chrF | translate Opus | translate Sonnet | char-gloss chrF | char-gloss Opus | char-gloss Sonnet |
|---|---|---|---|---|---|---|
| claude-opus-4-7 | 0.244 | 0.800 | 0.780 | 0.213 | 0.716 | 0.700 |
| claude-opus-4-7-thinking | 0.242 | 0.802 | 0.770 | 0.207 | 0.736 | 0.706 |
| claude-sonnet-4-6 | 0.231 | 0.776 | 0.756 | 0.157 | 0.694 | 0.700 |
| Qwen3.5-35B-A3B | 0.225 | 0.728 | 0.732 | 0.175 | 0.620 | 0.630 |
| glm-5 | 0.241 | 0.748 | 0.750 | 0.176 | 0.638 | 0.644 |
| minimax-m2.1 | 0.216 | 0.704 | 0.708 | 0.173 | 0.695 | 0.685 |
| deepseek-3.2 | 0.240 | 0.754 | 0.738 | 0.139 | 0.538 | 0.554 |
| minimax-m2.5 | 0.219 | 0.704 | 0.688 | 0.161 | 0.654 | 0.662 |
| qwen3-coder-next | 0.227 | 0.746 | 0.746 | 0.116 | 0.602 | 0.592 |
| claude-haiku-4-5-20251001 | 0.204 | 0.675 | 0.682 | 0.128 | 0.578 | 0.574 |

