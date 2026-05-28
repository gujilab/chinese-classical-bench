# Trimmed Leaderboard (discriminating items only)

Excluded **189** items with discrimination ≤ 0 or dead variance.

Remaining items per task: translate 72, punctuate 73, char-gloss 54, idiom-source 59, fill-in 71, compress 82


| Tier | Model | translate | punctuate | char-gloss | idiom-source | fill-in | compress | Avg |
|---|---|---|---|---|---|---|---|---|
| A | claude-opus-4-7 | 0.811 | 0.897 | 0.778 | 0.678 | 0.944 | 0.170 | **0.713 ±0.027** |
| A | claude-opus-4-7-thinking | 0.817 | 0.888 | 0.804 | 0.644 | 0.972 | 0.080 | **0.701 ±0.026** |
| B | claude-sonnet-4-6 | 0.800 | 0.869 | 0.744 | 0.492 | 0.718 | 0.178 | **0.634 ±0.032** |
| B | deepseek-3.2 | 0.778 | 0.816 | 0.556 | 0.831 | 0.493 | 0.179 | **0.609 ±0.032** |
| B | glm-5 | 0.750 | 0.901 | 0.670 | 0.831 | 0.324 | 0.167 | **0.607 ±0.030** |
| B | minimax-m2.1 | 0.703 | 0.765 | 0.756 | 0.678 | 0.634 | 0.097 | **0.605 ±0.034** |
| B | Qwen3.5-35B-A3B | 0.739 | 0.826 | 0.637 | 0.407 | 0.254 | — | **0.572 ±0.038** |
| B | minimax-m2.5 | 0.700 | 0.765 | 0.700 | 0.525 | 0.549 | 0.096 | **0.556 ±0.034** |
| B | qwen3-coder-next | 0.753 | 0.859 | 0.585 | 0.475 | 0.451 | 0.119 | **0.540 ±0.034** |
| C | claude-haiku-4-5-20251001 | 0.683 | 0.794 | 0.570 | 0.153 | 0.225 | 0.090 | **0.419 ±0.029** |

## Key finding

Removing noisy items (32% of bench) splits the original 2-tier ranking into 3 tiers:
- **A**: Opus models separate significantly from the pack
- **B**: Sonnet + all open-source models form a broad middle tier
- **C**: Haiku remains significantly below

The B-tier internal spread widens from 0.048 → 0.069 but CIs still overlap,
confirming that N→200 expansion is needed to resolve B-tier ordering.
