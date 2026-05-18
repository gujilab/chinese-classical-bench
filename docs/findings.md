# Findings: what a psychometric audit revealed about this benchmark

*2026-05-16. 10 models, 6 tasks × 100 questions, all analysis retroactive over
stored predictions (zero new model calls). Numbers from
[`item-analysis.md`](item-analysis.md), [`quality-audit.md`](quality-audit.md),
and [`../leaderboard.md`](../leaderboard.md).*

## TL;DR

Treating a benchmark's own stored predictions as data — item difficulty,
item–total discrimination, and metric cross-checks — is enough to find
problems that grep-style validation misses. For this benchmark it surfaced
one task whose gold is partly broken, quantified how badly chrF mis-measures
two tasks, and showed which tasks carry independent signal. None of it
required re-running a model.

## 1. `char-gloss` is broken at the gold level, not the model level

`char-gloss` has the lowest headline of all six tasks (chrF ≈ 0.12–0.21). The
naive reading is "models are bad at glossing Classical characters." The
psychometrics say otherwise:

- **27 of 100 items are floor items** — every one of the 10 models scores
  ≈0, so the item separates nothing.
- **18 of those 27 have gold = `"同本义。"`** — a dictionary cross-reference
  stub ("same as the original meaning", defined elsewhere), not a usable
  gloss. The build script sampled the placeholder sense instead of resolving
  it. Models produce *correct* glosses (`留宿`, `香气`, `从上取物`) and chrF
  against `同本义。` returns ≈0 for all of them.

This is the highest-value finding because it inverts the conclusion: the
char-gloss headline is not a measurement of model ability, it is an artifact
of broken gold plus a metric that cannot see paraphrase. The 18 items are now
flagged `metadata._audit_issue`; they are not deleted (stored results stay
comparable) but downstream consumers can filter them.

## 2. chrF and the LLM judge disagree by 4–5× on the open-ended tasks

On the two free-text tasks, the headline chrF and the (already-stored,
two-judge cross-validated) LLM-judge score tell completely different stories:

- **char-gloss**: chrF 0.12–0.21 vs judge 0.54–0.74.
- **translate**: chrF 0.20–0.24 vs judge 0.68–0.80.

chrF rewards literal character-n-gram overlap; a correct Classical gloss or
translation phrased differently from the reference scores near zero. The gap
is not a constant offset — it reorders models (e.g. on char-gloss `glm-5`
edges `minimax-m2.1` by chrF but trails it by judge). **chrF on these two
tasks is not a valid ranking signal on its own.** It remains useful only as a
cheap, reproducible floor; the judge-rescored table is the one to read. This
empirically confirms the long-suspected "chrF too strict on paraphrase"
concern with concrete numbers rather than intuition.

## 3. Scorer hygiene: a silent traditional-script penalty

`score_fill_in` did not apply the traditional→simplified normalization that
`idiom-source` already uses. Models answering in traditional script (e.g.
`饑` for gold `饥`) were marked wrong on a correct answer. Fixed and rescored
retroactively across all 10 models: opus-4-7 0.84→0.86,
opus-4-7-thinking 0.87→0.88, glm-5 0.44→0.45. Small in aggregate, but it is
pure measurement error and it was invisible until item-level inspection
showed strong models "failing" items weak models "passed." (Two latent
`rescore.py` bugs were fixed in passing — see quality-audit.)

## 4. Task structure: one redundant pair, one genuinely orthogonal task

Spearman correlation of the six task scores across the 10 models (n=10,
directional only):

- `punctuate` ↔ `translate` ρ = **+0.88** — the only pair above 0.8. They
  largely rank models the same way; a v2 could justify dropping or merging
  one, or keep both only if the *failure modes* differ (worth a follow-up).
- `compress` ↔ `char-gloss` ρ = **−0.02**, `compress` ↔ `fill-in` ρ =
  **+0.03** — `compress` (modern→Classical compression, the newest task) is
  near-orthogonal to the rest. It is the task that adds the most independent
  information; adding it was the right call.
- Everything else sits in a moderate 0.3–0.6 band: correlated (all tap
  Classical competence) but not redundant.

## 5. Difficulty distribution: the bench is bimodal, not calibrated

- `idiom-source` has **23 ceiling items** (every model solves) and 13 floor
  — only ~64% of items do any discriminating work.
- `translate`, `char-gloss`, `compress` are floor-heavy (mean difficulty
  0.23 / 0.16 / 0.12), partly real difficulty, partly metric artifact (§2).
- The healthiest tasks by item–total discrimination are `fill-in` (0.45) and
  `idiom-source` (0.42); `translate` is weakest (0.17) — again, mostly chrF.

A well-calibrated benchmark wants items clustered around 50% solve rate. This
one is bimodal (trivial or impossible), which inflates variance and wastes
question budget.

## 6. Contamination is real — but only on `idiom-source` (and a bit `compress`)

Full method and numbers: [`contamination.md`](contamination.md). Proxy:
source-canonicity tier (3 = core canon like 论语/史记/诗经, massively
over-represented in any training corpus; 1 = obscure dynastic histories).
Spearman(canonicity, item difficulty):

- `idiom-source` **ρ=+0.68** — core-canon items mean difficulty **0.806**
  vs obscure-source **0.069**. This task is largely a *do-you-recognize-this-
  famous-allusion* recall test, and it is exactly the task carrying the 23
  ceiling items from §5. The two findings are the same finding.
- `compress` ρ=+0.42 — moderately recall-influenced (compressing a familiar
  passage is easier).
- `translate` / `punctuate` / `fill-in` **ρ≈0.06–0.08** — contamination-
  robust; difficulty there is skill/metric, not memorization.

So "is this benchmark just measuring memorization?" has a precise answer:
**no for 4 of 6 tasks, substantially yes for `idiom-source`.** Bench-wide
ρ=+0.34 is a misleading average — the honest statement is per-task.

## 7. The one redundant-looking task pair isn't redundant

Full analysis: [`task-redundancy.md`](task-redundancy.md). `punctuate`↔
`translate` ρ=0.88 (§4) is **not** redundancy — at n=10 with a wide ability
spread, even `punctuate`'s `char_preserved` sub-metric (a dimension
`translate` cannot express) correlates ρ=0.86 with translate. Inter-task ρ
has no power to detect redundancy at this n. The construct test settles it:
the hardest `translate` items are 经/子/集 (semantic transfer on dense
philosophy/literature); the hardest `punctuate` items are **14/15 from 史**
(boundary segmentation on histories). Disjoint material, different skills —
keep both. Bonus diagnostic: the minimax models silently rewrite **38/100**
inputs on `punctuate` (a fidelity failure `translate` can't surface);
`char_preserved` should be a labelled leaderboard column.

## Honest scope and limitations

- **n = 10 models, 100 questions/task.** Discrimination and task-correlation
  numbers are directional, not significant. Every section says so.
- **Excluding all 31 audit-flagged items** moves the bench-wide dead-item
  rate only 18% → 14% and leaves mean discrimination flat (0.317 → 0.319).
  Bad gold explains the *char-gloss floor specifically*, **not** the
  bench-wide dead mass. The remaining dead items are genuine
  ceiling/floor and need *harder replacement items*, not relabeling — do not
  oversell the audit.
- **The LLM judge is not ground truth.** It is two-model cross-validated
  (Opus + Sonnet agree), which is a reasonable proxy for these tasks but not
  a substitute for human adjudication on the contested items.

## Implications for v1.x

Priority order, now that the follow-ups have run:

1. **`idiom-source` is the worst task and for two compounding reasons** — 23
   ceiling items (§5) *and* ρ=0.68 contamination (§6), the same problem: it
   samples famous canon. ~~Status: blocked (source deleted)~~ **UNBLOCKED &
   STAGED.** The deleted dictionary has a public CC0 equivalent
   (`pwxcoo/chinese-xinhua` `idiom.json`, pinned by commit + sha256 in
   `scripts/fetch_idiom_source_data.py`). `scripts/build_idiom_source_v2.py`
   parses each idiom's `derivation` (出处) for book + quote and resamples a
   full **100-item Tier-1-only** set → `data/idiom_source.v2.jsonl`.
   Result: tier mix `{1: 100}` (contamination ρ≈0 *by construction* — every
   source is an obscure dynastic history or 说文, none core canon), balanced
   across 11 books, **0 idiom overlap** with v1, gold = the verbatim
   derivation citation (verifiable, not a mechanical guess — far higher
   confidence than the char-gloss candidates). **Not yet adopted**: swapping
   the item set invalidates the stored predictions, so it needs a scoped
   `idiom-source` rerun (cost) — but the deterministic, auditable generation
   half is done and the recall-vs-competence flaw is structurally fixed in
   the staged set.
2. **`char-gloss` 18 circular-gold items**: ~~10 mechanical 说文 + 8
   blocked~~ **UPGRADED to 18/18 verified.** The same CC0 source
   (`chinese-xinhua/word.json`, pinned) states each char's 本义 in a
   parenthetical — which is exactly what the stub "同本义" pointed at, in
   modern Chinese matching the task's answer style (e.g. 罗→用绳线结成的捕鸟网,
   僮→未成年的人, 捎→选取). `data/char_gloss.candidates.jsonl` now has 18/18
   `_status: candidate-verified`, 0 blocked (说文 kept only as a low-
   confidence fallback). Still **not adopted**: a few are 本义 vs the quote's
   contextual sense, so they want a judge pass, and swapping items needs a
   scoped char-gloss rerun. Until then the `_audit_issue` filter is the fix.
3. ~~**Report a canonicity-stratified leaderboard**~~ **DONE**. `leaderboard.md`
   now has a T3/T2/T1 + recall-gap section ranked by the obscure-source (T1)
   column. New result: the recall gap (T3−T1) reorders the board —
   `claude-opus-4-7` has the **largest** gap (+0.220) and drops from #1
   overall to #3 on obscure sources; `Qwen3.5`/`glm-5` lead the
   contamination-robust ranking. Recall reliance is itself model-
   discriminating.
4. ~~**Promote the judge-rescored table to primary**~~ **DONE**. `translate`/
   `char-gloss` headline is now the Opus judge with judge-aware bootstrap
   CIs; chrF is a labelled reproducible floor in a transparency table beside
   the independent Sonnet judge (the two judges agree within ~0.02 and sit
   ~0.5 above chrF — the promotion is not cherry-picked).
5. ~~**Surface `char_preserved`**~~ **DONE**. Added as a labelled `Preserve`
   column. Both `punctuate` and `translate` kept (redundancy resolved, §7).

Net: items 3–5 shipped (free, no rerun); items 1–2 remain correctly
*blocked on funding or source recovery* — saying so plainly beats shipping
fabricated replacements. Also fixed in passing: Avg now flags models with
missing tasks (`Qwen3.5` lacks `compress`, which inflated its Avg).

A residual methodology note worth a future pass: the `Avg` still averages
*available* task headlines, so an incomplete model is not strictly
comparable — a stricter leaderboard would require all 6 tasks or impute.
