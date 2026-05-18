# Data Quality Audit

> Last run: 2026-05-16 (psychometric pass). Prior pass: 2026-05-13.
> Method: programmatic checks + item-level discrimination analysis
> (`scripts/item_analysis.py`) + targeted inspection of every
> negative-discrimination and floor item against 10 models' stored predictions.

## 2026-05-16 — psychometric pass (the big one)

The item-level discrimination analysis surfaced a problem the 2026-05-13
grep-pass missed entirely: **18 `char-gloss` items have gold = `"同本义。"`**
— a dictionary placeholder ("same as the original meaning", defined
elsewhere), not a usable gloss. Every model scores chrF≈0 on these regardless
of answer quality (models produce correct glosses like `留宿`, `香气`,
`从上取物`). These are exactly 18 of char-gloss's 27 floor items and they drag
the whole task's headline (chrF 0.164, the lowest of 6 tasks).

Also confirmed by reading gold + predictions:
- `idiom-source#52` 经邦论道 — gold `隋书`, but the idiom canonically
  originates from `尚书·周官` ("论道经邦"); the strong models answered 尚书 and
  were marked wrong. **Disputed source**, flagged.
- `fill-in#19` 天_下民 — gold `降` (孟子 quoting 尚书); `佑` is an attested
  variant in transmitted 尚书 editions ("天佑下民"). **Ambiguous cloze**, flagged.

**Scorer fix (not a data fix):** `score_fill_in` did not apply the t2s
normalization that `idiom-source` already uses, so models answering in
traditional script (e.g. `饑` for gold `饥`, `fill-in#4`) scored 0 on a
correct answer. Fixed in `scorers.py`; `rescore.py` re-applied it to all 10
models retroactively (no new model calls). Net effect: opus-4-7 fill-in
0.84→0.86, opus-4-7-thinking 0.87→0.88, glm-5 0.44→0.45. Two latent
`rescore.py` bugs fixed in passing: it crashed on `_*.json` and would have
wiped the LLM-judge columns (now merges instead of clobbering).

**Honest scope (`docs/item-analysis.md`):** excluding all 31 flagged items
drops the bench-wide dead-item rate 18%→14% and leaves discrimination flat
(0.317→0.319). So bad gold explains the **char-gloss floor specifically**,
not the bench-wide dead mass — the remaining dead items (notably
idiom-source's 23 ceiling items) are genuinely too easy/too hard and need
harder *replacement* items, not relabeling. Tracked as a v1.x task.

**Follow-ups (same day, all zero-cost):**
- *Contamination probe* ([`contamination.md`](contamination.md)): difficulty
  vs source-canonicity. `idiom-source` ρ=+0.68 (recall-driven, = its 23
  ceiling items); `compress` +0.42; `translate`/`punctuate`/`fill-in` clean
  (≈0.06). Per-task, not the misleading bench-wide +0.34 average.
- *Redundancy* ([`task-redundancy.md`](task-redundancy.md)): `punctuate`↔
  `translate` ρ=0.88 is an n=10 ability artifact, not redundancy — hardest
  material is disjoint (punctuate 93% 史, translate 0% 史). Keep both.
- *char-gloss regeneration*: `data/char_gloss.candidates.jsonl` stages 10
  mechanical 说文-derived candidates + 8 blocked, via
  `scripts/regen_char_gloss_candidates.py`. **Not adopted** — 说文 gives 本义
  ≠ contextual sense; needs human/judge review + a scoped rerun. The
  original modern-gloss dictionary is gone.
- *idiom-source ceiling fix*: ~~blocked~~ **staged**. Source recovered from
  public CC0 `pwxcoo/chinese-xinhua` (pinned commit+sha256,
  `scripts/fetch_idiom_source_data.py`). `scripts/build_idiom_source_v2.py`
  → `data/idiom_source.v2.jsonl`: 100 Tier-1-only items (tier mix {1:100},
  contamination ρ≈0 by construction), 11 obscure books, 0 overlap with v1,
  gold = verbatim derivation citation. Not adopted in place (needs a scoped
  idiom-source rerun); v1 stays canonical until then.

Total flagged after this pass: **31 records** (18 new char-gloss circular +
2 new disputed + 11 from the 2026-05-13 pass below).

## Summary

Each flagged record is annotated in the data file with a
`metadata._audit_issue` field — contributors can filter:

```python
ds = ds.filter(lambda x: not x.get("metadata", {}).get("_audit_issue"))
```

Items are **not removed** so historical model results (`results/<model>.json`)
remain comparable. Scoring metric impact is minor (≤2 percentage points on
any single task across any model).

## By task

### translate (1/100)
- `translate#51` — input "新招四会化蒙化注化穆" (a list of place names from 南齐书·志/卷十四) is identical to reference. Models get a free chrF point by echoing input. Affects all models equally; minimal rank impact.

### punctuate (8/100) — the biggest concentration
The benchmark accidentally sampled 8 items from 校勘记 (textual commentary),
calendar/almanac tables, or chronological enumerations rather than narrative
prose. These reference texts use whitespace formatting instead of 。，、 etc.,
so punctuation-position F1 is meaningless on them.
- `punctuate#18` — input contains Western commas leaked from `<史部,正史类,后汉书>` metadata residue.
- `punctuate#6, #26, #34, #62` — references are 校勘记 (textual commentary), no narrative punctuation.
- `punctuate#22` — 历法表 (calendar/almanac tabular data).
- `punctuate#46` — enumeration of titled officials.
- `punctuate#83` — chronological list of earthquake records.

**Impact**: punct_f1 is dragged down ~5% for all models (the wrong-genre items have very low F1 regardless of the model's answer). A v2 of `punctuate` should filter source records by `category in {本纪, 列传, 子, 经}` and minimum prose density.

### char-gloss (1/100)
- `char-gloss#95` — character "制" in the quote "先王之制,大都不过参国之一" (左传·隐公元年, meaning *system/regulation*) has reference "又" which is incorrect. The gloss generator failed for this record (the `book` field is also empty). Should be `制度` or `规定`.

### idiom-source (1/100)
- `idiom-source#70` — for 妙语解颐 (《汉书·匡衡传》"匡说《诗》，解人颐"), the `metadata.expected_quote` stores the colloquial annotation "使人笑不能止也" instead of the actual source passage. This only affects the *secondary* `quote_chrf` metric; the **headline `book_em` metric is unaffected** (the book "汉书" is correct).

### fill-in (0/100)
No issues found.

## Method

Automated checks (`grep`-able, see `scripts/validate_results.py` for the result-file equivalent):
- Duplicate IDs, missing required fields (`input`, `reference`, `metadata`)
- `input == reference` (degenerate test)
- References too short for the task (excluding fill-in's single-char design)
- Punctuation in `punctuate` inputs (should be stripped) and absence in references
- Character overlap between `idiom-source` `input` and `expected_quote`
- `fill-in` reference must be a single CJK character
- Source/book distribution balance per task

Manual inspection confirmed each automatic flag before annotation.

## How to fix in v1.x

Each issue is fixable individually but doing so would invalidate the stored
predictions in `results/*.json`. Plan:

1. v1.1 of `data/*.jsonl` would **replace** flagged items with cleaner samples
   regenerated from `chinese-classical-corpus`.
2. Existing result files would need a **rerun** of the affected questions only
   (not full benchmark) to keep historical comparisons intact.
3. Once all flagged items are replaced, leaderboard numbers will likely move
   by ≤2% per task per model — directionally similar, slightly cleaner.
