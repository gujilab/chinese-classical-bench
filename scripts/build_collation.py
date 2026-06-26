"""Build collation task: 校勘 (textual collation) questions.

Tests whether the model can parse classical-Chinese textual-criticism
conventions (一作 / 俗本作 / 当作 / 讹作 / 误作 …) and identify the
canonical reading.

Rules per marker (only single-character variants are kept for the prototype):
  X一作Y    → main reading is X, variant Y in some edition       → canonical = X
  X一本作Y  → same                                                → canonical = X
  X或作Y    → same                                                → canonical = X
  X又作Y    → same                                                → canonical = X
  X别作Y    → same                                                → canonical = X
  X俗本作Y  → 俗本 is non-canonical                                 → canonical = X
  X旧本作Y  → 旧本 is non-canonical                                 → canonical = X
  X讹作Y    → reports a miscopy; main reading is X                 → canonical = X
  X当作Y    → received X should be Y (emendation)                  → canonical = Y
  X当为Y    → same                                                → canonical = Y
  X应作Y    → same                                                → canonical = Y
  X应为Y    → same                                                → canonical = Y

Question form:
  • input  = a 60–120 char window around the marker (the 【...】 note kept)
  • reference = the canonical single character
  • metadata.marker / metadata.variant / metadata.correct kept for analysis
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS = Path.home() / "Documents/zion/classical-corpus/output/corpus.jsonl"
OUT = REPO / "data" / "collation.jsonl"

N_TARGET = 50
WIN = 50  # half-window of chars around the marker

# Each rule: regex (X = group 1, marker, Y = group 2), canonical side ('X'|'Y'), marker label.
RULES: list[tuple[str, str, str]] = [
    (r"([一-鿿])(一作)([一-鿿])", "X", "一作"),
    (r"([一-鿿])(一本作)([一-鿿])", "X", "一本作"),
    (r"([一-鿿])(或作)([一-鿿])", "X", "或作"),
    (r"([一-鿿])(俗本作)([一-鿿])", "X", "俗本作"),
    (r"([一-鿿])(旧本作)([一-鿿])", "X", "旧本作"),
    (r"([一-鿿])(讹作)([一-鿿])", "X", "讹作"),
    (r"([一-鿿])(当作)([一-鿿])", "Y", "当作"),
    (r"([一-鿿])(当为)([一-鿿])", "Y", "当为"),
]
# Markers dropped from the prototype (low precision, too few clean candidates after filtering):
# 又作 (mostly "also did/wrote"), 别作 ("separately do"), 应作 / 应为 ("should be" in argument).

# 当作/当为 outside of 【】 notes are mostly false positives ("当作" = "should do …").
# For these markers, require the match to be inside a 三家注 bracket.
EMENDATION_MARKERS = {"当作", "当为", "应作", "应为"}

# These markers are polysemous in classical Chinese:
#   当作 = "should be written as" (collation) | "treat as / do" (regular)
#   当为 = "should be written as" (collation) | "should be" (regular)
#   又作 = "another reading is" (collation)   | "also wrote/did" (regular)
#   别作 = "alternatively read as" (collation)| "separately make/do" (regular)
# For these, require a collation context cue (案/疑/讹/误/字/本/旧/监/非/误也) within ±20 chars.
AMBIGUOUS_MARKERS = {"当作", "当为"}
# Strong collation-context cues only — bare "旧" / "字" are too common and let through
# regular prose like "旧泚传曰" or "兼宰相字" as false positives.
COLLATION_CUES = [
    "案", "疑", "讹", "监本", "非是", "者误", "误也", "之误", "误耳", "字误",
    "校", "改正", "脱字", "脱文", "衍文", "本作", "本误", "误字",
]

# Characters that appear before a collation marker as part of the commentary apparatus
# (e.g. 徐广曰一作Y, 本或作Y, 案当作Y) rather than as the canonical reading itself.
# When the regex matches one of these as X, the real canonical char lives elsewhere
# (typically the main-text char before the surrounding 【】 bracket) — skip these matches.
NOISE_X = set("曰本案言谓作又云说也故即则当应注是非有无如以")


def _in_bracket(text: str, idx: int) -> bool:
    """True iff text[idx] sits inside a 【 ... 】 block."""
    open_pos = text.rfind("【", 0, idx)
    close_pos = text.rfind("】", 0, idx)
    return open_pos > close_pos


def harvest(rec: dict) -> list[dict]:
    content = rec.get("content", "")
    if not isinstance(content, str) or len(content) < 60:
        return []
    out: list[dict] = []
    for pattern, side, marker in RULES:
        for m in re.finditer(pattern, content):
            x, _, y = m.group(1), m.group(2), m.group(3)
            if x == y:
                continue
            # Sanity: variant must look like a meaningful character, drop obvious noise (numerals etc.).
            if not all(re.match(r"[一-鿿]", c) for c in (x, y)):
                continue
            mid = m.start()
            if marker in EMENDATION_MARKERS and not _in_bracket(content, mid):
                continue
            # Skip if the canonical side is one of the commentary-apparatus noise chars
            # (e.g. matching "曰一作Y" → X=曰 isn't a real reading).
            canonical_char = x if side == "X" else y
            other_char = y if side == "X" else x
            if canonical_char in NOISE_X or other_char in NOISE_X:
                continue
            # Polysemous markers require an explicit collation-context cue nearby.
            if marker in AMBIGUOUS_MARKERS:
                ctx_start = max(0, mid - 20)
                ctx_end = min(len(content), m.end() + 20)
                ctx = content[ctx_start:mid] + content[m.end():ctx_end]
                if not any(cue in ctx for cue in COLLATION_CUES):
                    continue
            window_start = max(0, mid - WIN)
            window_end = min(len(content), m.end() + WIN)
            window = content[window_start:window_end]
            canonical = x if side == "X" else y
            variant = y if side == "X" else x
            out.append(
                {
                    "source": rec.get("source", ""),
                    "rec_id": rec.get("id", ""),
                    "marker": marker,
                    "canonical": canonical,
                    "variant": variant,
                    "window": window,
                    "marker_span": (mid - window_start, m.end() - window_start),
                    "category": rec.get("category", ""),
                }
            )
    return out


def main() -> None:
    rng = random.Random(42)
    pool_by_marker: dict[str, list[dict]] = defaultdict(list)
    seen_windows: set[str] = set()

    print(f"reading {CORPUS}...")
    n_records = 0
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n_records += 1
            for cand in harvest(rec):
                # de-dup by window content (some notes repeat verbatim in different volumes)
                if cand["window"] in seen_windows:
                    continue
                seen_windows.add(cand["window"])
                pool_by_marker[cand["marker"]].append(cand)

    total = sum(len(v) for v in pool_by_marker.values())
    print(f"scanned {n_records} records, harvested {total} candidates across {len(pool_by_marker)} markers")
    for m, lst in sorted(pool_by_marker.items(), key=lambda kv: -len(kv[1])):
        print(f"  {m:8} = {len(lst):>5}")

    # Stratified sample: target 50 across markers, weight ~ sqrt(pool) so big markers
    # don't dominate but rare ones still appear.
    import math

    weights = {m: math.sqrt(len(lst)) for m, lst in pool_by_marker.items() if lst}
    wsum = sum(weights.values())
    targets = {m: max(1, round(N_TARGET * w / wsum)) for m, w in weights.items()}
    # rebalance to exact N
    diff = N_TARGET - sum(targets.values())
    keys_by_size = sorted(targets, key=lambda k: -len(pool_by_marker[k]))
    i = 0
    while diff != 0:
        k = keys_by_size[i % len(keys_by_size)]
        if diff > 0:
            targets[k] += 1
            diff -= 1
        elif targets[k] > 1:
            targets[k] -= 1
            diff += 1
        i += 1

    picked: list[dict] = []
    for m, t in targets.items():
        lst = pool_by_marker[m]
        picks = rng.sample(lst, min(t, len(lst)))
        picked.extend(picks)
    rng.shuffle(picked)
    picked = picked[:N_TARGET]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, c in enumerate(picked, 1):
            window = c["window"].strip()
            instruction = (
                "下列古文出自传世史书三家注，其中含有校勘异文标记。"
                "请按校勘学惯例（一作 / 俗本作 / 当作 / 讹作 …）"
                "判断主文中的正字（仅一字），直接给出该字。"
            )
            item = {
                "id": f"collation#{i}",
                "task": "collation",
                "instruction": instruction,
                "input": window,
                "reference": c["canonical"],
                "metadata": {
                    "source": c["source"],
                    "category": c["category"],
                    "marker": c["marker"],
                    "variant": c["variant"],
                    "rec_id": c["rec_id"],
                },
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"wrote {len(picked)} items to {OUT}")

    # Marker distribution of the actual sample
    final_dist = defaultdict(int)
    for c in picked:
        final_dist[c["marker"]] += 1
    print("final sample by marker:")
    for m, n in sorted(final_dist.items(), key=lambda kv: -kv[1]):
        print(f"  {m:8} = {n}")


if __name__ == "__main__":
    main()
