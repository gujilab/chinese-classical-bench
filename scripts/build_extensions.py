"""Build 100-item extensions for the simple tasks (translate / punctuate /
fill-in / compress).

Each extension is a disjoint set of 100 new items sampled from the same source
pool with the same filtering logic as v1, but with seed=43 (v1 used seed=42)
and after removing any input already present in v1.

Outputs:
  data/translate.ext.jsonl
  data/punctuate.ext.jsonl
  data/fill_in.ext.jsonl
  data/compress.ext.jsonl

Not handled here:
  - idiom-source: already has data/idiom_source.v2.jsonl (Tier-1 replacement)
  - char-gloss: already has data/char_gloss.candidates.jsonl (18-item upgrade)
  - collation: still at 50-item prototype; expand via build_collation.py
    (re-run with larger N_TARGET when adopted).

Cost note: these extensions are *free to build* but each one taken to
production costs an extra 100 inference calls × 10 models × judge calls. Hold
until ready to commit to a re-baseline.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CORPUS_REPO = Path.home() / "Documents/zion/classical-corpus"
INSTRUCT_DIR = CORPUS_REPO / "output/instruct"
CORPUS_JSONL = CORPUS_REPO / "output/corpus.jsonl"
DATA = REPO / "data"

EXT_SEED = 43
N_EXT = 100


def _load_v1_inputs(path: Path) -> set[str]:
    """Return the set of `input` strings already present in v1 file."""
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.add(r["input"])
    return out


# ---------------------------------------------------------------------------
# translate extension
# ---------------------------------------------------------------------------

def build_translate_ext() -> None:
    src = INSTRUCT_DIR / "translate.jsonl"
    v1_inputs = _load_v1_inputs(DATA / "translate.jsonl")
    rng = random.Random(EXT_SEED)
    pool: dict[str, list[dict]] = defaultdict(list)
    CATEGORIES = ("经", "史", "子", "集")

    with src.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") != "c2m":
                continue
            if r.get("_has_box"):
                continue
            src_len = len(r["input"])
            if not (10 <= src_len <= 60):
                continue
            cat = r.get("category", "")
            if cat not in CATEGORIES:
                continue
            if r["input"] in v1_inputs:
                continue
            pool[cat].append(r)

    per_cat = N_EXT // len(CATEGORIES)
    samples: list[dict] = []
    for cat in CATEGORIES:
        items = pool.get(cat, [])
        picked = rng.sample(items, min(per_cat, len(items)))
        samples.extend(picked)
    rng.shuffle(samples)
    samples = samples[:N_EXT]

    out_path = DATA / "translate.ext.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(samples, 1):
            out = {
                "id": f"translate-ext#{i}",
                "task": "translate",
                "instruction": "将下列古文翻译成现代汉语：",
                "input": r["input"],
                "reference": r["output"],
                "metadata": {
                    "source": r["source"],
                    "category": r["category"],
                    "_extension": True,
                },
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"translate.ext.jsonl: wrote {len(samples)} (disjoint from {len(v1_inputs)} v1)")


# ---------------------------------------------------------------------------
# punctuate extension
# ---------------------------------------------------------------------------

def build_punctuate_ext() -> None:
    src = INSTRUCT_DIR / "punctuate.jsonl"
    v1_inputs = _load_v1_inputs(DATA / "punctuate.jsonl")
    rng = random.Random(EXT_SEED)
    pool: dict[str, list[dict]] = defaultdict(list)

    with src.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("_has_box"):
                continue
            in_len = len(r["input"])
            if not (30 <= in_len <= 200):
                continue
            if r["input"] in v1_inputs:
                continue
            pool[r.get("source", "")].append(r)

    sources = sorted(pool.keys())
    src_iters = {s: iter(rng.sample(pool[s], len(pool[s]))) for s in sources}
    samples: list[dict] = []
    while len(samples) < N_EXT:
        any_taken = False
        for s in sources:
            if len(samples) >= N_EXT:
                break
            try:
                samples.append(next(src_iters[s]))
                any_taken = True
            except StopIteration:
                continue
        if not any_taken:
            break
    rng.shuffle(samples)

    LEADING_NOISE = "，。：；、！？「」『』《》（）()【】 　"
    out_path = DATA / "punctuate.ext.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(samples, 1):
            ref = r["output"].lstrip(LEADING_NOISE).strip()
            inp = r["input"].strip()
            out = {
                "id": f"punctuate-ext#{i}",
                "task": "punctuate",
                "instruction": "为下列古文添加标点：",
                "input": inp,
                "reference": ref,
                "metadata": {
                    "source": r.get("source", ""),
                    "category": r.get("category", ""),
                    "_extension": True,
                },
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"punctuate.ext.jsonl: wrote {len(samples)} (disjoint from {len(v1_inputs)} v1)")


# ---------------------------------------------------------------------------
# fill_in extension
# ---------------------------------------------------------------------------

STOPWORDS = set("之乎者也而其以为与于焉夫且乃则若所是非有无可不")
PUNCT = set("，。：；、！？「」『』《》（）()【】 　“”‘’\n")


def _good_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？]", text)
    out = []
    for p in parts:
        p = p.strip()
        p = re.sub(r"[【].*?[】]", "", p)
        p = re.sub(r"[（(].*?[)）]", "", p)
        p = p.replace("　", "").strip()
        p = re.sub(r"\s+", "", p)
        cn = [c for c in p if "一" <= c <= "鿿"]
        if 8 <= len(cn) <= 25:
            out.append(p)
    return out


def _pick_mask(sentence: str, rng: random.Random) -> int | None:
    chars = list(sentence)
    n = len(chars)
    cands = []
    for i, c in enumerate(chars):
        if i == 0 or i == n - 1:
            continue
        if not ("一" <= c <= "鿿"):
            continue
        if c in STOPWORDS or c in PUNCT:
            continue
        # unique within sentence
        if sentence.count(c) != 1:
            continue
        cands.append(i)
    if not cands:
        return None
    return rng.choice(cands)


def build_fill_in_ext() -> None:
    v1_inputs = _load_v1_inputs(DATA / "fill_in.jsonl")
    rng = random.Random(EXT_SEED)
    # Same source filters as v1: prefer 经 records (论语/孟子/大学/中庸 …)
    JING = {"论语", "孟子", "大学", "中庸", "诗经", "尚书", "礼记", "周易"}
    pool: list[dict] = []
    with CORPUS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("source") not in JING:
                continue
            content = r.get("content", "")
            for sent in _good_sentences(content):
                pool.append({"source": r["source"], "category": r.get("category", ""), "sentence": sent})

    rng.shuffle(pool)
    samples: list[dict] = []
    seen_inputs: set[str] = set()
    for entry in pool:
        if len(samples) >= N_EXT:
            break
        idx = _pick_mask(entry["sentence"], rng)
        if idx is None:
            continue
        masked = entry["sentence"][:idx] + "□" + entry["sentence"][idx + 1:]
        if masked in v1_inputs or masked in seen_inputs:
            continue
        ans = entry["sentence"][idx]
        samples.append({
            "input": masked,
            "reference": ans,
            "source": entry["source"],
            "category": entry["category"],
            "sentence": entry["sentence"],
        })
        seen_inputs.add(masked)

    out_path = DATA / "fill_in.ext.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(samples, 1):
            out = {
                "id": f"fill-in-ext#{i}",
                "task": "fill-in",
                "instruction": "请填出下列古文中 □ 处缺失的一个汉字：",
                "input": s["input"],
                "reference": s["reference"],
                "metadata": {
                    "source": s["source"],
                    "category": s["category"],
                    "sentence": s["sentence"],
                    "_extension": True,
                },
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"fill_in.ext.jsonl: wrote {len(samples)} (disjoint from {len(v1_inputs)} v1)")


# ---------------------------------------------------------------------------
# compress extension
# ---------------------------------------------------------------------------

CN_RANGES = [(0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0x20000, 0x2A6DF), (0x2A700, 0x2EBEF)]


def _cn_len(s: str) -> int:
    n = 0
    for ch in s:
        cp = ord(ch)
        for lo, hi in CN_RANGES:
            if lo <= cp <= hi:
                n += 1
                break
    return n


def build_compress_ext() -> None:
    src = INSTRUCT_DIR / "translate.jsonl"
    v1_inputs = _load_v1_inputs(DATA / "compress.jsonl")
    rng = random.Random(EXT_SEED)
    by_cat: dict[str, list[dict]] = defaultdict(list)

    with src.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("task") != "m2c":
                continue
            if r.get("_has_box"):
                continue
            mod, cls = r["input"], r["output"]
            mlen, clen = _cn_len(mod), _cn_len(cls)
            if not (100 <= mlen <= 300):
                continue
            if not (30 <= clen <= mlen):
                continue
            ratio = clen / mlen
            if not (0.30 <= ratio <= 0.75):
                continue
            if mod in v1_inputs:
                continue
            cat = r.get("category") or "?"
            by_cat[cat].append({
                "modern": mod, "classical": cls,
                "source": r.get("source", ""), "category": cat,
            })

    # Same round-robin pattern as v1
    cats = sorted(by_cat.keys())
    iters = {c: iter(rng.sample(by_cat[c], len(by_cat[c]))) for c in cats}
    samples: list[dict] = []
    while len(samples) < N_EXT:
        any_taken = False
        for c in cats:
            if len(samples) >= N_EXT:
                break
            try:
                samples.append(next(iters[c]))
                any_taken = True
            except StopIteration:
                continue
        if not any_taken:
            break
    rng.shuffle(samples)

    INSTRUCTION = "将下列现代汉语压缩成等义的文言文，力求简洁，不要解释，直接输出文言文："
    out_path = DATA / "compress.ext.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, s in enumerate(samples, 1):
            out = {
                "id": f"compress-ext#{i}",
                "task": "compress",
                "instruction": INSTRUCTION,
                "input": s["modern"],
                "reference": s["classical"],
                "metadata": {
                    "source": s["source"], "category": s["category"],
                    "_extension": True,
                },
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"compress.ext.jsonl: wrote {len(samples)} (disjoint from {len(v1_inputs)} v1)")


def main() -> None:
    print("building task extensions (disjoint from v1, seed=43)...")
    build_translate_ext()
    build_punctuate_ext()
    build_fill_in_ext()
    build_compress_ext()
    print("\nDone. Extensions ready in data/*.ext.jsonl.")
    print("To use: merge with v1 for a 200-item run when re-baselining (costs model calls).")


if __name__ == "__main__":
    main()
