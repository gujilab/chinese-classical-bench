"""Source-canonicity tiers — single source of truth.

A 3-level ordinal proxy for how over-represented a source text is in any
LLM training corpus (used by contamination_probe.py and aggregate.py so the
definition can't drift between them):

  3 = core canon, effectively memorized verbatim by every LLM
      (Four Books + the most-anthologized classics + 史记)
  2 = well-known classics/histories, quoted but less verbatim
  1 = everything else (obscure dynastic histories, specialized 子)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

TIER3 = {"论语", "孟子", "大学", "中庸", "诗经", "周易", "尚书", "礼记",
         "左传", "老子", "庄子", "孙子兵法", "史记"}
TIER2 = {"汉书", "后汉书", "三国志", "资治通鉴", "韩非子", "荀子",
         "战国策", "国语", "墨子", "孝经", "尔雅", "仪礼", "周礼",
         "吕氏春秋", "淮南子", "列子", "公羊传", "穀梁传", "晋书"}

TASK_FILES = {
    "translate": "translate.jsonl", "punctuate": "punctuate.jsonl",
    "char-gloss": "char_gloss.jsonl", "idiom-source": "idiom_source.jsonl",
    "fill-in": "fill_in.jsonl", "compress": "compress.jsonl",
}


def tier(book: str) -> int:
    return 3 if book in TIER3 else 2 if book in TIER2 else 1


def book_of(metadata: dict) -> str:
    """Bare source-book name from an item's metadata (source|book)."""
    src = metadata.get("source") or metadata.get("book") or ""
    return re.split(r"[·/]", src)[0].strip() if src else ""


def id_to_tier(data_dir: Path) -> dict[str, int]:
    """{item_id: canonicity tier} across all task data files."""
    out: dict[str, int] = {}
    for fname in TASK_FILES.values():
        fp = data_dir / fname
        if not fp.exists():
            continue
        for line in fp.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            book = book_of(r.get("metadata", {}))
            if book:
                out[r["id"]] = tier(book)
    return out
