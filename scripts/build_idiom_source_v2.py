"""Build idiom-source **v2** — contamination-robust, Tier-1 sources only.

Why: findings.md §6 shows idiom-source is the worst task — 23 ceiling items
+ ρ=0.68 source-canonicity (it samples famous canon, so it partly measures
*recognising* 论语/史记 rather than Classical competence). v2 resamples the
whole task from **Tier-1 (obscure) corpus books** so the score reflects
competence over recall.

Source: data/_vendor/xinhua_idiom.json (pinned CC0, see
fetch_idiom_source_data.py). Each idiom's `derivation` names its 出处; we
parse the first 《…》 as the source work and the trailing clause as the
quote, keep only books that are (a) in our CC0 corpus whitelist — so models
have fair access to ground truth — and (b) canonicity Tier 1.

NOT adopted in place: replacing items invalidates the stored predictions in
results/*.json, so adoption needs a scoped idiom-source rerun (cost). Output
is staged as data/idiom_source.v2.jsonl with _status/_provenance/_needs.

Usage:  python scripts/build_idiom_source_v2.py
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from canonicity import tier  # noqa: E402

SRC = REPO / "data" / "_vendor" / "xinhua_idiom.json"
OUT = REPO / "data" / "idiom_source.v2.jsonl"
EXISTING = REPO / "data" / "idiom_source.jsonl"
N_TARGET = 100
PER_BOOK_CAP = 14  # keep the set spread across books, not one history

# Corpus whitelist (same fairness rule as the original builder): the model
# must have had access to the source text in our CC0 corpus.
KNOWN_BOOKS = {
    "论语", "孟子", "大学", "中庸", "诗经", "尚书", "礼记", "周易",
    "左传", "公羊传", "穀梁传", "孝经", "尔雅",
    "史记", "汉书", "后汉书", "三国志", "晋书", "宋书", "南齐书",
    "梁书", "陈书", "魏书", "北齐书", "周书", "南史", "北史", "隋书",
    "资治通鉴", "说文解字",
    "庄子", "老子", "荀子", "韩非子", "孙子兵法", "墨子",
    "战国策", "国语", "吕氏春秋",
}

# derivation e.g. '语出《法华经·法师功德品》下至阿鼻地狱。”'
#                  '三国·魏·曹操《整齐风俗令》阿党比周，先圣所疾也。”'
_WORK = re.compile(r"《([^《》]+?)》(.*)")
_QCLEAN = re.compile(r'^[，。：、\s"”’]+|[，。\s"”’]+$')


def parse_derivation(d: str) -> tuple[str, str] | None:
    """→ (bare_book, quote) or None."""
    m = _WORK.search(d or "")
    if not m:
        return None
    work = m.group(1).strip()
    book = re.split(r"[·、]", work)[0].strip()
    tail = m.group(2)
    # xinhua format: 《book·chapter》<quote>。” [secondary 朝·作者《…》…。”]
    # the primary quote ends at the first closing ” — cut there so a
    # trailing secondary citation never leaks into the quote.
    cut = re.search(r'[”"]', tail)
    if cut:
        tail = tail[:cut.start()]
    elif "《" in tail:                      # no ” but a secondary work starts
        tail = tail[:tail.index("《")]
    quote = _QCLEAN.sub("", tail.strip()).rstrip('”"’。').strip()
    if not book or not quote:
        return None
    return book, quote


def main() -> int:
    if not SRC.exists():
        print("run scripts/fetch_idiom_source_data.py first")
        return 1
    data = json.loads(SRC.read_text(encoding="utf-8"))
    used = {json.loads(l)["input"]
            for l in EXISTING.open(encoding="utf-8") if l.strip()}

    pool = []
    for r in data:
        word = (r.get("word") or "").strip()
        if not word or not (3 <= len(word) <= 10) or word in used:
            continue
        pr = parse_derivation(r.get("derivation", ""))
        if not pr:
            continue
        book, quote = pr
        if book not in KNOWN_BOOKS or tier(book) != 1:
            continue
        if not (6 <= len(quote) <= 100):
            continue
        pool.append({"idiom": word, "book": book, "quote": quote,
                     "explanation": (r.get("explanation") or "").strip()})

    # dedup idioms, deterministic order (sorted), round-robin by book under a
    # per-book cap so no single history dominates → harder, balanced set.
    pool.sort(key=lambda c: (c["book"], c["idiom"]))
    seen, by_book = set(), {}
    for c in pool:
        if c["idiom"] in seen:
            continue
        seen.add(c["idiom"])
        by_book.setdefault(c["book"], []).append(c)

    picked: list[dict] = []
    books = sorted(by_book)
    idx = 0
    while len(picked) < N_TARGET and any(by_book.values()):
        b = books[idx % len(books)]
        idx += 1
        lst = by_book.get(b)
        if lst and sum(1 for p in picked if p["book"] == b) < PER_BOOK_CAP:
            picked.append(lst.pop(0))
        if idx > len(books) * (PER_BOOK_CAP + 2):
            break

    with OUT.open("w", encoding="utf-8") as f:
        for i, c in enumerate(picked, 1):
            f.write(json.dumps({
                "id": f"idiom-source#{i}",
                "task": "idiom-source",
                "instruction": "下列成语出自哪部典籍？请给出书名和原文引文。",
                "input": c["idiom"],
                "reference": f"出自《{c['book']}》：「{c['quote']}」",
                "metadata": {
                    "book": c["book"],
                    "book_full": f"《{c['book']}》",
                    "expected_quote": c["quote"],
                    "explanation": c["explanation"],
                    "_status": "candidate",
                    "_provenance": "pwxcoo/chinese-xinhua idiom.json "
                                   "@8de1001 derivation; Tier-1 source",
                    "_needs": "scoped idiom-source rerun before adoption "
                              "(replacing items invalidates stored results)",
                },
            }, ensure_ascii=False) + "\n")

    tiers = Counter(tier(p["book"]) for p in picked)
    books_c = Counter(p["book"] for p in picked)
    print(f"pool (Tier-1 ∩ corpus, deduped): {len(seen):,}")
    print(f"wrote {len(picked)} → {OUT.relative_to(REPO)}")
    print(f"tier mix: {dict(tiers)}  (target: all tier 1)")
    print(f"books: {dict(books_c.most_common())}")
    print(f"overlap with existing idiom_source.jsonl idioms: "
          f"{len(used & {p['idiom'] for p in picked})} (should be 0)")
    if len(picked) < N_TARGET:
        print(f"⚠ only {len(picked)}/{N_TARGET} — Tier-1∩corpus pool "
              f"exhausted; report honestly, do not pad with canon")
    return 0


if __name__ == "__main__":
    sys.exit(main())
