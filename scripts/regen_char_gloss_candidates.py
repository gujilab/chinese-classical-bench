"""Stage replacement *candidates* for the 18 circular-gold char-gloss items.

NOT an in-place fix. These items' gold is the dictionary stub "同本义。"
("same as the 本义, defined elsewhere"). The fix is to supply that 本义.

Two sources, in confidence order:
  1. **chinese-xinhua word.json** (CC0, pinned via fetch_idiom_source_data.py)
     — every entry's `explanation` states the 本义 in a parenthetical, e.g.
     穹 "(形声…本义穷尽)". We extract that phrase. This is exactly the gloss
     "同本义" was pointing at, in modern Chinese, matching the task's
     expected answer style ("一个简短的现代汉语短语"). High confidence,
     deterministic, verifiable → `_status: candidate-verified`.
  2. 说文解字 head as a fallback only (`_status: candidate-shuowen`,
     low confidence — 说文 is terse Classical, often ≠ contextual sense).
  3. neither → `_status: blocked`.

Adoption still needs human/judge review AND a scoped char-gloss rerun
(replacing items invalidates results/*.json). Output:
data/char_gloss.candidates.jsonl. Tracked in docs/quality-audit.md.

Usage:  python scripts/regen_char_gloss_candidates.py
"""
from __future__ import annotations
import json, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "data" / "_vendor"
SHUOWEN = Path.home() / "Documents/zion/classical-corpus/output/shuowen.json"
OUT = REPO / "data" / "char_gloss.candidates.jsonl"

try:
    from opencc import OpenCC
    _t2s = OpenCC("t2s").convert
except Exception:
    def _t2s(s: str) -> str:
        return s

_BENYI = re.compile(r"本义([^)）]+)[)）]")
_SW_CUT = re.compile(r"[，。]?\s*(?:从|從|凡|《|讀若|读若|聲|声|象)")


def xinhua_benyi(explanation: str) -> str | None:
    """Extract & tidy the 本义 phrase from a word.json explanation."""
    m = _BENYI.search(explanation or "")
    if not m:
        return None
    phrase = m.group(1).strip(" 　,，;；、.。")
    # keep it short: first one or two sense units, cap length
    parts = re.split(r"[;；。]", phrase)
    out = parts[0].strip()
    if len(out) <= 4 and len(parts) > 1 and parts[1].strip():
        out = f"{out}；{parts[1].strip()}"
    return out[:14] or None


def shuowen_gloss(content: str) -> str | None:
    if not content:
        return None
    head = _SW_CUT.split(content, 1)[0].strip()
    head = re.sub(r"[，。、；：！？\s]+$", "", head)
    head = re.sub(r"也$", "", head)
    head = _t2s(head).strip("，。、；：！？ ")
    return head or None


def main() -> None:
    xh = {}
    wf = VENDOR / "xinhua_word.json"
    if wf.exists():
        for e in json.loads(wf.read_text(encoding="utf-8")):
            xh.setdefault(e["word"], e.get("explanation", ""))
    else:
        print("WARN: run scripts/fetch_idiom_source_data.py for word.json")
    sw = {}
    if SHUOWEN.exists():
        for e in json.loads(SHUOWEN.read_text(encoding="utf-8")):
            sw.setdefault(e["char"], e["content"])

    recs = [json.loads(l) for l in
            (REPO / "data" / "char_gloss.jsonl").open(encoding="utf-8")
            if l.strip()]
    flagged = [r for r in recs
               if r.get("metadata", {}).get("_audit_issue", "")
               .startswith("circular gold")]

    out, n_v, n_s, n_b = [], 0, 0, 0
    for r in flagged:
        ch = r["metadata"].get("char", "")
        rec = {
            "id": r["id"], "task": "char-gloss",
            "instruction": r["instruction"], "input": r["input"],
            "metadata": {k: v for k, v in r["metadata"].items()
                         if k != "_audit_issue"},
        }
        cand = xinhua_benyi(xh.get(ch, ""))
        if cand:
            rec.update(reference=cand, _status="candidate-verified",
                       _provenance=f"chinese-xinhua word.json「{ch}」本义",
                       _needs="human/judge review + scoped rerun")
            n_v += 1
        else:
            sg = shuowen_gloss(sw.get(ch, "")) if ch in sw else None
            if sg and len(sg) <= 8:
                rec.update(reference=sg, _status="candidate-shuowen",
                           _provenance=f"说文解字「{ch}」 (low confidence)",
                           _needs="human/judge review + scoped rerun")
                n_s += 1
            else:
                rec.update(reference=r["reference"], _status="blocked",
                           _provenance=f"no source for {ch!r}",
                           _needs="manual gloss")
                n_b += 1
        out.append(rec)

    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in out), encoding="utf-8")
    print(f"{len(out)} flagged → {n_v} verified (xinhua 本义), "
          f"{n_s} shuowen-fallback, {n_b} blocked → {OUT.relative_to(REPO)}")
    for r in out:
        s = {"candidate-verified": "VERIF", "candidate-shuowen": "SHUOW",
             "blocked": "BLOCK"}[r["_status"]]
        print(f"  [{s}] {r['id']} 字{r['metadata'].get('char','?')} "
              f"→ {r['reference']!r}")


if __name__ == "__main__":
    main()
