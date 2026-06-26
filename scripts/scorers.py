"""Per-task scoring functions.

Each scorer takes (prediction: str, record: dict) → dict of metric_name: float.
The record is the full benchmark question (including reference + metadata).
"""

import re
import unicodedata
from collections import Counter

PUNCT_CN = set("，。：；、！？「」『』《》（）()【】“”‘’,.;:!?\"'-—…")


def _strip(s: str) -> str:
    """Normalize whitespace and collapse runs of whitespace."""
    s = unicodedata.normalize("NFKC", s).strip()
    s = re.sub(r"\s+", "", s)
    return s


def _chinese_chars(s: str) -> str:
    return "".join(c for c in s if "一" <= c <= "鿿")


def _ngrams(s: str, n: int) -> Counter:
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def chrf(pred: str, ref: str, beta: float = 2.0, max_n: int = 6) -> float:
    """Character n-gram F-score (chrF), n=1..6, F2 by default."""
    if not pred or not ref:
        return 0.0
    f_scores = []
    for n in range(1, max_n + 1):
        p_ng = _ngrams(pred, n)
        r_ng = _ngrams(ref, n)
        if not p_ng or not r_ng:
            continue
        overlap = sum((p_ng & r_ng).values())
        precision = overlap / sum(p_ng.values()) if p_ng else 0.0
        recall = overlap / sum(r_ng.values()) if r_ng else 0.0
        if precision + recall == 0:
            f_scores.append(0.0)
        else:
            b2 = beta * beta
            f_scores.append(
                (1 + b2) * precision * recall / (b2 * precision + recall)
            )
    return sum(f_scores) / len(f_scores) if f_scores else 0.0


def char_f1(pred: str, ref: str) -> float:
    """Bag-of-chars F1."""
    p = Counter(pred)
    r = Counter(ref)
    if not p or not r:
        return 0.0
    overlap = sum((p & r).values())
    precision = overlap / sum(p.values())
    recall = overlap / sum(r.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------- task-specific ----------


def score_translate(pred: str, rec: dict) -> dict:
    pred = _strip(pred)
    ref = _strip(rec["reference"])
    return {
        "chrf": chrf(pred, ref),
        "char_f1": char_f1(pred, ref),
    }


def score_punctuate(pred: str, rec: dict) -> dict:
    """Score punctuation by char-aligned punctuation F1."""
    pred = _strip(pred)
    ref = _strip(rec["reference"])
    # The plain (no-punct) input should appear stripped of punctuation in both.
    pred_plain = "".join(c for c in pred if c not in PUNCT_CN)
    ref_plain = "".join(c for c in ref if c not in PUNCT_CN)
    char_match = pred_plain == ref_plain  # did the model preserve text?
    # Punctuation positions: map char-index → punct-after
    def punct_map(s: str) -> dict:
        m = {}
        idx = 0
        for c in s:
            if c in PUNCT_CN:
                m.setdefault(idx - 1, []).append(c)
            else:
                idx += 1
        return m

    p_map = punct_map(pred)
    r_map = punct_map(ref)
    keys = set(p_map) | set(r_map)
    if not keys:
        return {"punct_f1": 1.0, "char_preserved": float(char_match)}
    tp = sum(1 for k in keys if k in p_map and k in r_map)
    fp = sum(1 for k in p_map if k not in r_map)
    fn = sum(1 for k in r_map if k not in p_map)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "punct_f1": f1,
        "char_preserved": float(char_match),
        "chrf": chrf(pred, ref),
    }


def score_char_gloss(pred: str, rec: dict) -> dict:
    """Short gloss; use chrF + char-F1 (punct-stripped, since gloss can have
    free punctuation/synonyms)."""
    pred_n = "".join(c for c in _strip(pred) if c not in PUNCT_CN)
    ref_n = "".join(c for c in _strip(rec["reference"]) if c not in PUNCT_CN)
    return {
        "chrf": chrf(pred_n, ref_n),
        "char_f1": char_f1(pred_n, ref_n),
    }


# --- idiom-source helpers --------------------------------------------------

# Common alternate names for books in our ground truth. Canonical → variants
# that should count as a hit.
_BOOK_ALIASES = {
    "老子":   ["道德经"],
    "庄子":   ["南华经", "南华真经"],
    "史记":   ["太史公书"],
    "汉书":   ["前汉书"],
    "周易":   ["易经"],
    "尚书":   ["书经"],
    "诗经":   ["毛诗"],
    "左传":   ["左氏春秋", "春秋左氏传", "左氏传"],
    "公羊传": ["春秋公羊传"],
    # OpenCC's t2s maps 穀→谷, so the normalized form of 穀梁傳 is 谷梁传.
    "穀梁传": ["春秋穀梁传", "谷梁传", "春秋谷梁传"],
}

_BOOK_PUNCT = "《》「」『』\"\"''（）()【】 \t\n"

# Lazy opencc import: t2s normalization is a real improvement (models often
# quote Classical Chinese in traditional script) but we don't want to hard-
# require the dep. Fall back to identity if missing.
try:
    from opencc import OpenCC as _OpenCC
    _cc_t2s = _OpenCC("t2s")
    def _t2s(s: str) -> str:
        return _cc_t2s.convert(s)
except ImportError:
    import sys as _sys
    print("INFO: opencc not installed; t2s normalization disabled (pip install opencc-python-reimplemented)", file=_sys.stderr)
    def _t2s(s: str) -> str:
        return s
except Exception as _opencc_err:  # pragma: no cover
    import sys as _sys
    print(f"WARN: opencc failed to load ({_opencc_err}); t2s normalization disabled", file=_sys.stderr)
    def _t2s(s: str) -> str:
        return s


def _norm_book(s: str) -> str:
    """Normalize traditional→simplified and strip brackets/whitespace so
    substring matching does the right thing across script variants."""
    s = _t2s(s)
    for c in _BOOK_PUNCT:
        s = s.replace(c, "")
    return s


def _book_hit(pred: str, book: str) -> bool:
    p = _norm_book(pred)
    if _norm_book(book) in p:
        return True
    for alias in _BOOK_ALIASES.get(book, ()):
        if _norm_book(alias) in p:
            return True
    return False


def score_idiom_source(pred: str, rec: dict) -> dict:
    """Book name match (with t2s normalization + common aliases) + quote chrF."""
    pred = _strip(pred)
    book = rec["metadata"]["book"]
    quote = _strip(rec["metadata"]["expected_quote"])
    return {
        "book_em": float(_book_hit(pred, book)),
        "quote_chrf": chrf(pred, quote),
    }


def score_fill_in(pred: str, rec: dict) -> dict:
    """Single char answer.

    Try to extract the answer:
      1. quoted: 「X」/"X"/'X' → X (single char)
      2. only one Chinese char in entire pred → that char
      3. else first Chinese char (often correct for terse responses)
    """
    pred_s = _strip(pred)
    ans = _strip(rec["reference"])
    extracted = ""
    # 1. quoted single char
    m = re.search(r'[「『"\'"‘“]([一-鿿])[」』"\'"’”]', pred_s)
    if m:
        extracted = m.group(1)
    else:
        cn = _chinese_chars(pred_s)
        if len(cn) == 1:
            extracted = cn
        elif cn:
            extracted = cn[0]
    # Models often answer in traditional script (饑 vs 饥); normalize both
    # sides t2s before comparing, same as the idiom-source book matcher.
    ex_n, ans_n = _t2s(extracted), _t2s(ans)
    return {
        "exact_match": float(ex_n == ans_n and ex_n != ""),
        "in_pred": float(ans in pred_s or ans_n in _t2s(pred_s)),
    }


def score_compress(pred: str, rec: dict) -> dict:
    """Compress task: modern Chinese → classical Chinese.

    Rewards both fidelity (chrf vs reference classical) and compression
    (output shorter than input). Efficiency = chrf * max(0, 1 - ratio).
    """
    ref = rec["reference"]
    inp = rec["input"]
    chrf_val = chrf(pred, ref)
    pred_cn = _chinese_chars(pred)
    inp_cn = _chinese_chars(inp)
    ratio = len(pred_cn) / max(len(inp_cn), 1)
    compression = max(0.0, 1.0 - ratio)
    efficiency = chrf_val * compression
    return {
        "chrf": round(chrf_val, 4),
        "ratio": round(ratio, 4),
        "compression": round(compression, 4),
        "efficiency": round(efficiency, 4),
    }


def score_collation(pred: str, rec: dict) -> dict:
    """Single-char answer — same shape as fill-in.

    Tests whether the model can parse classical-Chinese textual-criticism
    conventions (一作 / 俗本作 / 当作 / 讹作 …) and return the canonical
    reading. T2S normalization applied on both sides.
    """
    pred_s = _strip(pred)
    ans = _strip(rec["reference"])
    extracted = ""
    m = re.search(r'[「『"\'"‘“]([一-鿿])[」』"\'"’”]', pred_s)
    if m:
        extracted = m.group(1)
    else:
        cn = _chinese_chars(pred_s)
        if len(cn) == 1:
            extracted = cn
        elif cn:
            extracted = cn[0]
    ex_n, ans_n = _t2s(extracted), _t2s(ans)
    return {
        "exact_match": float(ex_n == ans_n and ex_n != ""),
        "in_pred": float(ans in pred_s or ans_n in _t2s(pred_s)),
    }


SCORERS = {
    "translate": score_translate,
    "punctuate": score_punctuate,
    "char-gloss": score_char_gloss,
    "idiom-source": score_idiom_source,
    "fill-in": score_fill_in,
    "compress": score_compress,
    "collation": score_collation,
}


def score(record: dict, prediction: str) -> dict:
    fn = SCORERS[record["task"]]
    return fn(prediction, record)
