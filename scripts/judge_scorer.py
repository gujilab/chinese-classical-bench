"""LLM-as-judge scorer hook for translate / char-gloss.

Drop-in supplement for `scorers.py`. Does NOT modify the existing chrF-based
pipeline; instead exposes `score_with_judge(pred, rec, task)` returning
{judge: int 0-5, judge_norm: float 0..1}. Integrate by adding to the per-item
score dict alongside chrf, or by post-processing results with this function.

Why a separate file: chrF/char_f1 are cheap deterministic metrics. The judge
costs ~14s + dollars per call (claude-opus-4-7 via kcli-gw). Keep it opt-in.

See: experiments/llm-judge/report.md for correlation analysis and rationale.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

DEFAULT_BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:8990/v1")
# No hardcoded key: read from env, error at call time if unset.
DEFAULT_API_KEY = os.environ.get("JUDGE_API_KEY", "")
DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-4-7")

PROMPT_TRANSLATE = """你是古文翻译评分专家。请阅读原文、参考翻译、模型译文，给模型译文打分 0-5 整数:
0 = 完全错误或胡说
1 = 大部分错误
2 = 部分正确，关键信息错
3 = 基本正确，细节有损
4 = 准确流畅，与参考几乎等价
5 = 比参考更精准或同样优秀

原文: {input}
参考译文: {reference}
模型译文: {prediction}

直接输出整数 0-5，不要解释。"""

PROMPT_CHAR_GLOSS = """你是古汉语字义评分专家。请评分模型对古文中某字含义的解释 0-5 整数:
0 = 字义完全错
1 = 含义偏离
2 = 部分正确
3 = 基本正确
4 = 准确
5 = 准确且包含语境

问题: {input}
参考释义: {reference}
模型释义: {prediction}

直接输出整数 0-5，不要解释。"""

PROMPTS = {"translate": PROMPT_TRANSLATE, "char-gloss": PROMPT_CHAR_GLOSS}
_DIGIT_RE = re.compile(r"[0-5]")


def _call(prompt: str, base_url: str, api_key: str, model: str,
          timeout: int = 90, max_tokens: int = 8) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "你是评分助手。只输出一个 0-5 之间的整数。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def _parse(raw: str) -> int | None:
    if not raw:
        return None
    m = _DIGIT_RE.search(raw)
    return int(m.group(0)) if m else None


def score_with_judge(
    pred: str,
    rec: dict,
    task: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    retries: int = 3,
) -> dict:
    """Return {'judge': int 0-5 or None, 'judge_norm': float 0..1 or None}.

    Caller decides how to combine with chrF (e.g. average with judge_norm,
    or use judge_norm as primary metric for translate/char-gloss).
    """
    if task not in PROMPTS:
        raise ValueError(f"judge not configured for task={task!r}")
    if not api_key:
        raise RuntimeError(
            "judge API key not set — export JUDGE_API_KEY (or pass api_key=...)"
        )
    prompt = PROMPTS[task].format(
        input=rec.get("input", ""),
        reference=rec.get("reference", ""),
        prediction=pred,
    )
    last_err: Exception | None = None
    raw = ""
    for attempt in range(retries):
        try:
            raw = _call(prompt, base_url, api_key, judge_model)
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                ConnectionError) as e:
            last_err = e
        except Exception as e:  # noqa: BLE001
            last_err = e
    else:
        # All retries failed.
        return {"judge": None, "judge_norm": None, "error": str(last_err)}

    judge_score = _parse(raw)
    if judge_score is None:
        return {"judge": None, "judge_norm": None, "judge_raw": raw}
    return {"judge": judge_score, "judge_norm": judge_score / 5.0, "judge_raw": raw}


# ---- helper to extend an existing item's `scores` dict ----

def augment_item_with_judge(item: dict, rec: dict, task: str, **kwargs) -> dict:
    """Mutate-and-return: adds judge fields into item['scores']."""
    res = score_with_judge(item.get("prediction", ""), rec, task, **kwargs)
    item.setdefault("scores", {}).update(
        {k: v for k, v in res.items() if v is not None}
    )
    return item


if __name__ == "__main__":
    # Quick CLI: python judge_scorer.py task input_json
    import sys
    if len(sys.argv) < 3:
        print("Usage: python judge_scorer.py <task> <json-record-with-prediction>")
        sys.exit(2)
    task = sys.argv[1]
    rec = json.loads(sys.argv[2])
    pred = rec.pop("prediction", "")
    print(json.dumps(score_with_judge(pred, rec, task), ensure_ascii=False))
