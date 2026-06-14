"""
Generate (idiom_prompt, literal_prompt, question) triples for each 典故.

For each 典故 we ask Sonnet 4.6 to produce:
- A real-world scenario prompt that USES the 典故 in 1 short reference (含 idiom).
- The SAME scenario but with the 典故 expanded into a literal Chinese description
  conveying the same situation/strategy/state. The literal version should not mention
  the 典故 name. The two should be otherwise as similar as possible.
- A downstream question about the scenario that is identical in both versions.

Output: scenarios.jsonl
"""
import asyncio
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

BASE_URL = os.environ.get("KCLI_BASE_URL", "http://localhost:8990/v1")
API_KEY = os.environ.get("KCLI_API_KEY")
if not API_KEY:
    sys.exit("KCLI_API_KEY not set — export it before running (no default key).")
MODEL = os.environ.get("GEN_MODEL", "claude-sonnet-4-6")
CONCURRENCY = 1

HERE = Path(__file__).parent
SEED = HERE / "scenarios_seed.txt"
OUT = HERE / "scenarios.jsonl"

SYS_PROMPT = """你是一位中文典故与提示工程专家。

任务：给定一个中文典故，请生成一个用于 prompt 压缩对比实验的三元组：
1. idiom_prompt：一段真实世界的应用场景描述（中文，1-2 句话），其中**自然嵌入这个典故**（典故 4 字直接出现一次，不要解释它）。场景应该是政治、商业、管理、人际、决策、教育等现实领域，让 LLM 觉得这是一个合理的求助/分析请求。
2. literal_prompt：把上面 idiom_prompt 中**那个典故**展开成 1-2 句**白话描述**（不点出典故名），保留其他部分完全不变。展开应当准确传达典故所代表的策略/情境/状态，不要更短，应当更详细。
3. question：一个对场景的下游提问（中文，1 句话），两个版本通用。提问应有分析深度，比如"这种策略风险何在？"、"应该怎么应对？"、"用 3 点分析其利弊"。

要求：
- idiom_prompt 和 literal_prompt 必须叙述同一个情境，只在"典故 vs 白话展开"这一段上不同。
- literal_prompt 通常比 idiom_prompt 长 30-100 字。
- 不要在 prompt 中出现"典故"、"成语"等元描述词。

输出 JSON：{"idiom_prompt": "...", "literal_prompt": "...", "question": "..."}
只输出 JSON，不要 markdown 围栏，不要任何额外文字。"""


def load_seed():
    items = []
    for line in SEED.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            idiom, gloss = line.split("|", 1)
        else:
            idiom, gloss = line, ""
        items.append({"idiom": idiom.strip(), "gloss": gloss.strip()})
    # dedup
    seen = set()
    uniq = []
    for it in items:
        if it["idiom"] in seen:
            continue
        seen.add(it["idiom"])
        uniq.append(it)
    return uniq


async def gen_one(client, sem, idiom, gloss):
    async with sem:
        user = f"典故：{idiom}\n（参考含义：{gloss}）\n\n请输出 JSON。"
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=600,
                    temperature=0.4,
                )
                txt = r.choices[0].message.content.strip()
                # strip markdown fences
                if txt.startswith("```"):
                    txt = txt.strip("`")
                    if txt.lower().startswith("json"):
                        txt = txt[4:]
                    txt = txt.strip()
                # find first { ... last }
                s = txt.find("{")
                e = txt.rfind("}")
                if s == -1 or e == -1:
                    raise ValueError(f"no json: {txt[:200]}")
                obj = json.loads(txt[s : e + 1])
                if not all(k in obj for k in ("idiom_prompt", "literal_prompt", "question")):
                    raise ValueError(f"missing keys: {obj.keys()}")
                # sanity: idiom must appear in idiom_prompt and NOT in literal_prompt
                if idiom not in obj["idiom_prompt"]:
                    raise ValueError(f"idiom '{idiom}' not in idiom_prompt")
                return {
                    "idiom": idiom,
                    "gloss": gloss,
                    "idiom_prompt": obj["idiom_prompt"].strip(),
                    "literal_prompt": obj["literal_prompt"].strip(),
                    "question": obj["question"].strip(),
                }
            except Exception as e:
                print(f"[{idiom}] attempt {attempt+1} failed: {e}", file=sys.stderr)
                await asyncio.sleep(2)
        return None


async def main():
    seed = load_seed()
    print(f"Loaded {len(seed)} seed idioms")
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120.0)
    sem = asyncio.Semaphore(CONCURRENCY)

    # already-done set so we can resume
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                done.add(obj["idiom"])
    todo = [it for it in seed if it["idiom"] not in done]
    print(f"{len(done)} done, {len(todo)} to do")

    tasks = [gen_one(client, sem, it["idiom"], it["gloss"]) for it in todo]
    f = open(OUT, "a")
    completed = 0
    for coro in asyncio.as_completed(tasks):
        r = await coro
        if r:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            completed += 1
            print(f"[{completed}/{len(todo)}] {r['idiom']}")
    f.close()


if __name__ == "__main__":
    asyncio.run(main())
