"""
Judge: for each (scenario, model) pair, compare idiom-version answer vs literal-version
answer. Use claude-opus-4-7 as judge. Randomize A/B order.

Output: judge_results.jsonl
  { idiom, model, judge, winner: "idiom"|"literal"|"tie", reason, ordering: "iA_lB"|"lA_iB" }
"""
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from collections import defaultdict

from openai import AsyncOpenAI

BASE_URL = os.environ.get("KCLI_BASE_URL", "http://localhost:8990/v1")
API_KEY = os.environ.get("KCLI_API_KEY")
if not API_KEY:
    sys.exit("KCLI_API_KEY not set — export it before running (no default key).")
JUDGE = os.environ.get("JUDGE_MODEL", "claude-opus-4-7")
CONCURRENCY = 4

HERE = Path(__file__).parent
SCEN = HERE / "scenarios.jsonl"
RES = HERE / "results.jsonl"
OUT = HERE / "judge_results.jsonl"

JUDGE_SYS = """你是一位中文文本质量评审专家。

给定一个场景描述和一个问题，以及两位 AI 模型针对同一问题给出的两份回答（标为 A 和 B），请客观比较两份回答的质量。

评判维度（按重要性）：
1. 对场景的理解深度（是否抓住核心矛盾、识别策略本质）
2. 分析的洞察力（是否给出有价值的判断，而非泛泛之谈）
3. 结构清晰度与可操作性
4. 简洁有力（在同等内容下更精炼者优）

请输出 JSON：{"winner": "A" / "B" / "Tie", "reason": "一句话说明判断理由（不超过 60 字）"}

只输出 JSON，不要 markdown 围栏。"""


def load_pairs():
    scen = {s["idiom"]: s for s in (json.loads(l) for l in SCEN.read_text().splitlines() if l.strip())}
    results = [json.loads(l) for l in RES.read_text().splitlines() if l.strip()]
    by = defaultdict(dict)
    for r in results:
        if r.get("answer"):
            by[(r["idiom"], r["model"])][r["version"]] = r
    pairs = []
    for (idiom, model), v in by.items():
        if "idiom" in v and "literal" in v:
            pairs.append({"idiom": idiom, "model": model, "scenario": scen[idiom], "i": v["idiom"], "l": v["literal"]})
    return pairs


def load_done():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                done.add((o["idiom"], o["model"]))
    return done


async def judge_one(client, sem, pair):
    async with sem:
        # randomize ordering
        rng = random.Random(hash((pair["idiom"], pair["model"])) & 0xFFFFFFFF)
        idiom_is_A = rng.random() < 0.5
        if idiom_is_A:
            A_ans, B_ans = pair["i"]["answer"], pair["l"]["answer"]
            ordering = "iA_lB"
        else:
            A_ans, B_ans = pair["l"]["answer"], pair["i"]["answer"]
            ordering = "lA_iB"

        # Both versions share the SAME question; show only ONE prompt (the question)
        # to avoid leaking which version is which via prompt text.
        question = pair["scenario"]["question"]
        scenario_brief = pair["scenario"]["idiom_prompt"]  # show idiom version to judge; question only matters

        user = (
            f"场景：{scenario_brief}\n\n"
            f"问题：{question}\n\n"
            f"=== 回答 A ===\n{A_ans}\n\n"
            f"=== 回答 B ===\n{B_ans}\n\n"
            f"请评判。"
        )

        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=JUDGE,
                    messages=[
                        {"role": "system", "content": JUDGE_SYS},
                        {"role": "user", "content": user},
                    ],
                    max_tokens=200,
                    temperature=0.0,
                )
                txt = r.choices[0].message.content.strip()
                if txt.startswith("```"):
                    txt = txt.strip("`")
                    if txt.lower().startswith("json"):
                        txt = txt[4:]
                    txt = txt.strip()
                s = txt.find("{")
                e = txt.rfind("}")
                obj = json.loads(txt[s : e + 1])
                w = obj.get("winner", "").strip()
                if w not in ("A", "B", "Tie"):
                    raise ValueError(f"bad winner: {w}")
                if w == "Tie":
                    winner = "tie"
                elif (w == "A" and idiom_is_A) or (w == "B" and not idiom_is_A):
                    winner = "idiom"
                else:
                    winner = "literal"
                return {
                    "idiom": pair["idiom"],
                    "model": pair["model"],
                    "judge": JUDGE,
                    "winner": winner,
                    "winner_raw": w,
                    "ordering": ordering,
                    "reason": obj.get("reason", ""),
                }
            except Exception as e:
                print(f"[{pair['idiom']}/{pair['model']}] judge attempt {attempt+1}: {e}", file=sys.stderr)
                await asyncio.sleep(3 + attempt * 2)
        return None


async def main():
    pairs = load_pairs()
    done = load_done()
    print(f"{len(pairs)} pairs to judge; {len(done)} already done")
    todo = [p for p in pairs if (p["idiom"], p["model"]) not in done]
    print(f"To run: {len(todo)} judge calls")

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=180.0)
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [judge_one(client, sem, p) for p in todo]

    f = open(OUT, "a")
    done_count = 0
    for coro in asyncio.as_completed(tasks):
        r = await coro
        if r:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            done_count += 1
            print(f"[{done_count}/{len(todo)}] {r['idiom']}/{r['model']} winner={r['winner']}")
    f.close()


if __name__ == "__main__":
    asyncio.run(main())
