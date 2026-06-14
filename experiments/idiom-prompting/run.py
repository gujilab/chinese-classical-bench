"""
Run experiment: 3 models × 2 versions × N scenarios.

For each scenario × model × version:
- Build full prompt: <prompt> + "\n\n" + question
- Send to model, capture output + API token usage
- Also compute tiktoken cl100k_base token count for fair comparison
  (since kcli-gw adds a large server-side system prompt the prompt_tokens
  from API is not a clean signal).

Output: results.jsonl
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import tiktoken
from openai import AsyncOpenAI

BASE_URL = os.environ.get("KCLI_BASE_URL", "http://localhost:8990/v1")
API_KEY = os.environ.get("KCLI_API_KEY")
if not API_KEY:
    sys.exit("KCLI_API_KEY not set — export it before running (no default key).")
MODELS = ["claude-opus-4-7", "deepseek-3.2", "qwen3-coder-next"]
CONCURRENCY = 8
MAX_TOKENS = 700  # answer budget

HERE = Path(__file__).parent
IN = HERE / "scenarios.jsonl"
OUT = HERE / "results.jsonl"

ENC = tiktoken.get_encoding("cl100k_base")


def build_prompt(scenario, version):
    body = scenario["idiom_prompt"] if version == "idiom" else scenario["literal_prompt"]
    return f"{body}\n\n问题：{scenario['question']}"


def load_scenarios():
    return [json.loads(l) for l in IN.read_text().splitlines() if l.strip()]


def load_done():
    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                done.add((o["idiom"], o["model"], o["version"]))
    return done


async def call_one(client, sem, scenario, model, version):
    async with sem:
        prompt = build_prompt(scenario, version)
        tt_tokens = len(ENC.encode(prompt))
        char_len = len(prompt)
        for attempt in range(3):
            try:
                t0 = time.time()
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=MAX_TOKENS,
                    temperature=0.3,
                )
                dt = time.time() - t0
                ans = r.choices[0].message.content.strip()
                usage = getattr(r, "usage", None)
                return {
                    "idiom": scenario["idiom"],
                    "model": model,
                    "version": version,
                    "prompt": prompt,
                    "answer": ans,
                    "tt_prompt_tokens": tt_tokens,
                    "prompt_char_len": char_len,
                    "api_prompt_tokens": usage.prompt_tokens if usage else None,
                    "api_completion_tokens": usage.completion_tokens if usage else None,
                    "latency_s": round(dt, 2),
                }
            except Exception as e:
                err = str(e)[:200]
                print(
                    f"[{scenario['idiom']}/{model}/{version}] attempt {attempt+1}: {err}",
                    file=sys.stderr,
                )
                await asyncio.sleep(3 + attempt * 2)
        return {
            "idiom": scenario["idiom"],
            "model": model,
            "version": version,
            "prompt": prompt,
            "answer": None,
            "tt_prompt_tokens": tt_tokens,
            "prompt_char_len": char_len,
            "api_prompt_tokens": None,
            "api_completion_tokens": None,
            "latency_s": None,
            "error": "failed_after_retries",
        }


async def main():
    scenarios = load_scenarios()
    done = load_done()
    print(f"Loaded {len(scenarios)} scenarios; {len(done)} runs already done")

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=180.0)
    sem = asyncio.Semaphore(CONCURRENCY)

    todo = []
    for s in scenarios:
        for m in MODELS:
            for v in ("idiom", "literal"):
                if (s["idiom"], m, v) not in done:
                    todo.append((s, m, v))

    print(f"To run: {len(todo)} calls (3 models × 2 versions × {len(scenarios)} scenarios)")
    tasks = [call_one(client, sem, s, m, v) for (s, m, v) in todo]

    f = open(OUT, "a")
    completed = 0
    for coro in asyncio.as_completed(tasks):
        r = await coro
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        completed += 1
        status = "OK" if r.get("answer") else "FAIL"
        print(f"[{completed}/{len(todo)}] {r['idiom']}/{r['model']}/{r['version']} {status}")
    f.close()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
