# Chinese Classical Bench

[![license](https://img.shields.io/badge/code-MIT-yellow)](LICENSE) [![data](https://img.shields.io/badge/questions-from%20CC0%20corpus-lightgrey)](https://huggingface.co/datasets/gujilab/chinese-classical-corpus) [![HF dataset](https://img.shields.io/badge/%F0%9F%A4%97-dataset-blue)](https://huggingface.co/datasets/gujilab/chinese-classical-bench) [![HF leaderboard](https://img.shields.io/badge/%F0%9F%A4%97-leaderboard-orange)](https://huggingface.co/spaces/gujilab/chinese-classical-bench-leaderboard)

中国古典语言能力评测基准 (Classical Chinese benchmark) — **6 个任务 × 100 题 = 600 道**，覆盖翻译、断句、字义、典故、续写填空、压缩。

> 📊 在线排行榜: [🤗 Space — chinese-classical-bench-leaderboard](https://huggingface.co/spaces/gujilab/chinese-classical-bench-leaderboard)
> 🤗 本评测集也在 HuggingFace: [gujilab/chinese-classical-bench](https://huggingface.co/datasets/gujilab/chinese-classical-bench) — `load_dataset("gujilab/chinese-classical-bench", "translate")`
> 🔗 配套语料集: [gujilab/chinese-classical-corpus](https://huggingface.co/datasets/gujilab/chinese-classical-corpus) (CC0 公有领域)

## 为什么做这个

中文（尤其文言文）在 LLM 时代常被说成"高密度优势"。这套基础设施（bench + corpus + [experiments/](experiments/)）想把这个论点变成可验证的数字 —— **包括它在哪些场景成立、在哪些场景不成立**。

### Tokenizer 层面（实证：[tokenizer_study/](tokenizer_study/report.md)）
- 7 个主流 tokenizer 横评：DeepSeek-V3 / Qwen2.5 / Qwen3 上 **文言文 = 英文 0.57×、现代中文 0.69×**
- 老 GPT-3.5/4 (cl100k_base) **对中文比英文还费 19% token**
- 这一层的"省 token"是真的

### Prompt 层面（实证：[experiments/](experiments/)）

| 场景 | 字符/Token 节省 | 任务准确率/质量 | 结论 |
|---|---|---|---|
| **英文 prompt → 文言文 prompt**（中文域任务） | **−74%** 字符 | **+2pp** 准确率 | **大赢** |
| **现代中文 prompt → 文言文 prompt**（同任务改写） | −4.7% 字符 | **−11.3pp** 准确率 | 不是 free lunch |
| **典故 prompt vs 字面展开 prompt**（盲评 140 对） | −25% token | 字面赢 49% > 典故赢 38% | 省 token，质量有代价 |

### 校准结论
> 中文的高密度 = 真的 **tokenizer-level 优势**，但 ≠ **LLM-task-level free lunch**。
> 替换英文 prompt 是真节省；替换现代中文或用典故压缩，**省 token 是真，质量代价也是真**。
> 这个 nuance 之前自己也忽略了 —— 详细数据见 [experiments/prompt-compression](experiments/prompt-compression/report.md) 和 [experiments/idiom-prompting](experiments/idiom-prompting/report.md)。

### 时间深度（未实证）
3000+ 年单一书写系统，跨时代知识图谱、概念演化建模的训练信号只有中文有 —— 这一条尚未做实证 task，欢迎社区贡献。

---

现状问题：训练语料古典占比极低，公开 benchmark（CMMLU / C-Eval）几乎只评白话，没人针对古典的"高密度短文本"优化 tokenizer 或评测体系。本仓库 + 配套语料集是想把这条赛道做扎实的两个基础设施 —— 让 LLM 中文古典能力变得**可量化、可对比、可训练**。

具体到这个 bench，600 题在回答两个问题：

1. **哪些模型在中文古典任务上已经达到可用水平？** —— 给做古籍/教育/出版/法律/中医等应用的人选模型用
2. **不同能力维度（理解 / 操控 / 记忆 / 推理 / 压缩）的 ceiling 和 bottleneck 在哪？** —— 给训练/微调的人定方向用

主要面向 Qwen / DeepSeek / GLM / MiniMax / Yi / InternLM 等中文模型横评，也扩展到 Claude / GPT / Gemini 作为对照组。

## 6 个任务

| Task | 名称 | 题数 | 主要指标 |
|------|------|------|------|
| `translate` | 古译今 | 100 | chrF (n=1..6, F2) |
| `punctuate` | 断句加标点 | 100 | 标点位置 F1 + 字符保留率 |
| `char-gloss` | 字义解释 | 100 | chrF (punct-stripped) + char-F1 |
| `idiom-source` | 典故出处 | 100 | 书名 EM + 引文 chrF |
| `fill-in` | 字词填空 | 100 | 单字精确匹配 |
| `compress` | 现代汉语→文言文压缩 | 100 | chrF × (1 − ratio) = **efficiency** |

详见 [docs/tasks.md](docs/tasks.md)。

## Quick Start

```bash
# 1. 重生成题目（可选，已 commit 的题目就在 data/）
python scripts/build_translate.py
python scripts/build_punctuate.py
python scripts/build_char_gloss.py
python scripts/build_idiom_source.py
python scripts/build_fill_in.py
python scripts/build_compress.py

# 2. 跑评测（需先启动 vLLM endpoint）
python scripts/eval_runner.py \
  --model Qwen/Qwen3-7B-Instruct \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --concurrency 8

# 调试可加 --limit 5 --tasks translate fill-in
# 也支持 OpenAI / Anthropic 兼容端点

# 3. 看单模型结果
cat results/Qwen_Qwen3-7B-Instruct.json | jq '.tasks | to_entries[] | {task: .key, summary: .value.summary}'

# 4. 跑完多个模型后聚合排行榜
python scripts/aggregate.py --out leaderboard.md
```

## 文件布局

```
data/                  # 6 × 100 道题 (jsonl)
scripts/
  build_*.py           # 各 task 题目生成脚本
  eval_runner.py       # OpenAI 兼容 API 调用 + 打分
  scorers.py           # chrF / char-F1 / 标点 F1 / EM
  aggregate.py         # 多模型结果 → markdown 排行榜
results/               # 每个模型一个 json
docs/tasks.md          # 任务详细说明
```

## Leaderboard

完整带 95% 置信区间与 LLM judge 重排的版本见 [`leaderboard.md`](leaderboard.md)。

| Model | translate (chrF) | punctuate (Punct F1) | char-gloss (chrF) | idiom-source (Book EM) | fill-in (Exact) | compress (Eff) | Avg |
|---|---|---|---|---|---|---|---|
| **claude-opus-4-7** | **0.244** | **0.800** | **0.213** | 0.650 | 0.860 | **0.245** ⁑ | **0.502** |
| claude-opus-4-7-thinking | 0.242 | 0.790 | 0.207 | 0.630 | **0.880** | 0.194 ⁑ | 0.490 |
| claude-sonnet-4-6 | 0.231 | 0.785 | 0.157 | 0.560 | 0.700 | 0.163 | 0.433 |
| deepseek-3.2 | 0.240 | 0.745 | 0.139 | **0.740** | 0.550 | 0.163 | 0.429 |
| glm-5 | 0.241 | 0.799 | 0.176 | **0.740** | 0.450 | 0.153 | 0.427 |
| minimax-m2.1 | 0.216 | 0.709 | 0.173 | 0.660 | 0.630 | 0.094 | 0.414 |
| Qwen3.5-35B-A3B | 0.225 | 0.753 | 0.175 | 0.500 | 0.380 | — | 0.407 |
| minimax-m2.5 | 0.219 | 0.709 | 0.174 | 0.618 ⁑ | 0.590 | 0.092 | 0.400 |
| qwen3-coder-next | 0.227 | 0.767 | 0.116 | 0.540 | 0.520 | 0.113 | 0.380 |
| claude-haiku-4-5 | 0.204 | 0.729 | 0.128 | 0.340 | 0.350 | 0.087 | 0.306 |

> 上表 headline 为可复现的 chrF / EM / efficiency 点估计（`Avg` = 各 task headline 的简单平均）；完整的 **LLM-judge 重排 + 95% CI** 版本见 [`leaderboard.md`](leaderboard.md)（judge headline 下 Opus 4.7 ≈ 0.678）。bootstrap CI 由 `scripts/bootstrap_ci.py` (2000 iters, n=每 task 有效题数) 给出：**Opus 4.7 vs Opus thinking 的 Avg CI 重叠**——两个旗舰统计意义上不可区分；Sonnet / DeepSeek / GLM-5 同样多向重叠。
> `compress (Eff)` = chrF × (1 − ratio)，同时奖励压缩率和保真度。人类参考上限 ≈ 0.49。
> ⁑ **该 cell 经回溯修正**：原始运行中部分题 API 调用失败、被存成空预测并误判 0 分，污染了汇总。已把空预测标记为 error 并在剩余真实成功题上重算 —— `claude-opus-4-7` compress (n=60)、`claude-opus-4-7-thinking` compress (n=47)、`minimax-m2.5` idiom-source (n=89)。详见下方脚注与「已知 limitation」。

**核心发现**：

1. **Claude Opus 4.7 接管榜首**（chrF-headline Avg 0.502；judge-headline 0.678 见 `leaderboard.md`）—— **6 项里 5 项第一**：翻译 / 断句 / 字义 / 单字填空，外加**剔除 API 失败项重算后，压缩 efficiency 也跃居第一 (0.245)**，超过原先并列第一的 Sonnet / DeepSeek (0.163)；仅典故记忆 (0.65) 不是绝对最强。同代 Sonnet 4.6 仅 0.433、Haiku 4.5 仅 0.306 —— 同代差距巨大，说明"Claude 中文古典弱"是尺寸问题，不是数据问题
2. **Thinking 模式略逊，但"压缩崩塌"是数据假象，不是真退步** —— Opus 4.7 thinking 0.490 vs 非 thinking 0.502，多数 task 持平到略低。之前榜单上的 **"thinking 压缩崩塌 0.091"是 artifact**：该次运行 100 题里 **53 题 API 调用失败、被当 0 分**。**剔除失败项后，thinking compress 真实 efficiency = 0.194（n=47，chrF 0.43、ratio 0.55），plain = 0.245（n=60，ratio 0.52）—— 两者都健康，不存在崩塌**。该型号已下架无法重跑，但当前旗舰 **Opus 4.6 干净数据（0 错误）印证：thinking 0.182 ≈ plain 0.180、ratio≈0.54**，thinking 与 plain 在压缩上几乎相等。延长推理对"记忆 / 分寸"类任务大体无帮助，但谈不上"崩塌"
3. **压缩任务 Opus 4.7 修正后领先，"Opus 过度压缩"论断已撤销** —— 旧结论里"Opus ratio 仅 0.31、过度压缩"同样是 artifact：API 失败题 ratio 被记为 0 把均值拉低。**剔除后 Opus 4.7 ratio = 0.52、thinking = 0.55，正好落在人类基线 0.51 附近，并非过度压缩**；当前旗舰 Opus 4.6 干净数据 ratio≈0.54 进一步印证。efficiency 排序：**Opus 4.7 0.245 > thinking 0.194 > Sonnet 4.6 / DeepSeek V3.2 0.163 > glm-5 0.153**。**这个任务直接验证了"中文古典作为信息密度工具"的可行性 —— 顶级模型已能稳定输出 ~50% 压缩、保真度可接受的文言文**
4. **典故识别国产模型已被 GLM-5 追平 DeepSeek** —— DeepSeek V3.2 / GLM-5 并列 (0.74) > MiniMax-2.1 (0.66) > Opus 4.7 (0.65)；老一轮 DeepSeek 独占第一的护城河没了，但相对 Claude 旗舰仍领先 9 个百分点
5. **fill-in（单字填空）Claude 全家通吃** —— Opus thinking 0.88 / Opus 0.86 / Sonnet 0.70 / MiniMax-2.1 0.63 / DeepSeek 0.55，单字级中文古文恢复 Anthropic 系压倒性优势
6. **GLM-5 综合最好的国产开源**（0.427）—— 5/6 项进前三（仅 fill-in 弱），与 DeepSeek (0.429) 仅差 0.002
7. **Sonnet 4.6 性价比意外** —— avg 0.433，比 DeepSeek (0.429) 高一点，曾在压缩任务并列第一（0.163）；不过修正后 Opus 系压缩反超，Sonnet 退居 compress 第三。仍是 Claude 系性价比之选
8. **MiniMax M2.1 → M2.5 是中文古典 retrograde** —— 新版 m2.5 (0.400) 仍低于老版 m2.1 (0.414)。其 idiom-source 原始运行有 11/100 题 API 失败、被当 0 分；**剔除后 m2.5 idiom-source = 0.618（n=89）**，仍低于 m2.1 (0.66)；fill-in 也回退 (0.59 vs 0.63)。新版可能针对其他能力做了优化但损伤了中文古典记忆

> 欢迎提交其他模型结果（开 PR 把 `results/<model>.json` 加进来即可）。
> Sonnet/Opus、DeepSeek、Llama、ChatGLM 等正在补充中。

### 关于分数

- `chrF` 是字符级 n-gram F2 分数（n=1..6），同义改写会扣分但语义对的话主要靠 `char_f1` 兜底
- **`chrF` 与语义质量相关性 moderate** — [experiments/llm-judge](experiments/llm-judge/report.md) 现已扩展到 10 模型 × 2 任务 × 100 题、**Opus 4.7 + Sonnet 4.6 双 judge 交叉验证**。Inter-judge Cohen κ_quad：translate **0.775** / char-gloss **0.894**；model-mean Spearman ρ：translate **0.948** / char-gloss **0.979**——两个独立 judge 几乎完全同意排名，等于不依赖人工标注就拿到了可信的语义评分。**结论：chrF 是方向正确的下限，不是质量指标**。最大反差是 **DeepSeek-3.2 translate**：chrF 把它排第 2 (0.240)，judge 把它压到第 7 (0.75)——它的 chrF 高分是被专有名词共现拉起来的，并不真懂原意。详见 `leaderboard.md` 的 Judge-rescored 子表
- `idiom-source` 的 Book EM 较宽松：模型答 "《史记》" 就算对，不要求卷次/篇名匹配
- `fill-in` 单字答案，模型能从带引号或单字输出中抽取（详见 `scorers.py`）

## Experiments（论点实证）

`bench` 之外，本仓库还有 4 个 thesis 验证实验：

| 实验 | 论点 | 结果 |
|---|---|---|
| [tokenizer_study/](tokenizer_study/) | 中文是高密度语言？ | ✅ 国产 tokenizer 上文言文 = 英文 0.57× |
| [experiments/prompt-compression/](experiments/prompt-compression/) | 文言文 prompt 真省 token 又保准确率？ | HALF — vs 英文大赢，vs 现代中文反损 11pp |
| [experiments/llm-judge/](experiments/llm-judge/) | chrF 是好的质量指标？ | ❌ Pearson 0.46-0.47，重排后 Sonnet ↑、GLM ↓ |
| [experiments/idiom-prompting/](experiments/idiom-prompting/) | 典故 = 语义级 RAG 压缩？ | HALF — 省 25% token，但盲评字面版 49% > 典故 38% |

每个实验自带 README、可复现 script、原始 jsonl 数据、最终 report.md。结果不是单方向支持原命题，**有支持有修正** —— 这是命题该有的样子。

## 已知 limitation

- **部分早期运行有 API 失败题被误判 0 分（已回溯修正）** —— `claude-opus-4-7` compress 100 题里 40 题、`claude-opus-4-7-thinking` compress 53 题、`minimax-m2.5` char-gloss 7 题 / idiom-source 11 题，原始运行 API 调用失败、被存成空预测且**无 error 标记**，被打成全 0 分污染了汇总（空预测数 == 该 task 记录的 errors 数，已核对一致）。已用 `scripts/flag_empty_errors.py` 把这些题标记为 error 并在剩余真实成功题上重算 summary（compress n=60 / 47、char-gloss n=93、idiom-source n=89）。**opus-4-7 系列该型号已从网关下架、无法重跑**；当前旗舰 **Opus 4.6 干净数据（0 错误，仅跑了 compress）**作为参照印证，存于 [`results/_supplementary/`](results/_supplementary/)（不进主榜单以免出现只有 1/6 task 的稀疏行）—— Opus 4.6 compress efficiency=0.180 / thinking=0.182、ratio≈0.54，与 4.7 修正后的结论一致。
- `translate` / `char-gloss` 用 chrF 评分，对同义改写过严 — **已加 LLM judge 实验：[experiments/llm-judge](experiments/llm-judge/)**（Pearson 0.46-0.47，建议结合使用而非替换）
- 6 个 task 题目均从配套 corpus 抽样，可能与某些模型的训练数据有重合污染（开源模型大多训练过《十三经》《史记》）
- 100 题/task 是 trade-off：太少噪声大，太多跑评测贵 — 后续可能扩到 200/task
- **`idiom-source` 有一版 contamination-robust 重采样 `data/idiom_source.v2.jsonl`（已生成、尚未启用）** —— 它从 Tier-1 冷门书目重新抽样，意在让分数反映"古文能力"而非"认出《论语》/《史记》"（详见 `scripts/build_idiom_source_v2.py` 与 `docs/findings.md` §6）。**当前 `eval_runner.py` 仍读 v1**（`data/idiom_source.jsonl`）：切到 v2 会让所有已存 `results/*.json` 的 idiom-source 预测失效、必须全量重跑该任务，故暂不切换。v2 文件保留待后续做一次专门的 idiom-source rerun 时再启用，不是被遗忘的半成品。
- **31 题有数据质量问题**，已用 `metadata._audit_issue` 标注（含 char-gloss 18 题 gold 为字典占位符 `同本义。`，由题目区分度分析发现）。详见 [docs/quality-audit.md](docs/quality-audit.md)，方法与全部发现见 **[docs/findings.md](docs/findings.md)**（题目难度/区分度心理测量学审计，零成本回溯；含[污染探针](docs/contamination.md) `idiom-source` ρ=0.68 与[任务冗余分析](docs/task-redundancy.md)）。可通过 `ds.filter(lambda x: not x['metadata'].get('_audit_issue'))` 过滤。题目未删除以保持已有 result 文件兼容

## Contributing

想让你的模型上榜：跑 `scripts/eval_runner.py` → 把 `results/<model>.json` + 重新生成的 `leaderboard.md` 一起开 PR。CI 会自动校验 schema 并核对 `leaderboard.md`。详见 [CONTRIBUTING.md](CONTRIBUTING.md)。也欢迎改进打分器（chrF 对同义改写偏严、典故书名匹配可以更聪明）—— 结果文件里存了每题原始 prediction，打分器改了能对所有已有模型回溯重打分。

## License

题目和评分代码：**MIT**（见 [LICENSE](LICENSE)）。源数据来自 [chinese-classical-corpus](https://huggingface.co/datasets/gujilab/chinese-classical-corpus) (CC0)。
