# Collation 任务（校勘 / textual collation）—— prototype v0

## 一句话

测试 LLM 能否解析中国古典 三家注 体系中的校勘标记惯例 (`一作` / `俗本作` / `当作` / `讹作` …) 并据此判断主文应取的正字。

## 为什么做这个

现有的 chinese-classical-bench 六个任务 (translate / punctuate / char-gloss / idiom-source / fill-in / compress) 覆盖了"译注 + 句读 + 训诂 + 用典 + 词法 + 简化"，但缺一项中国古典文献学最经典的技能：**校勘**——在传世异文之间根据校勘惯例判定正字。
现有的中文 LLM 评测中没有此类任务。素材天然存在于二十四史的三家注和资治通鉴胡注里。

## v0 设计

**输入** (`input`)：含校勘异文标记的一段 60-120 字窗口（保留 `【...】` 注文）。

**输出** (`reference`)：主文中的正字（单字）。

**任务定义**：模型读窗口，按下列校勘学惯例判定主文中的正字。

### 校勘标记规则（v0 共 8 类）

| 标记 | 语义 | 正字侧 | 例 | 池中数 |
|------|------|--------|----|--------|
| `X一作Y` | 主作 X，他本作 Y | X | 卿一作庆 → 卿 | 437 |
| `X一本作Y` | 主作 X，一本作 Y | X | 端一本作从 → 端 | 38 |
| `X或作Y` | 主作 X，或本作 Y | X | 雒或作雄 → 雒 | 213 |
| `X俗本作Y` | 主作 X，俗本作 Y（俗本通常非是） | X | 击 俗本作鹰 → 击 | 8 |
| `X旧本作Y` | 主作 X，旧本作 Y | X | 同 旧本作熹 → 同 | 8 |
| `X讹作Y` | 注家报告：被讹抄为 Y | X | 夷讹作不 → 夷 | 24 |
| `X当作Y` | 注家校改：受 X 当为 Y | **Y** | 莫当作幕 → 幕 | 202 |
| `X当为Y` | 注家校改：受 X 当为 Y | **Y** | 为当为易 → 易 | 30 |

总计候选池 ~960 条；v0 抽 50 条，按 `sqrt(pool)` 加权分层确保多样性。

### 关键 design 决策

1. **保留 `【...】` 注文不剥离**——任务考的是"能否理解校勘标记的约定"，而非凭空校勘。剥离注文就成 fill-in 的特例。
2. **双向歧义**——`一作/或作` 系列 → X 是正字；`当作/当为` 系列 → Y 是正字。这两族方向相反，无法用"永远选主文字"的简单策略蒙混。
3. **避坑 marker**——`又作` / `别作` / `应作` / `应为` 在古文中常作非校勘义（"也做"/"另作"/"应该是"），即使加 cue 过滤也噪音偏高，v0 不收。
4. **当作 / 当为 加 cue 过滤**——必须 ±20 字内有 案/疑/讹/监本/非是/者误/误也/之误/字误/改正/脱字/脱文/衍文/本作/本误/误字 这类校勘语，否则视为常用义（如 "颛顼当为金徳"），剔除。
5. **NOISE_X 黑名单**——剔除 `曰一作Y` / `本或作Y` / `案当作Y` 这类引语虚词被误抓为 X 的情况。

## 输出 schema (`data/collation.jsonl`)

```json
{
  "id": "collation#1",
  "task": "collation",
  "instruction": "下列古文出自传世史书三家注，其中含有校勘异文标记。请按校勘学惯例（一作 / 俗本作 / 当作 / 讹作 …）判断主文中的正字（仅一字），直接给出该字。",
  "input": "...60-120字窗口...",
  "reference": "X",
  "metadata": {
    "source": "史记",
    "category": "史",
    "marker": "一作",
    "variant": "Y",
    "rec_id": "shiji#42"
  }
}
```

## 评分（`scripts/scorers.py:score_collation`）

与 `fill-in` 同构 —— 单字精确匹配 + 简繁归一：

- `exact_match`：从模型输出中抽取首个/被引号包裹的汉字，t2s 归一后比对 reference
- `in_pred`：reference 在模型输出中（宽松，用于 sanity check）

## 数据来源分布

50 题源于：史记 / 后汉书 / 三国志 / 资治通鉴 / 北齐书 / 北史 / 等含三家注或胡注的史书。集中于 `classical-corpus/output/corpus.jsonl` 中标注的 1506 条（占 12.5%）校勘标记记录。

## 已知局限（v1+ todo）

1. **校改方向歧义**：`X当作Y` 中 X、Y 都是正常字，但有时校改的不是单字而是单词、词序、句法关系。v0 只取单字校。
2. **元注记忆**：开源 LLM 训练语料里大量包含三家注本身，部分题可能"已经见过原文"——污染评测需要类似 `idiom-source` 的 canonicity 分层（参 `docs/contamination.md`）。
3. **校勘三大类未拆分**：v0 只测"判定正字"，未测"识别讹/脱/衍/倒类型"。v1 计划增加 4-way classification 子任务。
4. **N=50** 是原型，置信区间会宽（预计 ±0.10）。v1 扩到 200 条与其他任务对齐。

## 复现

```bash
cd chinese-classical-bench
python scripts/build_collation.py    # 重建 data/collation.jsonl
```

构建用固定种子 `random.Random(42)`，可复现。

## v0 状态

- ✅ data/collation.jsonl (50 题)
- ✅ scripts/build_collation.py
- ✅ scripts/scorers.py: `score_collation`
- ✅ scripts/eval_runner.py: 加入 TASK_FILES
- ⏳ baseline 跑分待补（重跑 10 个现有模型，每个 ~50 calls × $cost；未列入 v0 prototype 范围）
- ⏳ leaderboard.md 集成（待跑过 baseline 再加）
