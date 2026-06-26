# Baseline 扩展 —— 免费 / 低价路线图

## 现状

10 个 baseline 已跑：4 个 Claude + 4 个国产 (glm-5, deepseek-3.2, minimax-m2.1, minimax-m2.5, qwen3-coder-next, Qwen3.5-35B-A3B)。

历史 result 中 `base_url` 都是 `http://127.0.0.1:8990/v1`，但这只是端口记录 —— 当前 `kcli-gw` 只路由到 Kiro/Claude（见 `kcli-gw/crates/kcli-core/src/kiro/provider.rs`），之前的 GLM/DeepSeek/MiniMax/Qwen 跑分应该是不同时间用别的多 provider 网关绑同一端口跑出来的。

## 严格免费路径（zero out-of-pocket）

| 选项 | 可加模型 | 阻力 | 备注 |
|------|---------|------|------|
| **本机 Ollama** | Llama-3.3-70B (Q4) / Qwen2.5-32B / Yi-9B / InternLM2.5-7B / Gemma-2 等 | 需先 `brew install ollama && ollama pull <model>`；M 系 Mac 跑 7-9B 流畅，30B+ 慢 | 完全本地，跑分质量取决于量化等级 |
| **HuggingFace Inference free tier** | 大量开源模型 | 每个 model rate-limit ~5 RPM，跑 100 题 × 6 任务 = 600 calls / 模型 ≈ 2 小时 | 偶尔 503，需要重试 |
| **OpenRouter free models** | gemma-2-9b-it, llama-3.2-3b-instruct, qwen-2.5-7b-instruct, hermes-3-llama-3.1-405b (limited) | 需 OpenRouter 帐号 + API key（注册免费） | 部分模型有日限额 ~50 calls |
| **Groq free tier** | Llama-3.3-70B, llama-3.1-70b-versatile, mixtral-8x7b | 需注册 + key；30 RPM / 6K TPM | 速度极快，是 Llama 跑分最便宜路径 |

## 低价路径（几美元 1 模型）

| Provider | 适合模型 | 估算 |
|---------|---------|------|
| **DeepSeek API** | deepseek-3.2 (重跑), deepseek-r1-0528 | ~$0.10 / 全 bench |
| **Together AI** | Yi-34B, InternLM-2.5, Llama-3.3, Qwen3 系列 | 注册送 $5；够跑 5-10 个模型 |
| **DeepInfra** | Yi-Lightning, InternLM3, Llama 全套 | 注册送少量额度，按 token 计费很便宜 |
| **Moonshot Kimi API** | Kimi K2 / kimi-latest | 国内手机号注册送试用额度 |
| **零一万物 Yi API** | yi-lightning / yi-large | 需中国大陆手机号 |
| **vLLM on g6e.xlarge** | 任意 HF 模型 | $1-2 / 小时；600 calls 大概 5-10 分钟，全程不到 $0.30 / 模型 |

## 优先推荐：免费 3 baseline

按"研究价值 ÷ 接入难度"排：

1. **Llama-3.3-70B（Groq 免费 tier）**
   - 西方主流开源旗舰，目前 bench 没有任何"非中文优先"模型作对照
   - canonicity gap 预测：T1 obscure 上应该明显低于 Qwen3.5-35B，T3 上差距更大 —— 验证"中文优先训练"假设
   - 接入：注册 → 拷贝 API key → `--base-url https://api.groq.com/openai/v1 --model llama-3.3-70b-versatile`

2. **InternLM2.5 / InternLM3（Together AI 免费额度 或 本地 Ollama）**
   - 上海 AI Lab 出的中文模型，传言古文训练偏多
   - canonicity gap 预测：T1 表现可能最强（如果传言属实）
   - 接入：Together AI 注册 → `meta-internlm/internlm2_5-20b-chat`

3. **Yi-Lightning（零一万物 API 试用 或 OpenRouter）**
   - 01.ai 旗舰，传言古文能力强
   - canonicity gap 预测：与 Qwen3.5 同梯队
   - 接入：OpenRouter 上 `01-ai/yi-lightning`

加完这 3 个，"开源中文模型 vs 西方模型 vs Claude" 三条线都齐了，canonicity gap 分析也有了第一个西方对照点（Llama）。

## 我能自动做什么

**不能**：登录任何外部 provider、注册账号、添加 API key、安装 Ollama —— 都需要用户操作。

**能**：
- 用户提供任何 OpenAI-兼容 endpoint + key 后，5 分钟内把 eval_runner 跑一遍 6 任务 / 1 模型
- 写好 vLLM-on-g6e 启动脚本（参考 `llm-eval` skill）
- 写 Ollama 一键脚本：`ollama pull <model> && python scripts/eval_runner.py --base-url http://localhost:11434/v1 --api-key dummy --model <model>`
- 跑分完毕后自动 rescore / regenerate leaderboard / 更新 canonicity 表 / 提 commit

## 阻塞与建议

**短路径**：用户挑 1 个 provider 注册 + 给我一个 API key，我把推荐的 3 个免费模型跑完，~30-60 分钟出新 leaderboard。

**长路径**：跑完 3 个后下一批要花钱（Yi-Large / Kimi K2 / GPT-5 / Gemini-3），每个 ≤$1，但需明确预算。
