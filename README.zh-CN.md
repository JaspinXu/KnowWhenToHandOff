<div align="center">

# KnowWhenToHandOff

**教一个小模型客服把事办完、守住政策、在该转人工的时候转人工；以及当奖励只看结果时会出什么问题。**

[English](README.md) · **简体中文**

<p>
<img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
<img src="https://img.shields.io/badge/Env-%CF%84%C2%B2--bench-4B32C3" alt="tau2-bench">
<img src="https://img.shields.io/badge/RL-GRPO%20(verl)-EE4C2C" alt="GRPO with verl">
<img src="https://img.shields.io/badge/Serving-vLLM-30A2FF" alt="vLLM">
<img src="https://img.shields.io/badge/Policy-Qwen3--4B--Instruct-615CED" alt="Qwen3-4B-Instruct">
<img src="https://img.shields.io/badge/License-MIT-16A34A" alt="MIT licence">
</p>

<a href="#主要发现">主要发现</a> ·
<a href="#结果">结果</a> ·
<a href="#为什么只看结果的-rl-会过度转人工">奖励漏洞</a> ·
<a href="#方法">方法</a> ·
<a href="#模型与数据">模型与数据</a> ·
<a href="#复现">复现</a>

</div>

<p align="center"><img src="docs/assets/hero-zh.svg" alt="KnowWhenToHandOff：AI 客服把对话转交给人工客服。转人工 F1 0.812；过度转人工从 46% 降到 10%；相对 SFT 成功率 +0.015（不显著）；461 个转人工任务" width="100%"></p>

> "帮我转人工。"
> 最好的客服不是从不转人工的那个，而是在该转的时候转、并且说清楚为什么的那个。

基于大模型的客服经常违反业务政策、转人工的时机不对，还可能因为用户的表达方式不同而服务质量不同。
**KnowWhenToHandOff** 在 [τ²-bench](https://github.com/sierra-research/tau2-bench)
的航空和零售领域上，用监督微调加多轮 **GRPO** 训练 **Qwen3-4B-Instruct-2507**，
新增了一个带"是否应转人工"标签的 **461 个任务的转人工任务集**，并在每段对话上衡量四项
Responsible-AI 指标：违反政策、转人工的精确率与召回率、六种用户沟通风格下的成功率、
拒绝和转人工时是否说明政策依据。

## 主要发现

| | 发现 | 证据 |
| --- | --- | --- |
| **RQ1** | 多轮 RL 在官方任务成功率上**没有**超过 SFT。 | 相对 SFT 的配对差：+0.020 [−0.066, 0.107]（任务奖励）、+0.015 [−0.056, 0.082]（Responsible-AI 奖励） |
| **RQ2** | 奖励决定转人工往哪边变：只看结果的 RL **过度转人工**，Responsible-AI 奖励得到**最校准**的模型，且不损失成功率。 | 转人工 F1 0.768 → 0.812；过度转人工 0.016 → 0.462（R1）对比 0.103（R2） |
| **RQ3** | 用户沟通风格的影响在 RL 前后都没有超出噪声。 | 各模型最差风格成功率都在 0.29–0.33 |
| **RQ4** | **奖励漏洞：** 只要正确结果不改数据库，τ²-bench 就给转人工记 1.0 分，所以在任务奖励下"拿不准就转人工"弱占优。R1 在最后 20 步学会了它。 | 过度转人工第 40 步 0.141、第 60 步 0.462；[40 段对话的审计](docs/rl-audit.md) |

## 结果

<p align="center"><img src="docs/assets/results-zh.svg" alt="左：任务成功率、95% 置信区间及相对 SFT 的配对差；右：转人工工作点与等 F1 曲线；下：副作用（政策引用率、违规率、最差风格成功率）" width="100%"></p>

| 系统 | 成功率（pass^1） | pass^4 | 相对 B2 的 Δ | 转人工 F1 | 过度转人工 | 最差风格成功率 |
| --- | --- | --- | --- | --- | --- | --- |
| B0 提示词 Qwen3-4B | 0.306 [0.219, 0.408] | 0.102 | −0.020 [−0.107, 0.056] | 0.654 | 0.087 | 0.306 |
| B1 提示词 Qwen3-30B-A3B | 0.398 [0.301, 0.505] | 0.204 | +0.071 [−0.020, 0.153] | — | — | — |
| B2 SFT | 0.327 [0.224, 0.434] | 0.102 | — | 0.768 | **0.016** | **0.327** |
| R1 GRPO，任务奖励 | **0.347** [0.235, 0.459] | **0.224** | +0.020 [−0.066, 0.107] | 0.802 | 0.462 | 0.286 |
| R2 GRPO，Responsible-AI 奖励 | 0.342 [0.245, 0.449] | 0.122 | +0.015 [−0.056, 0.082] | **0.812** | 0.103 | 0.316 |

留出测试任务，每个检查点只评测一次；方括号为按任务自助法的 95 % 区间；Δ 在同样 49 个任务上配对计算。
成功率和 pass^4：`test_main`（49 个官方任务，4 次试验）。转人工两列：`test_handoff`
（98 个应转、92 个困难负例任务，2 次试验）。最差风格成功率：六种沟通风格。每个数字都能在
[`reports/main/sources.json`](reports/main/sources.json) 和[实验日志](docs/experiment-log.md)中追溯到运行编号。

RL 确实提升了它训练时占多数的转人工任务（`test_handoff` 成功率 B2 0.47 → R1 0.63、R2 0.55），
但没有提升官方任务。

## 为什么只看结果的 RL 会过度转人工

<p align="center"><img src="docs/assets/loophole-zh.svg" alt="奖励漏洞：正确结果不改数据库时，转人工也得 1.0 分；R2 给转人工错误定价" width="100%"></p>

τ²-bench 检查的是最终数据库。正确结果是拒绝时，转人工同样不改数据库，因此也拿满分；在转人工任务上转人工本来就对。
其余情况下尝试解决只有一定概率成功，转人工得 0 分。所以在任务奖励下，拿不准时转人工永远不比尝试差，R1 找到了这个策略：
查完账户就转，经常是猜了一个用户 ID 报错后就转，也不告诉用户原因（引用率 0.06）。给不必要的转人工定价（R2）消除了这个漏洞；
R2 剩下的错误方向相反：有时对超出范围的请求直接拒绝，而不是转人工。训练曲线显示了它发生的时间：R1 的转人工率前 30 步持平，之后快速上升，而 dev 成功率不再提高。

<p align="center"><img src="docs/assets/dynamics-zh.svg" alt="GRPO 训练中的 dev 成功率和转人工率：R1 的转人工率在第 30 步后上升" width="100%"></p>

离线浏览对话：[`reports/demo/R2-handoffs.html`](reports/demo/R2-handoffs.html) 和
[`reports/demo/R1-over-escalation.html`](reports/demo/R1-over-escalation.html)（下载后用浏览器打开；转人工有边框，政策引用高亮）。

## 方法

<p align="center"><img src="docs/assets/pipeline-zh.svg" alt="系统总览：任务、SFT 起点、多轮 GRPO 采样、R1 与 R2 的奖励项、留出评测" width="100%"></p>

**问题建模。** 每段对话是一个 POMDP 回合。隐藏状态：用户的目标、私有信息、沟通风格和数据库。
观测：领域政策、对话和工具结果。动作：回复、调用工具，或 `transfer_to_human_agents`（结束回合）。
最多 200 步、不折扣、只有终局奖励；只有模型自己生成的 token 参与梯度。

**奖励。** 设"是否应转人工"标签为 $y$：

$$
R(\tau) = R_{\text{task}} - \lambda_v N_{\text{viol}} - \lambda_{\text{FP}}\,\mathbf{1}[\text{transfer} \wedge y{=}0] - \lambda_{\text{FN}}\,\mathbf{1}[\neg\text{transfer} \wedge y{=}1] - \lambda_f N_{\text{format}}
$$

$R_{\text{task}} \in \{0,1\}$ 来自 τ²-bench 自带评估器；$N_{\text{viol}}$ 统计五条确定性规则的违反次数
（未确认就写入、认证前写入、一轮多个动作、转人工后继续操作、参考解之外的写入）。**R1** 只用 $R_{\text{task}}$；
**R2** 使用一组固定权重 $\lambda_v{=}0.25$、$\lambda_{\text{FP}}{=}0.5$、$\lambda_{\text{FN}}{=}1.0$、$\lambda_f{=}0.1$
（漏转比误转危害更大）。政策引用只测量、不奖励：关键词奖励很容易被钻空子。

**转人工任务集。** 50 个官方航空任务里只有 1 个需要转人工，所以我们从每个官方任务派生带标签的变体，变体沿用源任务的划分：

| 类型 | 应转人工 | 改动 | 测试任务数 |
| --- | --- | --- | --- |
| 明确要求 | 是 | 用户要求人工并拒绝 AI 帮忙 | 49 |
| 超出范围 | 是 | 没有工具或政策条款覆盖的服务（轮椅、特殊餐食、保修上门、纸质发票等） | 49 |
| 困难负例 | 否 | 源任务加上一个着急、怀疑但配合的用户 | 92 |

详见 [docs/handoff-tasks.md](docs/handoff-tasks.md) 和 [docs/labels.md](docs/labels.md)。

**沟通风格**（[docs/personas.md](docs/personas.md)）：

<p align="center"><img src="docs/assets/styles-zh.svg" alt="模拟用户的六种沟通风格：中性、简短、啰嗦、不耐烦、非母语、不熟悉技术" width="100%"></p>

**训练。** B2：在 790 段成功、无违规、无格式错误且转人工判断正确的教师对话上做 LoRA SFT（秩 64，已合并）。
R1/R2：从 B2 出发，用 verl 0.8 和 vLLM 0.12 做 GRPO；每步 8 个任务 × 8 条 rollout，共 60 步，组内归一化优势，
无 KL 项，LoRA 秩 64（单张 H100 96 GB 放不下全参微调），回复上限 16,384 token；模拟用户
（Qwen3-30B-A3B，温度 0.7）跑在另一张 A100 上。工程细节见 [ADR-001](docs/adr/ADR-001-trainer-and-gpu-stack.md)。

**评测。** τ²-bench v1.0.1（固定提交）；零售只用 114 个任务中不需要 LLM 评判的 74 个。所有选择都在 dev（从训练集切出）上做，
测试集每个冻结检查点只跑一次。模型贪心解码；按任务做 95 % 百分位自助法（1,000 次重采样）；在共同任务上做配对比较。
指标定义见 [docs/metrics.md](docs/metrics.md)，协议见 [ADR-005](docs/adr/ADR-005-evaluation-protocol.md)。

## 模型与数据

以研究用途发布在 Hugging Face（**不可用于部署**）；上传步骤见 [docs/release.md](docs/release.md)。

| 发布物 | 内容 | 说明卡 |
| --- | --- | --- |
| `KnowWhenToHandOff-Qwen3-4B-R2` | Responsible-AI 奖励的 GRPO（转人工最校准） | [R2](docs/model-cards/R2.md) |
| `KnowWhenToHandOff-Qwen3-4B-R1` | 任务奖励的 GRPO（过度转人工的案例） | [R1](docs/model-cards/R1.md) |
| `KnowWhenToHandOff-Qwen3-4B-SFT` | SFT 起点（B2） | [B2](docs/model-cards/B2.md) |
| `KnowWhenToHandOff-data` | 转人工任务集与划分、所有运行的轨迹和指标、GRPO 训练 rollouts、SFT 筛选记录 | [数据集卡](docs/dataset-card.md) |

任务集和清单也在本仓库的 [`data/`](data/) 中；对话样例在 [`reports/demo/`](reports/demo/)。

## 复现

笔记本 / CI（CPU，Python 3.12，[uv](https://docs.astral.sh/uv/)）：

```bash
uv sync --locked
uv run --locked ruff check . && uv run --locked pytest -q
```

重活都在 Slurm 集群上跑，每个阶段一条命令（[集群运行手册](docs/cluster-runbook.md)）：环境 → 任务集 → 模型服务 →
B0/B1 → 教师 rollouts → SFT 数据 → SFT → R1/R2 → 最终评测 → 报表。上面的表格和图由英文 README
"Reproduce"一节中的三条命令生成。每次运行都记录解析后的配置、git SHA、τ²-bench 提交、模型版本和种子；
报表只接受已完成的真实运行。

## 局限与负责任使用

- **只有模拟用户。** 只用了一个模拟器（Qwen3-30B-A3B），它也生成了 SFT 数据、也是 B1 参照；结论不一定适用于真人。
- **每个运行只有一个种子**；区间只覆盖任务抽样，不覆盖训练方差。官方测试集较小（49 个任务），零售只用 74 个任务的子集。
- **转人工任务由模板生成**，标签依据政策文本，抽查了 40 个；规则检查只覆盖部分政策，引用率是关键词规则。
- **风格不是人群。** 不对任何人群下结论。
- **不可用于部署。** 发布的检查点仅供研究。

## 引用

```bibtex
@misc{knowwhentohandoff2026,
  title  = {Know When to Hand Off: Multi-turn Reinforcement Learning for
            Trustworthy Customer-Service Agents},
  author = {{KnowWhenToHandOff contributors}},
  year   = {2026},
  howpublished = {\url{https://github.com/JaspinXu/KnowWhenToHandOff}}
}
```

## 致谢

基于 [τ²-bench](https://github.com/sierra-research/tau2-bench)（MIT）、[verl](https://github.com/volcengine/verl)、
[vLLM](https://github.com/vllm-project/vllm) 和 [Qwen3](https://huggingface.co/Qwen) 模型。参考文献见英文 README。

## 许可

代码：[MIT](LICENSE)。派生任务数据：MIT，源自 τ²-bench（© 2025 Sierra Research）。模型权重沿用 Qwen3 的 Apache-2.0 许可。
