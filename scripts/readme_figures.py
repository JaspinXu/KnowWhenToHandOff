# ruff: noqa: E501  (HTML/SVG templates and figure text are kept on one line)
"""Render the README figures (docs/assets/*.png) from HTML/SVG templates.

Every figure exists in English and Chinese and in a light and a dark theme
(``<name>-<lang>-<theme>.png``); the READMEs pick the theme with ``<picture>``.
Numbers come from committed report tables (``reports/main/main_table.csv``,
``reports/rl/dev_curves.csv``) or, for the test_handoff precision/recall, from
the runs' ``metrics.json`` values listed in ``HANDOFF`` below (also in the model
cards). Fonts are the OFL-licensed files in ``docs/assets/fonts`` (IBM Plex Sans
and Mono, Source Serif 4) plus Noto CJK from the system for Chinese.

Needs Python Playwright with Chromium:

    python scripts/readme_figures.py            # all figures
    python scripts/readme_figures.py --only loophole
"""

from __future__ import annotations

import argparse
import csv
import html
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
FONTS = ASSETS / "fonts"
WIDTH = 1280

# test_handoff (98 should-transfer + 92 hard-negative tasks, 2 trials):
# B0-test_handoff-s0-20261004, B2-test_handoff-s0-20261005,
# R1-final-test_handoff-s0-20261006, R2-final-test_handoff-s0-20261007.
HANDOFF = {
    "B0": {"recall": 0.526, "over": 0.087, "f1": 0.654},
    "B2": {"recall": 0.633, "over": 0.016, "f1": 0.768},
    "R1": {"recall": 0.959, "over": 0.462, "f1": 0.802},
    "R2": {"recall": 0.750, "over": 0.103, "f1": 0.812},
}

THEMES = {
    "light": {
        "bg": "#F6F2EA",
        "ink": "#1F1B16",
        "muted": "#6B645A",
        "faint": "#A39C90",
        "card": "#FFFFFF",
        "card-bd": "#E4DCCD",
        "grid": "#E7E0D3",
        "on-ac": "#FFFFFF",
        "amber-bg": "#FBEFD9",
        "amber-bd": "#E8C68A",
        "amber": "#A8641A",
        "green-bg": "#E7EFE1",
        "green-bd": "#AFC7A1",
        "green": "#3F6B35",
        "blue-bg": "#E4EAF5",
        "blue-bd": "#A3B6DA",
        "blue": "#2F5FA8",
        "rose-bg": "#F8E2DA",
        "rose-bd": "#E2A893",
        "rose": "#B8461F",
        "lilac-bg": "#ECE7F3",
        "lilac-bd": "#BDB0D7",
        "lilac": "#5B4A8B",
        "B0": "#8E8A82",
        "B1": "#C9C2B5",
        "B2": "#2F5FA8",
        "R1": "#C8501E",
        "R2": "#1F7A6B",
    },
    "dark": {
        "bg": "#14171C",
        "ink": "#ECE7DE",
        "muted": "#A39E95",
        "faint": "#6E6A63",
        "card": "#1E232A",
        "card-bd": "#343B45",
        "grid": "#2A3038",
        "on-ac": "#14171C",
        "amber-bg": "#2A2419",
        "amber-bd": "#6B5329",
        "amber": "#E0A04F",
        "green-bg": "#1D2820",
        "green-bd": "#44603F",
        "green": "#8CC27D",
        "blue-bg": "#1B2333",
        "blue-bd": "#3E5683",
        "blue": "#7EA3E6",
        "rose-bg": "#2C1E1B",
        "rose-bd": "#7A4434",
        "rose": "#EE8A63",
        "lilac-bg": "#241F30",
        "lilac-bd": "#54477A",
        "lilac": "#B3A2E6",
        "B0": "#8F8B84",
        "B1": "#5B5850",
        "B2": "#6F97E0",
        "R1": "#EE8A63",
        "R2": "#4FBFA9",
    },
}

# --------------------------------------------------------------------------- text
T = {
    "en": {
        "brand": "KnowWhenToHandOff",
        "pipeline.sub": "system overview",
        "legend.solid": "data / model flow",
        "legend.dash": "evaluation only",
        "c1": ("Tasks", "DATA"),
        "c1.a": (
            "τ²-bench airline + retail",
            "50 + 74 tasks · official splits · LLM-judged retail tasks excluded",
        ),
        "c1.b": ("Hand-off variants", "461 derived tasks · inherit the source split"),
        "c1.chips": [
            "explicit request · y=1",
            "out of scope · y=1",
            "hard negative · y=0",
        ],
        "c1.c": ("Simulated user", "Qwen3-30B-A3B · temperature 0.7"),
        "c1.styles": "communication styles",
        "c1.chips2": [
            "neutral",
            "terse",
            "verbose",
            "impatient",
            "non-native",
            "low-tech",
        ],
        "c2": ("Imitate", "SFT WARM START"),
        "c2.a": ("Teacher rollouts", "Qwen3-30B-A3B on train · 6 styles × 2 trials"),
        "c2.b": ("Filter", "success ∧ no violation ∧ correct hand-off"),
        "c2.c": ("790 dialogues", "assistant-token loss only"),
        "c2.d": ("B2 · SFT", "Qwen3-4B · LoRA r64, merged"),
        "c3": ("Roll out", "MULTI-TURN GRPO"),
        "c3.a": ("Policy", "vLLM inside verl · H100 96 GB"),
        "c3.b": ("Simulated user", "separate vLLM · A100 80 GB"),
        "c3.c": ("Tools + database", "τ²-bench domain"),
        "c3.d": ("8 tasks × 8 rollouts", "group-normalised advantage · 60 steps"),
        "c3.e": ("Only agent tokens learn", "user and tool tokens masked"),
        "c4": ("Reward", "R1 · R2"),
        "c4.a": ("R_task ∈ {0, 1}", "τ²-bench checks the final database"),
        "c4.terms": [
            ("− 0.25 · violations", "5 deterministic rules"),
            ("− 0.5 · unneeded transfer", "y = 0"),
            ("− 1.0 · missed transfer", "y = 1"),
            ("− 0.1 · format errors", "malformed tool calls"),
        ],
        "c4.r1": "task reward only",
        "c4.r2": "all five terms",
        "c5": ("Evaluate", "HELD-OUT TEST"),
        "c5.a": ("test_main", "49 official tasks × 4 trials"),
        "c5.b": ("test_styles", "× 5 styles × 2 trials"),
        "c5.c": ("test_handoff", "190 hand-off tasks × 2 trials"),
        "c5.chips": [
            "pass^k",
            "hand-off P / R",
            "over-escalation",
            "worst style",
            "citation",
        ],
        "c5.d": ("Once per checkpoint", "95% bootstrap over tasks · paired vs SFT"),
        "base.title": "provenance",
        "base": [
            "dev split for every decision",
            "git SHA · τ² commit · model revisions · seeds",
            "Slurm · one command per stage",
        ],
        "pipeline.cap": (
            "Figure 4.",
            "From τ²-bench tasks to three trained agents. Imitation gives a warm start; GRPO rolls out whole conversations with a simulated user; R1 and R2 differ only in the reward.",
        ),
        "loop.sub": "the reward loophole",
        "loop.h1": "What the task reward pays",
        "loop.col": ["Situation", "Transfer", "Try to solve"],
        "loop.rows": [
            ("User should get a human", "explicit request · out of scope", "1.0", "0"),
            (
                "Correct outcome is a refusal",
                "no database change, e.g. insurance after booking",
                "1.0",
                "1.0",
            ),
            ("Task needs a database write", "most official tasks", "0", "p"),
        ],
        "loop.verdict": "When p is low, transferring is never worse → “transfer when unsure” is weakly dominant.",
        "loop.h2": "Same test dialogue · airline task 6",
        "loop.user": "I need insurance added to my flight right away … please fix this now.",
        "loop.r1": "R1 · task reward",
        "loop.r1.say": "transfer_to_human_agents(…) after one account lookup",
        "loop.r1.tag": "reward 1.0 · counted as over-escalation",
        "loop.r2": "R2 · Responsible-AI reward",
        "loop.r2.say": "“The system does not allow adding insurance after the initial booking … This is a policy restriction.”",
        "loop.r2.tag": "reward 1.0 · cited refusal",
        "loop.h3": "R2 prices the hand-off",
        "loop.fix": [
            ("Refusal task, transfer", "1.0 − 0.5 = 0.5 < 1.0"),
            ("Should-transfer task, no transfer", "0 − 1.0 = −1.0"),
        ],
        "loop.big1": ("Over-escalation", "0.462", "0.103"),
        "loop.big2": ("Hand-off F1", "0.802", "0.812"),
        "loop.cap": (
            "Figure 2.",
            "An outcome check cannot tell an unnecessary hand-off from a correct refusal when neither changes the database. R1 exploited this late in training; pricing both hand-off errors (R2) removes it.",
        ),
        "res.sub": "results on held-out test tasks",
        "res.left": "Task success (pass¹) · test_main",
        "res.left.note": "95% bootstrap CI · all overlap",
        "res.right": "Hand-off operating points · test_handoff",
        "res.x": "over-escalation → transfers users who did not need a human",
        "res.y": "recall → transfers users who need one",
        "res.cap": (
            "Figure 1.",
            "RL does not raise official-task success (left), but the reward decides where hand-offs land (right): R1 trades precision for recall, R2 reaches the best F1 with little over-escalation.",
        ),
        "dyn.sub": "training dynamics",
        "dyn.left": "Dev success",
        "dyn.right": "Dev transfer rate",
        "dyn.note": "step 40 checkpoint, tested post hoc: over-escalation 0.141 (vs 0.462 at step 60)",
        "dyn.x": "GRPO step",
        "dyn.cap": (
            "Figure 3.",
            "Greedy dev evaluation every 10 steps (60 dialogues, all task kinds). R1's transfer rate stays flat for 30 steps, then climbs while success no longer improves.",
        ),
        "stats.sub": "at a glance",
        "stats": [
            ("HAND-OFF F1 · R2", "0.812", "", "SFT 0.768 · prompted 0.654"),
            ("OVER-ESCALATION", "46", "% → 10%", "task reward (R1) → RAI reward (R2)"),
            (
                "SUCCESS VS SFT",
                "+0.015",
                "",
                "95% CI [−0.056, 0.082] · not significant",
            ),
            ("HAND-OFF TASKS", "461", "", "derived from τ²-bench, with y labels"),
            ("RL DIALOGUES / RUN", "3,840", "", "60 steps × 8 tasks × 8 rollouts"),
        ],
        "stats.foot": "Qwen3-4B-Instruct-2507 · τ²-bench airline + retail · one seed per run",
        "styles.sub": "simulated users",
        "styles": [
            ("neutral", "control: the task's own persona"),
            ("terse", "short, keeps required details"),
            ("verbose", "fuller phrasing, same facts"),
            ("impatient", "brisk tone, still cooperates"),
            ("non-native", "simple vocabulary, same meaning"),
            ("low-tech-literacy", "asks what technical terms mean"),
        ],
        "styles.cap": (
            "Figure 5.",
            "Six communication styles injected into the simulated user's persona. They change how the user writes, never the task facts, and describe expression, not who a person is. Illustration made with an image model.",
        ),
        "names": {
            "B0": "B0 prompted 4B",
            "B1": "B1 prompted 30B",
            "B2": "B2 SFT",
            "R1": "R1 GRPO · task",
            "R2": "R2 GRPO · RAI",
        },
    },
    "zh": {
        "brand": "KnowWhenToHandOff",
        "pipeline.sub": "系统总览",
        "legend.solid": "数据 / 模型流",
        "legend.dash": "仅评测",
        "c1": ("任务", "DATA"),
        "c1.a": (
            "τ²-bench 航空 + 零售",
            "50 + 74 个任务 · 官方划分 · 剔除需 LLM 评判的零售任务",
        ),
        "c1.b": ("转人工变体", "派生 461 个任务 · 沿用源任务划分"),
        "c1.chips": ["明确要求 · y=1", "超出范围 · y=1", "困难负例 · y=0"],
        "c1.c": ("模拟用户", "Qwen3-30B-A3B · 温度 0.7"),
        "c1.styles": "沟通风格",
        "c1.chips2": ["中性", "简短", "啰嗦", "不耐烦", "非母语", "不熟悉技术"],
        "c2": ("模仿", "SFT WARM START"),
        "c2.a": ("教师 rollouts", "Qwen3-30B-A3B 跑训练集 · 6 种风格 × 2 次"),
        "c2.b": ("筛选", "成功 ∧ 无违规 ∧ 转人工判断正确"),
        "c2.c": ("790 段对话", "只在助手 token 上算损失"),
        "c2.d": ("B2 · SFT", "Qwen3-4B · LoRA r64，已合并"),
        "c3": ("采样", "MULTI-TURN GRPO"),
        "c3.a": ("策略模型", "verl 内的 vLLM · H100 96 GB"),
        "c3.b": ("模拟用户", "独立 vLLM · A100 80 GB"),
        "c3.c": ("工具 + 数据库", "τ²-bench 领域"),
        "c3.d": ("8 个任务 × 8 条 rollout", "组内归一化优势 · 60 步"),
        "c3.e": ("只有模型 token 参与学习", "用户和工具 token 被屏蔽"),
        "c4": ("奖励", "R1 · R2"),
        "c4.a": ("R_task ∈ {0, 1}", "τ²-bench 检查最终数据库"),
        "c4.terms": [
            ("− 0.25 · 违规次数", "5 条确定性规则"),
            ("− 0.5 · 不必要的转人工", "y = 0"),
            ("− 1.0 · 漏掉的转人工", "y = 1"),
            ("− 0.1 · 格式错误", "工具调用格式不对"),
        ],
        "c4.r1": "只用任务奖励",
        "c4.r2": "五项全用",
        "c5": ("评测", "HELD-OUT TEST"),
        "c5.a": ("test_main", "49 个官方任务 × 4 次"),
        "c5.b": ("test_styles", "× 5 种风格 × 2 次"),
        "c5.c": ("test_handoff", "190 个转人工任务 × 2 次"),
        "c5.chips": ["pass^k", "转人工 P / R", "过度转人工", "最差风格", "引用率"],
        "c5.d": ("每个检查点只评一次", "按任务 95% 自助法 · 与 SFT 配对比较"),
        "base.title": "可追溯",
        "base": [
            "所有选择都在 dev 上做",
            "git SHA · τ² 提交 · 模型版本 · 种子",
            "Slurm · 每个阶段一条命令",
        ],
        "pipeline.cap": (
            "图 4.",
            "从 τ²-bench 任务到三个训练好的模型。模仿学习提供起点；GRPO 让模型与模拟用户完整对话；R1 和 R2 只差在奖励。",
        ),
        "loop.sub": "奖励漏洞",
        "loop.h1": "任务奖励给多少分",
        "loop.col": ["情形", "转人工", "尝试解决"],
        "loop.rows": [
            ("用户应该转人工", "明确要求 · 超出范围", "1.0", "0"),
            ("正确结果是拒绝", "不改数据库，如订票后补买保险", "1.0", "1.0"),
            ("任务需要写数据库", "大多数官方任务", "0", "p"),
        ],
        "loop.verdict": "当 p 较低时，转人工永远不吃亏 →“拿不准就转人工”弱占优。",
        "loop.h2": "同一段测试对话 · 航空任务 6",
        "loop.user": "我要马上给我的航班加上保险……现在就给我办。",
        "loop.r1": "R1 · 任务奖励",
        "loop.r1.say": "查了一次账户就 transfer_to_human_agents(…)",
        "loop.r1.tag": "奖励 1.0 · 计为过度转人工",
        "loop.r2": "R2 · Responsible-AI 奖励",
        "loop.r2.say": "“系统不允许在初次预订后添加保险……这是政策限制。”",
        "loop.r2.tag": "奖励 1.0 · 说明依据的拒绝",
        "loop.h3": "R2 给转人工定价",
        "loop.fix": [
            ("应拒绝的任务却转人工", "1.0 − 0.5 = 0.5 < 1.0"),
            ("应转人工却没转", "0 − 1.0 = −1.0"),
        ],
        "loop.big1": ("过度转人工", "0.462", "0.103"),
        "loop.big2": ("转人工 F1", "0.802", "0.812"),
        "loop.cap": (
            "图 2.",
            "当转人工和正确拒绝都不改数据库时，只看结果的检查无法区分二者。R1 在训练后期钻了这个空子；给两种转人工错误都定价（R2）就堵住了它。",
        ),
        "res.sub": "留出测试集结果",
        "res.left": "任务成功率（pass¹）· test_main",
        "res.left.note": "95% 自助法区间 · 全部重叠",
        "res.right": "转人工的工作点 · test_handoff",
        "res.x": "过度转人工 → 把不需要人工的用户也转走",
        "res.y": "召回 → 转走了需要人工的用户",
        "res.cap": (
            "图 1.",
            "RL 没有提升官方任务成功率（左），但奖励决定了转人工落在哪里（右）：R1 用精确率换召回，R2 在很少过度转人工的情况下拿到最高 F1。",
        ),
        "dyn.sub": "训练过程",
        "dyn.left": "Dev 成功率",
        "dyn.right": "Dev 转人工率",
        "dyn.note": "第 40 步检查点事后测试：过度转人工 0.141（第 60 步为 0.462）",
        "dyn.x": "GRPO 步数",
        "dyn.cap": (
            "图 3.",
            "每 10 步在 dev 上贪心评测一次（60 段对话，包含所有任务类型）。R1 的转人工率前 30 步持平，之后快速上升，而成功率不再提高。",
        ),
        "stats.sub": "一览",
        "stats": [
            ("转人工 F1 · R2", "0.812", "", "SFT 0.768 · 提示词 0.654"),
            ("过度转人工", "46", "% → 10%", "任务奖励（R1）→ RAI 奖励（R2）"),
            ("相对 SFT 的成功率", "+0.015", "", "95% 区间 [−0.056, 0.082] · 不显著"),
            ("转人工任务", "461", "", "由 τ²-bench 派生，带 y 标签"),
            ("每次 RL 的对话数", "3,840", "", "60 步 × 8 任务 × 8 rollouts"),
        ],
        "stats.foot": "Qwen3-4B-Instruct-2507 · τ²-bench 航空 + 零售 · 每个运行一个种子",
        "styles.sub": "模拟用户",
        "styles": [
            ("中性", "对照：任务自带的人设"),
            ("简短", "回答简短，必要信息不缺"),
            ("啰嗦", "表达更长，事实不变"),
            ("不耐烦", "语气急，但仍配合"),
            ("非母语", "用词简单，意思不变"),
            ("不熟悉技术", "会问技术术语是什么意思"),
        ],
        "styles.cap": (
            "图 5.",
            "注入模拟用户人设的六种沟通风格。它们只改变用户怎么说话，不改任务事实；描述的是表达方式，不代表任何人群。插图由图像模型生成。",
        ),
        "names": {
            "B0": "B0 提示词 4B",
            "B1": "B1 提示词 30B",
            "B2": "B2 SFT",
            "R1": "R1 GRPO · 任务",
            "R2": "R2 GRPO · RAI",
        },
    },
}

# --------------------------------------------------------------------------- base


def e(text: str) -> str:
    """Escape text for HTML."""
    return html.escape(str(text))


def css(theme: dict[str, str]) -> str:
    """Page CSS for one theme."""
    variables = "".join(f"--{key}:{value};" for key, value in theme.items())
    fonts = FONTS.resolve().as_uri()
    return f"""
@font-face{{font-family:Plex;font-weight:400;src:url({fonts}/ibm-plex-sans-latin-400-normal.woff2)}}
@font-face{{font-family:Plex;font-weight:500;src:url({fonts}/ibm-plex-sans-latin-500-normal.woff2)}}
@font-face{{font-family:Plex;font-weight:600;src:url({fonts}/ibm-plex-sans-latin-600-normal.woff2)}}
@font-face{{font-family:Plex;font-weight:700;src:url({fonts}/ibm-plex-sans-latin-700-normal.woff2)}}
@font-face{{font-family:PlexMono;font-weight:400;src:url({fonts}/ibm-plex-mono-latin-400-normal.woff2)}}
@font-face{{font-family:PlexMono;font-weight:500;src:url({fonts}/ibm-plex-mono-latin-500-normal.woff2)}}
@font-face{{font-family:Serif;font-weight:200 900;src:url({fonts}/source-serif-4-latin-wght-normal.woff2)}}
@font-face{{font-family:Serif;font-style:italic;font-weight:200 900;src:url({fonts}/source-serif-4-latin-wght-italic.woff2)}}
:root{{{variables}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg)}}
#fig{{width:{WIDTH}px;padding:30px 36px 26px;background:var(--bg);color:var(--ink);
  font-family:Plex,"Noto Sans CJK SC",sans-serif;font-size:13px;line-height:1.35}}
.sans{{font-family:Plex,"Noto Sans CJK SC",sans-serif}}
.serif{{font-family:Serif,"Noto Serif CJK SC",serif}}
.mono{{font-family:PlexMono,"Noto Sans CJK SC",monospace}}
.head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px}}
.brand{{display:flex;align-items:center;gap:10px;font-family:PlexMono,"Noto Sans CJK SC",monospace;
  letter-spacing:.16em;font-size:12.5px;text-transform:uppercase;color:var(--muted)}}
.brand b{{color:var(--ink);font-weight:500}}
.legend{{display:flex;gap:18px;font-size:12px;color:var(--muted);align-items:center}}
.legend i{{display:inline-block;width:26px;border-top:1.6px solid var(--muted);margin-right:7px;vertical-align:middle}}
.legend i.d{{border-top-style:dashed}}
.cap{{margin-top:16px;font-size:13.5px;color:var(--muted);line-height:1.5}}
.cap b{{color:var(--ink);font-weight:600;margin-right:4px}}
.panel{{border:1.5px solid var(--pbd);background:var(--pbg);border-radius:18px;padding:14px 14px 16px}}
.p-amber{{--pbg:var(--amber-bg);--pbd:var(--amber-bd);--ac:var(--amber)}}
.p-green{{--pbg:var(--green-bg);--pbd:var(--green-bd);--ac:var(--green)}}
.p-blue{{--pbg:var(--blue-bg);--pbd:var(--blue-bd);--ac:var(--blue)}}
.p-rose{{--pbg:var(--rose-bg);--pbd:var(--rose-bd);--ac:var(--rose)}}
.p-lilac{{--pbg:var(--lilac-bg);--pbd:var(--lilac-bd);--ac:var(--lilac)}}
.ph{{display:flex;gap:10px;align-items:center;margin-bottom:12px}}
.num{{width:28px;height:28px;border-radius:50%;background:var(--ac);color:var(--on-ac);display:flex;
  align-items:center;justify-content:center;font-weight:600;font-size:14px;flex:none}}
.pt{{font-family:Serif,"Noto Serif CJK SC",serif;font-size:23px;font-weight:600;line-height:1.05}}
.kick{{font-family:PlexMono,monospace;font-size:10.5px;letter-spacing:.16em;color:var(--ac);margin-top:3px}}
.card{{background:var(--card);border:1px solid var(--card-bd);border-radius:11px;padding:9px 11px}}
.card h4{{margin:0;font-size:13.5px;font-weight:600;line-height:1.3}}
.card p{{margin:3px 0 0;font-size:11.5px;color:var(--muted);line-height:1.4}}
.card.hl{{border:1.6px solid var(--ac);box-shadow:0 0 0 3px color-mix(in srgb,var(--ac) 14%,transparent)}}
.stack{{display:flex;flex-direction:column;gap:7px}}
.down{{text-align:center;color:var(--ac);font-size:14px;line-height:1;margin:1px 0}}
.chips{{display:flex;flex-wrap:wrap;gap:5px}}
.chip{{border:1px solid var(--card-bd);background:var(--card);border-radius:999px;padding:3px 9px;font-size:11px;white-space:nowrap}}
.lab{{font-size:11px;color:var(--muted);margin:6px 0 4px;letter-spacing:.04em}}
.arrow{{display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:20px}}
"""


LOGO = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--R1)" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M4 5h11a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H9l-4 3v-3H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z"/>'
    '<path d="M16 19h4m0 0-2-2m2 2-2 2"/></svg>'
)


def head(t: dict, sub: str, legend: str = "") -> str:
    """Brand line with the figure's subtitle and an optional legend."""
    return (
        f'<div class="head"><div class="brand">{LOGO}<b>{e(t["brand"])}</b>'
        f"<span>/</span><span>{e(sub)}</span></div>{legend}</div>"
    )


def caption(pair: tuple[str, str]) -> str:
    return f'<div class="cap"><b>{e(pair[0])}</b>{e(pair[1])}</div>'


def card(title: str, body: str = "", extra: str = "") -> str:
    body_html = f"<p>{e(body)}</p>" if body else ""
    return f'<div class="card{extra}"><h4>{e(title)}</h4>{body_html}</div>'


def chips(items: list[str]) -> str:
    return (
        '<div class="chips">'
        + "".join(f'<span class="chip">{e(i)}</span>' for i in items)
        + "</div>"
    )


def panel(color: str, number: int, pair: tuple[str, str], inner: str) -> str:
    return (
        f'<div class="panel p-{color}"><div class="ph"><div class="num">{number}</div>'
        f'<div><div class="pt">{e(pair[0])}</div><div class="kick">{e(pair[1])}</div></div></div>'
        f"{inner}</div>"
    )


# --------------------------------------------------------------------------- figures


def fig_pipeline(t: dict) -> str:
    """Figure 1: system overview."""
    legend = (
        f'<div class="legend"><span><i></i>{e(t["legend.solid"])}</span>'
        f'<span><i class="d"></i>{e(t["legend.dash"])}</span></div>'
    )
    c1 = (
        '<div class="stack">'
        + card(*t["c1.a"])
        + card(*t["c1.b"])
        + chips(t["c1.chips"])
    )
    c1 += (
        card(*t["c1.c"])
        + f'<div class="lab">{e(t["c1.styles"])}</div>'
        + chips(t["c1.chips2"])
        + "</div>"
    )
    c2 = (
        '<div class="stack">'
        + card(*t["c2.a"])
        + '<div class="down">↓</div>'
        + card(*t["c2.b"])
    )
    c2 += '<div class="down">↓</div>' + card(*t["c2.c"]) + '<div class="down">↓</div>'
    c2 += card(*t["c2.d"], extra=" hl") + "</div>"
    loop = (
        '<div style="display:grid;grid-template-columns:1fr 18px 1fr;gap:4px;align-items:center">'
        + card(*t["c3.a"], extra=" hl")
        + '<div class="arrow" style="font-size:16px">⇄</div>'
        + card(*t["c3.b"])
        + "</div>"
    )
    c3 = '<div class="stack">' + loop + '<div class="down">↕</div>' + card(*t["c3.c"])
    c3 += '<div class="down">↓</div>' + card(*t["c3.d"]) + card(*t["c3.e"]) + "</div>"
    terms = "".join(
        f'<div style="padding:6px 0;border-top:1px dashed var(--card-bd)">'
        f'<div class="mono" style="font-size:12px;font-weight:500">{e(a)}</div>'
        f'<div style="font-size:11px;color:var(--muted)">{e(b)}</div></div>'
        for a, b in t["c4.terms"]
    )
    c4 = '<div class="stack">' + card(*t["c4.a"], extra=" hl")
    c4 += f'<div class="card" style="padding:6px 11px">{terms}</div>'
    c4 += (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:4px">'
        f'<div class="card" style="text-align:center;border-color:var(--R1)"><h4 style="color:var(--R1)">R1</h4><p>{e(t["c4.r1"])}</p></div>'
        f'<div class="card" style="text-align:center;border-color:var(--R2)"><h4 style="color:var(--R2)">R2</h4><p>{e(t["c4.r2"])}</p></div>'
        "</div></div>"
    )
    c5 = '<div class="stack">' + card(*t["c5.a"]) + card(*t["c5.b"]) + card(*t["c5.c"])
    c5 += chips(t["c5.chips"]) + card(*t["c5.d"], extra=" hl") + "</div>"
    cols = (
        panel("amber", 1, t["c1"], c1)
        + '<div class="arrow">→</div>'
        + panel("green", 2, t["c2"], c2)
        + '<div class="arrow">→</div>'
        + panel("blue", 3, t["c3"], c3)
        + '<div class="arrow">⇄</div>'
        + panel("rose", 4, t["c4"], c4)
        + '<div class="arrow" style="opacity:.7">⇢</div>'
        + panel("lilac", 5, t["c5"], c5)
    )
    base = "".join(f'<span class="chip">{e(x)}</span>' for x in t["base"])
    return (
        head(t, t["pipeline.sub"], legend)
        + '<div style="display:grid;grid-template-columns:1fr 20px 1fr 20px 1.08fr 20px 1fr 20px 1fr;align-items:stretch">'
        + cols
        + "</div>"
        + '<div style="margin-top:14px;border:1.5px dashed var(--card-bd);border-radius:14px;padding:10px 14px;'
        'display:flex;gap:12px;align-items:center">'
        f'<span class="mono" style="font-size:11px;letter-spacing:.14em;color:var(--muted);text-transform:uppercase">{e(t["base.title"])}</span>'
        f'<div class="chips">{base}</div></div>' + caption(t["pipeline.cap"])
    )


def fig_loophole(t: dict) -> str:
    """Figure 3: why outcome-only RL over-escalates."""
    col = t["loop.col"]
    rows = ""
    for i, (title, sub, transfer, solve) in enumerate(t["loop.rows"]):
        hot = i == 1
        tstyle = (
            "background:var(--rose);color:var(--on-ac);border-color:var(--rose)"
            if hot
            else "background:var(--card)"
        )
        rows += (
            '<div style="display:grid;grid-template-columns:1.7fr .8fr .8fr;gap:6px;margin-top:6px">'
            f'<div class="card"><h4>{e(title)}</h4><p>{e(sub)}</p></div>'
            f'<div class="card serif" style="display:flex;align-items:center;justify-content:center;'
            f'font-size:26px;font-weight:600;{tstyle}">{e(transfer)}{" !" if hot else ""}</div>'
            f'<div class="card serif" style="display:flex;align-items:center;justify-content:center;'
            f'font-size:26px;font-weight:600;{"font-style:italic" if solve == "p" else ""}">{e(solve)}</div></div>'
        )
    left = (
        '<div class="panel p-rose"><div class="pt" style="margin-bottom:8px">'
        f"{e(t['loop.h1'])}</div>"
        '<div style="display:grid;grid-template-columns:1.7fr .8fr .8fr;gap:6px;font-size:11px;'
        'color:var(--muted);letter-spacing:.06em;text-transform:uppercase" class="mono">'
        + "".join(f"<span>{e(c)}</span>" for c in col)
        + "</div>"
        + rows
        + f'<div style="margin-top:12px;font-size:13.5px;font-weight:600;line-height:1.45">{e(t["loop.verdict"])}</div></div>'
    )
    middle = (
        '<div class="panel p-blue"><div class="pt" style="margin-bottom:10px">'
        f"{e(t['loop.h2'])}</div>"
        '<div style="display:flex;justify-content:flex-end"><div class="card" style="max-width:88%;'
        f'background:var(--blue-bg);border-color:var(--blue-bd)"><p style="color:var(--ink);font-size:12.5px;margin:0">{e(t["loop.user"])}</p></div></div>'
        '<div class="card" style="margin-top:10px;border-left:4px solid var(--R1)">'
        f'<h4 style="color:var(--R1)">{e(t["loop.r1"])}</h4>'
        f'<p class="mono" style="color:var(--ink)">{e(t["loop.r1.say"])}</p>'
        f'<p style="margin-top:6px"><span class="chip" style="border-color:var(--R1);color:var(--R1)">{e(t["loop.r1.tag"])}</span></p></div>'
        '<div class="card" style="margin-top:8px;border-left:4px solid var(--R2)">'
        f'<h4 style="color:var(--R2)">{e(t["loop.r2"])}</h4>'
        f'<p style="color:var(--ink)">{e(t["loop.r2.say"])}</p>'
        f'<p style="margin-top:6px"><span class="chip" style="border-color:var(--R2);color:var(--R2)">{e(t["loop.r2.tag"])}</span></p></div></div>'
    )
    fixes = "".join(
        f'<div class="card" style="margin-top:6px"><h4>{e(a)}</h4>'
        f'<p class="mono" style="color:var(--ink);font-size:12.5px">{e(b)}</p></div>'
        for a, b in t["loop.fix"]
    )

    def big(item: tuple[str, str, str]) -> str:
        label, before, after = item
        return (
            '<div style="margin-top:12px"><div class="mono" style="font-size:10.5px;letter-spacing:.14em;'
            f'color:var(--muted);text-transform:uppercase">{e(label)}</div>'
            '<div class="serif" style="font-size:36px;font-weight:700;line-height:1.1">'
            f'<span style="color:var(--R1)">{e(before)}</span>'
            '<span style="color:var(--muted);font-size:24px;font-weight:400"> → </span>'
            f'<span style="color:var(--R2)">{e(after)}</span></div></div>'
        )

    right = (
        '<div class="panel p-green"><div class="pt" style="margin-bottom:4px">'
        f"{e(t['loop.h3'])}</div>"
        + fixes
        + big(t["loop.big1"])
        + big(t["loop.big2"])
        + "</div>"
    )
    return (
        head(t, t["loop.sub"])
        + '<div style="display:grid;grid-template-columns:1.25fr 1.2fr .9fr;gap:14px;align-items:stretch">'
        + left
        + middle
        + right
        + "</div>"
        + caption(t["loop.cap"])
    )


def _interval(cell: str) -> tuple[float, float, float]:
    point, rest = cell.split(" [")
    low, high = rest.rstrip("]").split(", ")
    return float(point), float(low), float(high)


def _svg_text(
    x, y, text, size=11, anchor="start", fill="var(--muted)", weight=400, cls="sans"
):
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" '
        f'font-weight="{weight}" class="{cls}">{e(text)}</text>'
    )


def fig_results(t: dict) -> str:
    """Figure 2: success bars with CIs and hand-off operating points."""
    with open(ROOT / "reports/main/main_table.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    w, h, left, bottom, top = 560, 330, 44, 40, 16
    plot_h = h - bottom - top
    ymax = 0.6
    bars = ""
    for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        y = top + plot_h * (1 - v / ymax)
        bars += f'<line x1="{left}" x2="{w - 6}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/>'
        bars += _svg_text(left - 8, y + 4, f"{v:.1f}", 10.5, "end")
    slot = (w - left - 6) / len(rows)
    for i, row in enumerate(rows):
        label = row["Run"]
        point, low, high = _interval(row["Success (pass^1)"])
        cx = left + slot * (i + 0.5)
        bw = slot * 0.56
        y = top + plot_h * (1 - point / ymax)
        bars += (
            f'<rect x="{cx - bw / 2:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{top + plot_h - y:.1f}" '
            f'rx="6" fill="var(--{label})"/>'
        )
        ylo = top + plot_h * (1 - low / ymax)
        yhi = top + plot_h * (1 - high / ymax)
        bars += (
            f'<path d="M{cx:.1f} {ylo:.1f}V{yhi:.1f}M{cx - 7:.1f} {ylo:.1f}h14M{cx - 7:.1f} {yhi:.1f}h14" '
            'stroke="var(--ink)" stroke-width="1.4" fill="none"/>'
        )
        bars += _svg_text(
            cx,
            y - 8 if high - point < 0.02 else yhi - 7,
            f"{point:.3f}",
            11.5,
            "middle",
            "var(--ink)",
            600,
        )
        bars += _svg_text(
            cx, h - 20, t["names"][label].split(" ")[0], 12, "middle", "var(--ink)", 600
        )
        bars += _svg_text(
            cx, h - 6, " ".join(t["names"][label].split(" ")[1:]), 10, "middle"
        )
    left_svg = f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">{bars}</svg>'

    sw, sh, sl, sb, st = 600, 330, 52, 44, 14
    ph, pw = sh - sb - st, sw - sl - 16
    x0, x1, y0, y1 = 0.0, 0.5, 0.45, 1.0

    def sx(v):
        return sl + pw * (v - x0) / (x1 - x0)

    def sy(v):
        return st + ph * (1 - (v - y0) / (y1 - y0))

    sc = ""
    for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        sc += f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{st}" y2="{st + ph}" stroke="var(--grid)"/>'
        sc += _svg_text(sx(v), st + ph + 16, f"{v:.1f}", 10.5, "middle")
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        sc += f'<line x1="{sl}" x2="{sl + pw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="var(--grid)"/>'
        sc += _svg_text(sl - 8, sy(v) + 4, f"{v:.1f}", 10.5, "end")
    sc += _svg_text(sl + pw / 2, sh - 6, t["res.x"], 11, "middle")
    sc += (
        f'<text transform="translate(14 {st + ph / 2:.1f}) rotate(-90)" font-size="11" '
        f'text-anchor="middle" fill="var(--muted)" class="sans">{e(t["res.y"])}</text>'
    )
    sc += (
        '<defs><marker id="ar1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--R1)"/></marker>'
        '<marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--R2)"/></marker></defs>'
    )
    b2, r1, r2 = HANDOFF["B2"], HANDOFF["R1"], HANDOFF["R2"]
    sc += (
        f'<path d="M{sx(b2["over"]) + 4:.1f} {sy(b2["recall"]) - 11:.1f} Q {sx(0.06):.1f} {sy(1.0):.1f} '
        f'{sx(r1["over"]) - 14:.1f} {sy(r1["recall"]) + 2:.1f}" stroke="var(--R1)" stroke-width="1.6" '
        'stroke-dasharray="5 4" fill="none" marker-end="url(#ar1)"/>'
    )
    sc += (
        f'<path d="M{sx(b2["over"]) + 9:.1f} {sy(b2["recall"]) - 9:.1f} L {sx(r2["over"]) - 9:.1f} '
        f'{sy(r2["recall"]) + 9:.1f}" stroke="var(--R2)" stroke-width="1.8" fill="none" marker-end="url(#ar2)"/>'
    )
    offsets = {
        "B0": (14, 4, "start"),
        "B2": (-2, 26, "start"),
        "R1": (-16, 30, "end"),
        "R2": (16, 20, "start"),
    }
    for label, row in HANDOFF.items():
        cx, cy = sx(row["over"]), sy(row["recall"])
        sc += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="10" fill="var(--{label})" stroke="var(--bg)" stroke-width="2"/>'
        dx, dy, anchor = offsets[label]
        sc += _svg_text(
            cx + dx, cy + dy, t["names"][label], 12, anchor, "var(--ink)", 600
        )
        sc += _svg_text(cx + dx, cy + dy + 14, f"F1 {row['f1']:.3f}", 11, anchor)
    right_svg = f'<svg width="{sw}" height="{sh}" viewBox="0 0 {sw} {sh}">{sc}</svg>'

    def box(title, note, svg):
        note_html = (
            f'<span style="font-size:11px;color:var(--muted)">{e(note)}</span>'
            if note
            else ""
        )
        return (
            '<div class="card" style="padding:14px 16px;border-radius:16px">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:8px">'
            f'<span class="serif" style="font-size:18px;font-weight:600">{e(title)}</span>{note_html}</div>{svg}</div>'
        )

    return (
        head(t, t["res.sub"])
        + '<div style="display:grid;grid-template-columns:592px 1fr;gap:16px">'
        + box(t["res.left"], t["res.left.note"], left_svg)
        + box(t["res.right"], "", right_svg)
        + "</div>"
        + caption(t["res.cap"])
    )


def fig_dynamics(t: dict) -> str:
    """Figure 4: dev success and transfer rate during GRPO."""
    with open(ROOT / "reports/rl/dev_curves.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    w, h, sl, sb, st = 576, 270, 44, 36, 14
    ph, pw = h - sb - st, w - sl - 70

    def chart(column: str, note: bool) -> str:
        def sx(v):
            return sl + pw * v / 60

        def sy(v):
            return st + ph * (1 - (v - 0.2) / 0.6)

        out = ""
        for v in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            out += f'<line x1="{sl}" x2="{sl + pw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="var(--grid)"/>'
            out += _svg_text(sl - 8, sy(v) + 4, f"{v:.1f}", 10.5, "end")
        for v in (0, 10, 20, 30, 40, 50, 60):
            out += _svg_text(sx(v), st + ph + 16, str(v), 10.5, "middle")
        out += _svg_text(sl + pw / 2, h - 4, t["dyn.x"], 11, "middle")
        for run in ("R2", "R1"):
            pts = [(int(r["Step"]), float(r[column])) for r in rows if r["Run"] == run]
            path = " ".join(
                f"{'M' if i == 0 else 'L'}{sx(s):.1f} {sy(v):.1f}"
                for i, (s, v) in enumerate(pts)
            )
            out += f'<path d="{path}" stroke="var(--{run})" stroke-width="2.6" fill="none" stroke-linejoin="round"/>'
            for s, v in pts:
                out += f'<circle cx="{sx(s):.1f}" cy="{sy(v):.1f}" r="3.6" fill="var(--{run})"/>'
            s, v = pts[-1]
            ends = {r["Run"]: float(r[column]) for r in rows if r["Step"] == "60"}
            shift = 0.0
            if abs(ends["R1"] - ends["R2"]) < 0.05:
                shift = -8.0 if ends[run] >= max(ends.values()) else 8.0
            out += _svg_text(
                sx(s) + 10,
                sy(v) + 4 + shift,
                f"{run} {v:.2f}",
                12,
                "start",
                f"var(--{run})",
                600,
            )
        if note:
            v40 = [
                float(r[column]) for r in rows if r["Run"] == "R1" and r["Step"] == "40"
            ][0]
            out += (
                f'<circle cx="{sx(40):.1f}" cy="{sy(v40):.1f}" r="8" fill="none" stroke="var(--R1)" '
                'stroke-width="1.6" stroke-dasharray="3 2"/>'
            )
        return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">{out}</svg>'

    def box(title, svg, foot=""):
        foot_html = (
            f'<div style="font-size:11.5px;color:var(--muted);margin-top:4px">◌ {e(foot)}</div>'
            if foot
            else ""
        )
        return (
            '<div class="card" style="padding:14px 16px;border-radius:16px">'
            f'<div class="serif" style="font-size:18px;font-weight:600;margin-bottom:6px">{e(title)}</div>{svg}{foot_html}</div>'
        )

    return (
        head(t, t["dyn.sub"])
        + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">'
        + box(t["dyn.left"], chart("success", False))
        + box(t["dyn.right"], chart("transferred", True), t["dyn.note"])
        + "</div>"
        + caption(t["dyn.cap"])
    )


def fig_stats(t: dict) -> str:
    """At-a-glance numbers."""
    colors = ["green", "rose", "blue", "amber", "lilac"]
    cards = ""
    for color, (label, big, unit, foot) in zip(colors, t["stats"], strict=True):
        cards += (
            f'<div class="panel p-{color}" style="padding:16px 18px">'
            f'<div class="mono" style="font-size:10.5px;letter-spacing:.14em;color:var(--ac)">{e(label)}</div>'
            f'<div class="serif" style="font-size:46px;font-weight:700;line-height:1.05;margin:16px 0 6px;white-space:nowrap">'
            f'{e(big)}<span class="mono" style="font-size:17px;font-weight:500;color:var(--ac)">{e(unit)}</span></div>'
            f'<div style="font-size:11.5px;color:var(--muted)">{e(foot)}</div></div>'
        )
    return (
        head(t, t["stats.sub"])
        + f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px">{cards}</div>'
        + f'<div class="mono" style="margin-top:14px;font-size:11.5px;color:var(--muted)">{e(t["stats.foot"])}</div>'
    )


def fig_styles(t: dict) -> str:
    """Figure 5: the six communication styles (illustration + labels)."""
    art = (ASSETS / "styles-art.png").resolve().as_uri()
    colors = ["blue", "rose", "green", "amber", "lilac", "rose"]
    labels = "".join(
        f'<div style="text-align:center;padding:0 6px"><div class="mono" style="font-size:11px;'
        f'letter-spacing:.12em;text-transform:uppercase;color:var(--{c})">{e(name)}</div>'
        f'<div style="font-size:12px;color:var(--muted);margin-top:3px">{e(desc)}</div></div>'
        for c, (name, desc) in zip(colors, t["styles"], strict=True)
    )
    return (
        head(t, t["styles.sub"])
        + '<div class="card" style="padding:12px;border-radius:18px">'
        f'<img src="{art}" style="width:100%;display:block;border-radius:10px">'
        f'<div style="display:grid;grid-template-columns:repeat(6,1fr);margin-top:10px">{labels}</div></div>'
        + caption(t["styles.cap"])
    )


FIGURES = {
    "stats": fig_stats,
    "pipeline": fig_pipeline,
    "results": fig_results,
    "loophole": fig_loophole,
    "dynamics": fig_dynamics,
    "styles": fig_styles,
}


def render(names: list[str]) -> None:
    """Render the named figures in both languages and themes."""
    from playwright.sync_api import sync_playwright

    ASSETS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": WIDTH, "height": 800}, device_scale_factor=2
        )
        for name in names:
            for lang, text in T.items():
                for theme_name, theme in THEMES.items():
                    doc = (
                        f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8">'
                        f'<style>{css(theme)}</style></head><body><div id="fig">'
                        f"{FIGURES[name](text)}</div></body></html>"
                    )
                    source = Path(tmp) / f"{name}-{lang}-{theme_name}.html"
                    source.write_text(doc, encoding="utf-8")
                    page.goto(source.as_uri())
                    page.evaluate("document.fonts.ready")
                    out = ASSETS / f"{name}-{lang}-{theme_name}.png"
                    page.locator("#fig").screenshot(path=str(out))
                    print(out.relative_to(ROOT))
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=list(FIGURES))
    args = parser.parse_args()
    render(args.only)


if __name__ == "__main__":
    main()
