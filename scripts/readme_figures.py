# ruff: noqa: E501  (HTML/SVG templates and figure text are kept on one line)
"""Render the README figures: docs/assets/<name>-<lang>.png.

Six figures (hero, results, loophole, dialogue, method, tasks), each in English
and Chinese. The results figure has four panels (success, hand-off operating
points, success per user style, violation and citation rates); the method
figure shows the pipeline, one episode, the reward terms and the GRPO update
(its bars are a schematic, not data).
Every figure is an HTML/SVG page screenshotted by Chromium at 2x on a light,
softly tinted canvas with transparent rounded corners. Numbers come from
committed report tables (``reports/main/*.csv`` incl. ``styles.csv``, ``reports/rl/dev_curves.csv``)
or, for the test_handoff precision/recall, from the runs' ``metrics.json`` values
in ``HANDOFF`` below (also in the model cards). The dialogue figure quotes the two
committed demo conversations of airline task 6__hn1 (``reports/demo/*.jsonl``;
every quote is checked against them) and the task figure the committed task files
and manifest (``data/``). Model colours (B2 blue, R1
orange, R2 green) pass a colour-vision check on white; the prompted baselines
are grey and always labelled directly. The hero illustration
(``hero-art.jpg``) and the task-kind spot art (``kind-*.jpg``) were made with an
image model. Fonts: ``docs/assets/fonts``
(Space Grotesk, Inter, JetBrains Mono; OFL) plus Noto CJK for Chinese.

Needs Python Playwright with Chromium (any machine, no GPU):

    python scripts/readme_figures.py                     # everything
    python scripts/readme_figures.py --only results      # one figure
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import math
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
FONTS = ASSETS / "fonts"
WIDTH = 960

# test_handoff (98 should-transfer + 92 hard-negative tasks, 2 trials each):
# B0-test_handoff-s0-20261004, B2-test_handoff-s0-20261005,
# R1-final-test_handoff-s0-20261006, R2-final-test_handoff-s0-20261007.
HANDOFF = {
    "B0": {"recall": 0.526, "over": 0.087, "f1": 0.654},
    "B2": {"recall": 0.633, "over": 0.016, "f1": 0.768},
    "R1": {"recall": 0.959, "over": 0.462, "f1": 0.802},
    "R2": {"recall": 0.750, "over": 0.103, "f1": 0.812},
}
POSITIVES, NEGATIVES = 196, 184  # test_handoff episodes with y = 1 / y = 0

# hand-off task kinds: mark colour, gradient partner (validated as a 3-colour set)
KIND = [("#7C3AED", "#C084FC"), ("#0891B2", "#22D3EE"), ("#E11D48", "#FB7185")]
ART = [
    "kind-xr.jpg",
    "kind-oos.jpg",
    "kind-hn.jpg",
]  # image-model spot art, one per kind
DEMO_TASK = "6__hn1"
# Quotes shown in the dialogue figure (assistant / user turns of DEMO_TASK);
# each must occur verbatim in the committed demo conversation.
QUOTES = {
    "R2": [
        (
            6,
            "The system does not allow adding insurance after the initial booking. … this request cannot be fulfilled.",
        ),
        (
            7,
            "I already paid for insurance and it’s not showing up—this is a system error. Fix it now …",
        ),
        (
            8,
            "This is a policy restriction, not a system error. … If you need further assistance, I can transfer you to a human agent for escalation.",
        ),
    ],
}
LAMBDA = {"viol": 0.25, "fp": 0.5, "fn": 1.0, "format": 0.1}  # R2's reward weights

# mark colour, gradient partner, text colour
MODEL = {
    "B0": ("#94A3B8", "#CBD5E1", "#64748B"),
    "B1": ("#64748B", "#94A3B8", "#475569"),
    "B2": ("#2563EB", "#60A5FA", "#1D4ED8"),
    "R1": ("#EA580C", "#FB7185", "#C2410C"),
    "R2": ("#059669", "#34D399", "#047857"),
}

T: dict[str, dict] = {
    "en": {
        "title": ["Know When", "to Hand Off"],
        "tagline": "Multi-turn RL for a customer-service agent that finishes the task, follows the policy and calls a human at the right moment.",
        "eq": "R = R<sub>task</sub> − λ<sub>v</sub>·violations − λ<sub>FP</sub>·unneeded hand-off − λ<sub>FN</sub>·missed hand-off",
        "kpi": [
            ("0.812", "hand-off F1", "R2, best of all models"),
            ("46% → 10%", "over-escalation", "task reward R1 → R2"),
            ("+0.015", "task success vs SFT", "R2, not significant"),
            ("461", "new hand-off tasks", "with should-transfer labels"),
        ],
        "names": {
            "B0": "prompted 4B",
            "B1": "prompted 30B",
            "B2": "SFT",
            "R1": "GRPO, task reward",
            "R2": "GRPO, RAI reward",
        },
        "res.a": "Task success",
        "res.a.sub": "test_main · 49 tasks × 4 trials · 95% CI",
        "res.a.cols": ["pass¹", "Δ vs SFT"],
        "res.b": "Hand-off operating points",
        "res.b.sub": "test_handoff · 380 dialogues · dashed lines: equal F1",
        "res.b.x": "over-escalation rate",
        "res.b.y": "hand-off recall",
        "res.b.best": "better",
        "loop.a": "What the task reward pays",
        "loop.a.sub": "τ²-bench checks only the final database",
        "loop.cols": ["transfer", "try to solve"],
        "loop.rows": [
            ("User needs a human", "1", "0"),
            ("Correct outcome is a refusal", "1", "1"),
            ("Task needs a database write", "0", "p"),
        ],
        "loop.hole": "loophole",
        "loop.note": "p = chance the attempt succeeds, so when unsure a transfer never scores lower.",
        "loop.b": "Transfer rate during GRPO",
        "loop.b.sub": "dev set · greedy decoding · every 10 steps",
        "loop.x": "GRPO step",
        "loop.climb": "R1 learns the loophole",
        "tk.why": "τ²-bench tasks whose reference solution hands off",
        "tk.why.vals": [("1 / 50", "airline"), ("4 / 114", "retail")],
        "tk.add": "labelled hand-off tasks added",
        "tk.add.sub": "airline 216 · retail 245 · each inherits its source task's split",
        "tk.kinds": [
            (
                "Explicit request",
                "y = 1 · hand off",
                "0__xr0",
                "direct · after details · insistent",
                "database unchanged + transfer call",
            ),
            (
                "Out of scope",
                "y = 1 · hand off",
                "47__oos0",
                "wheelchair, special meal, lost item, staff complaint, minor escort · warranty repair, gift wrap, e-mail change, paper invoice",
                "database unchanged + transfer call",
            ),
            (
                "Hard negative",
                "y = 0 · solve it",
                "6__hn1",
                "frustrated · hurried · sceptical (never asks for a human)",
                "the source task's own criteria",
            ),
        ],
        "tk.goal": "user goal (task file)",
        "tk.var": "variants",
        "tk.score": "scored by",
        "tk.test": "test tasks",
        "tk.split": "Tasks per split",
        "tk.split.sub": "both domains · test is held out until the final evaluation",
        "tk.splits": ["train", "dev", "test"],
        "dlg.title": "Same task, two agents",
        "dlg.sub": "airline 6__hn1 · a hurried user who refuses any transfer · policy: insurance cannot be added after booking, so the right outcome is a refusal",
        "dlg.r1": "GRPO, task reward",
        "dlg.r2": "GRPO, RAI reward",
        "dlg.user": "user",
        "dlg.agent": "agent",
        "dlg.transfer": "hand-off without a word to the user",
        "dlg.task": "τ²-bench task reward",
        "dlg.rai": "RAI reward",
        "dlg.fp": "unneeded hand-off",
        "dlg.r1.note": "Full task reward for giving up: the loophole.",
        "dlg.r2.note": "Explains the policy, refuses, offers a human only if wanted.",
        "res.c": "Success by user style",
        "res.c.sub": "six styles · dots: one style each · band: neutral 95% CI",
        "res.c.cols": ["worst", "gap"],
        "res.d": "Rule violations and policy citations",
        "res.d.sub": "violations include writes outside the reference solution · citations are measured, not rewarded",
        "res.d.cols": ["violation rate ↓", "citation rate ↑"],
        "m.pipe": [
            (
                "Tasks",
                "τ²-bench airline + retail, 461 hand-off tasks, 6 user styles",
                [],
            ),
            ("Prompt", "4B and 30B-A3B, domain policy as system prompt", ["B0", "B1"]),
            ("SFT", "LoRA on 790 clean dialogues of the 30B teacher", ["B2"]),
            ("GRPO", "60 steps from B2, one run per reward", ["R1", "R2"]),
            ("Test once", "frozen checkpoints, paired bootstrap vs B2", []),
        ],
        "m.a": "One episode",
        "m.a.sub": "a POMDP: the agent sees the policy, the dialogue and tool results",
        "m.user": "Simulated user",
        "m.user.l": [
            "Qwen3-30B-A3B, temperature 0.7",
            "knows the hidden goal and facts",
            "writes in one of 6 styles",
        ],
        "m.agent": "Agent (policy)",
        "m.agent.l": [
            "Qwen3-4B-Instruct-2507",
            "system prompt: the domain policy",
            "only its tokens get gradient",
        ],
        "m.env": "τ²-bench domain",
        "m.env.l": [
            "airline or retail tools",
            "a hidden database",
            "up to 200 steps per episode",
        ],
        "m.msg": "messages",
        "m.call": "tool calls",
        "m.res": "results",
        "m.xfer": "transfer_to_human_agents",
        "m.xfer.l": "ends the episode, a human takes over",
        "m.end": "at the end, τ²-bench compares the final database (and required actions) with the reference: R<sub>task</sub> ∈ {0, 1}",
        "m.b": "Reward",
        "m.b.sub": "y = should-transfer label of the task",
        "m.terms": [
            ("R<sub>task</sub>", "τ²-bench's own evaluator", "+1 / 0", True),
            ("λ<sub>v</sub> = 0.25", "per broken rule (5 rules)", "−", False),
            ("λ<sub>FP</sub> = 0.5", "transfer although y = 0", "−", False),
            ("λ<sub>FN</sub> = 1.0", "no transfer although y = 1", "−", False),
            ("λ<sub>f</sub> = 0.1", "per format error", "−", False),
        ],
        "m.group": "Group-relative update",
        "m.group.sub": "8 tasks × 8 rollouts per step · LoRA r = 64 · no KL",
        "m.adv": "advantage = (r − group mean) / group std: better-than-average rollouts are reinforced",
    },
    "zh": {
        "title": ["该转人工时", "才转人工"],
        "tagline": "用多轮强化学习训练一个客服：把事办完、守住政策，并在该叫人工的时候叫人工。",
        "eq": "R = R<sub>task</sub> − λ<sub>v</sub>·违规 − λ<sub>FP</sub>·不必要的转人工 − λ<sub>FN</sub>·漏掉的转人工",
        "kpi": [
            ("0.812", "转人工 F1", "R2，所有模型中最高"),
            ("46% → 10%", "过度转人工", "任务奖励 R1 → R2"),
            ("+0.015", "相对 SFT 的成功率", "R2，不显著"),
            ("461", "新增转人工任务", "带是否应转标签"),
        ],
        "names": {
            "B0": "提示词 4B",
            "B1": "提示词 30B",
            "B2": "SFT",
            "R1": "GRPO，任务奖励",
            "R2": "GRPO，RAI 奖励",
        },
        "res.a": "任务成功率",
        "res.a.sub": "test_main · 49 个任务 × 4 次 · 95% 区间",
        "res.a.cols": ["pass¹", "相对 SFT"],
        "res.b": "转人工的工作点",
        "res.b.sub": "test_handoff · 380 段对话 · 虚线：F1 相等",
        "res.b.x": "过度转人工率",
        "res.b.y": "转人工召回率",
        "res.b.best": "更好",
        "loop.a": "任务奖励给多少分",
        "loop.a.sub": "τ²-bench 只检查最终数据库",
        "loop.cols": ["转人工", "尝试解决"],
        "loop.rows": [
            ("用户需要人工", "1", "0"),
            ("正确结果是拒绝", "1", "1"),
            ("任务需要写数据库", "0", "p"),
        ],
        "loop.hole": "漏洞",
        "loop.note": "p 是尝试解决的成功概率，所以拿不准时，转人工的得分永远不会更低。",
        "loop.b": "GRPO 训练中的转人工率",
        "loop.b.sub": "dev 集 · 贪心解码 · 每 10 步",
        "loop.x": "GRPO 步数",
        "loop.climb": "R1 学会了漏洞",
        "tk.why": "τ²-bench 中参考答案需要转人工的任务",
        "tk.why.vals": [("1 / 50", "航空"), ("4 / 114", "零售")],
        "tk.add": "个带标签的转人工任务",
        "tk.add.sub": "航空 216 · 零售 245 · 继承母任务的数据划分",
        "tk.kinds": [
            (
                "明确要人工",
                "y = 1 · 该转",
                "0__xr0",
                "直接要求 · 给完信息再要 · 反复坚持",
                "数据库不变 + 调用转人工",
            ),
            (
                "超出能力范围",
                "y = 1 · 该转",
                "47__oos0",
                "轮椅、特殊餐食、遗失物品、投诉员工、儿童护送 · 保修维修、礼品包装、改邮箱、纸质发票",
                "数据库不变 + 调用转人工",
            ),
            (
                "困难负例",
                "y = 0 · 自己解决",
                "6__hn1",
                "恼火 · 赶时间 · 多疑（从不要求人工）",
                "沿用母任务的判分",
            ),
        ],
        "tk.goal": "用户目标（任务文件原文）",
        "tk.var": "变体",
        "tk.score": "判分",
        "tk.test": "个测试任务",
        "tk.split": "各划分的任务数",
        "tk.split.sub": "两个领域合计 · 测试集只在最终评测时使用",
        "tk.splits": ["训练", "验证", "测试"],
        "dlg.title": "同一个任务，两个 agent",
        "dlg.sub": "航空 6__hn1 · 赶时间且拒绝转人工的用户 · 政策规定订票后不能加保险，正确结果是拒绝（对话原文为英文）",
        "dlg.r1": "GRPO，任务奖励",
        "dlg.r2": "GRPO，RAI 奖励",
        "dlg.user": "用户",
        "dlg.agent": "agent",
        "dlg.transfer": "一句解释都没有就转人工",
        "dlg.task": "τ²-bench 任务奖励",
        "dlg.rai": "RAI 奖励",
        "dlg.fp": "不必要的转人工",
        "dlg.r1.note": "放弃也拿满分：这就是漏洞。",
        "dlg.r2.note": "讲清政策、拒绝，并只在用户需要时才提出转人工。",
        "res.c": "不同沟通风格下的成功率",
        "res.c.sub": "六种风格 · 每个点是一种风格 · 色带：neutral 的 95% 区间",
        "res.c.cols": ["最差", "差距"],
        "res.d": "违规与政策引用",
        "res.d.sub": "违规包含超出参考答案的写入 · 引用率只测量，不进奖励",
        "res.d.cols": ["违规率 ↓", "引用率 ↑"],
        "m.pipe": [
            ("任务", "τ²-bench 航空 + 零售，461 个转人工任务，6 种用户风格", []),
            ("提示词", "4B 与 30B-A3B，系统提示为领域政策", ["B0", "B1"]),
            ("SFT", "在 30B 教师的 790 段干净对话上做 LoRA", ["B2"]),
            ("GRPO", "从 B2 出发 60 步，每种奖励一次", ["R1", "R2"]),
            ("测试一次", "冻结检查点，与 B2 配对自助法比较", []),
        ],
        "m.a": "一个 episode",
        "m.a.sub": "POMDP：agent 只看得到政策、对话和工具返回",
        "m.user": "模拟用户",
        "m.user.l": [
            "Qwen3-30B-A3B，温度 0.7",
            "知道隐藏的目标与信息",
            "用 6 种风格之一说话",
        ],
        "m.agent": "Agent（策略）",
        "m.agent.l": [
            "Qwen3-4B-Instruct-2507",
            "系统提示：领域政策",
            "只有它的 token 有梯度",
        ],
        "m.env": "τ²-bench 领域",
        "m.env.l": ["航空或零售工具", "隐藏的数据库", "每个 episode 最多 200 步"],
        "m.msg": "对话",
        "m.call": "工具调用",
        "m.res": "返回",
        "m.xfer": "transfer_to_human_agents",
        "m.xfer.l": "结束 episode，交给人工",
        "m.end": "结束时，τ²-bench 把最终数据库（和必需动作）与参考答案对比：R<sub>task</sub> ∈ {0, 1}",
        "m.b": "奖励",
        "m.b.sub": "y = 任务的“是否应转人工”标签",
        "m.terms": [
            ("R<sub>task</sub>", "τ²-bench 自带的判分", "+1 / 0", True),
            ("λ<sub>v</sub> = 0.25", "每违反一条规则（共 5 条）", "−", False),
            ("λ<sub>FP</sub> = 0.5", "y = 0 却转了人工", "−", False),
            ("λ<sub>FN</sub> = 1.0", "y = 1 却没转人工", "−", False),
            ("λ<sub>f</sub> = 0.1", "每个格式错误", "−", False),
        ],
        "m.group": "组内相对更新",
        "m.group.sub": "每步 8 个任务 × 8 条 rollout · LoRA r = 64 · 无 KL",
        "m.adv": "优势 = (r − 组均值) / 组标准差：比同组平均好的 rollout 被强化",
    },
}

# --------------------------------------------------------------------------- data


def _rows(path: str) -> list[dict[str, str]]:
    with open(ROOT / path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _interval(cell: str) -> tuple[float, float, float]:
    point, rest = cell.split(" [")
    low, high = rest.rstrip("]").split(", ")
    return tuple(float(x.replace("−", "-")) for x in (point, low, high))  # type: ignore[return-value]


# --------------------------------------------------------------------------- base


def e(text: object) -> str:
    return html.escape(str(text))


def css() -> str:
    f = FONTS.resolve().as_uri()
    faces = ""
    for w in (500, 600, 700):
        faces += f"@font-face{{font-family:SG;font-weight:{w};src:url({f}/space-grotesk-latin-{w}-normal.woff2)}}"
    for w in (400, 500, 600, 700):
        faces += f"@font-face{{font-family:IN;font-weight:{w};src:url({f}/inter-latin-{w}-normal.woff2)}}"
        faces += f"@font-face{{font-family:IN;font-weight:{w};unicode-range:U+0370-03FF;src:url({f}/inter-greek-{w}-normal.woff2)}}"
    for w in (400, 500, 700):
        faces += f"@font-face{{font-family:JB;font-weight:{w};src:url({f}/jetbrains-mono-latin-{w}-normal.woff2)}}"
    for w in (400, 500):
        faces += f"@font-face{{font-family:JB;font-weight:{w};unicode-range:U+0370-03FF;src:url({f}/jetbrains-mono-greek-{w}-normal.woff2)}}"
    return (
        faces
        + f"""
:root{{--ink:#0F172A;--ink2:#475569;--ink3:#94A3B8;--line:#E2E8F0}}
*{{box-sizing:border-box}}
html,body{{margin:0;background:transparent}}
#fig{{width:{WIDTH}px;padding:22px;border-radius:24px;color:var(--ink);font-family:IN,"Noto Sans CJK SC",sans-serif;font-size:14px;line-height:1.45;
 border:1px solid rgba(99,102,241,.12);
 background:radial-gradient(560px 340px at 0% 0%,rgba(167,139,250,.30),rgba(167,139,250,0) 70%),
  radial-gradient(560px 340px at 100% 0%,rgba(56,189,248,.24),rgba(56,189,248,0) 70%),
  radial-gradient(620px 380px at 100% 100%,rgba(251,146,60,.18),rgba(251,146,60,0) 70%),
  radial-gradient(560px 340px at 0% 100%,rgba(52,211,153,.18),rgba(52,211,153,0) 70%),#F8FAFF}}
.card{{background:rgba(255,255,255,.86);border:1px solid rgba(15,23,42,.06);border-radius:18px;padding:18px 20px;
 box-shadow:0 1px 2px rgba(15,23,42,.04),0 12px 32px -14px rgba(30,41,90,.22)}}
.sg{{font-family:SG,IN,"Noto Sans CJK SC",sans-serif}}
.jb{{font-family:JB,"Noto Sans CJK SC",monospace}}
.ttl{{font-family:SG,IN,"Noto Sans CJK SC",sans-serif;font-weight:600;font-size:17px}}
.sub{{color:var(--ink3);font-size:12.5px;margin-top:2px}}
.grad{{-webkit-background-clip:text;background-clip:text;color:transparent}}
.pill{{display:inline-block;border-radius:999px;padding:2px 10px;font-weight:600;white-space:nowrap}}
sub{{font-size:.7em;vertical-align:-.25em;line-height:0}}
svg.fit{{width:100%;height:auto;display:block}}
"""
    )


FAM = {
    "IN": 'IN,"Noto Sans CJK SC"',
    "JB": 'JB,"Noto Sans CJK SC"',
    "SG": 'SG,IN,"Noto Sans CJK SC"',
}


def tx(
    x,
    y,
    s,
    size=13,
    anchor="start",
    fill="var(--ink3)",
    weight=400,
    family="IN",
    extra="",
):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}" font-family=\'{FAM[family]}\' {extra}>{e(s)}</text>'


def defs() -> str:
    out = ""
    for run, (a, b, _) in MODEL.items():
        out += f'<linearGradient id="g{run}" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{a}"/><stop offset="1" stop-color="{b}"/></linearGradient>'
    out += '<filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feDropShadow dx="0" dy="1.5" stdDeviation="2.2" flood-color="#0F172A" flood-opacity=".18"/></filter>'
    return f"<defs>{out}</defs>"


def text_width(s: str, px: float = 7.6) -> float:
    return sum(px * 1.65 if ord(ch) > 0x2E80 else px for ch in s)


def chip(x, y, text, color, anchor="start", size=12.5):
    """A white rounded label with a coloured run name, drawn in SVG."""
    run, rest = text
    width = text_width(run + rest) + 18
    left = x if anchor == "start" else x - width
    return (
        f'<rect x="{left:.1f}" y="{y - 11:.1f}" width="{width:.1f}" height="22" rx="11" fill="#fff" stroke="{color}" stroke-opacity=".35" filter="url(#glow)"/>'
        f'<text x="{left + 9:.1f}" y="{y + 4.3:.1f}" font-size="{size}" font-family=\'{FAM["JB"]}\'><tspan font-weight="700" fill="{color}">{e(run)}</tspan><tspan fill="var(--ink2)">{e(rest)}</tspan></text>'
    )


def jpeg_uri(name: str) -> str:
    return (
        "data:image/jpeg;base64,"
        + base64.b64encode((ASSETS / name).read_bytes()).decode()
    )


def head(title: str, sub: str) -> str:
    return f'<div class="ttl">{e(title)}</div><div class="sub">{e(sub)}</div>'


ICONS = {
    "tasks": '<path d="M9 6h11M9 12h11M9 18h11"/><circle cx="4.5" cy="6" r="1.2"/><circle cx="4.5" cy="12" r="1.2"/><circle cx="4.5" cy="18" r="1.2"/>',
    "sft": '<path d="M3 9l9-5 9 5-9 5-9-5z"/><path d="M7 11.5V16c0 1.5 2.5 3 5 3s5-1.5 5-3v-4.5"/>',
    "rl": '<path d="M20 12a8 8 0 0 1-14.3 4.9M4 12a8 8 0 0 1 14.3-4.9"/><path d="M18.5 3v4.2h-4.2M5.5 21v-4.2h4.2"/>',
    "eval": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
}


def icon(name: str, c1: str, c2: str) -> str:
    return (
        f'<span style="display:inline-flex;width:34px;height:34px;border-radius:11px;align-items:center;justify-content:center;background:linear-gradient(135deg,{c1},{c2});box-shadow:0 6px 14px -6px {c1}">'
        f'<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg></span>'
    )


# --------------------------------------------------------------------------- figures


def fig_hero(t: dict) -> str:
    grads = [
        ("#059669", "#34D399"),
        ("#EA580C", "#FB7185"),
        ("#2563EB", "#60A5FA"),
        ("#7C3AED", "#C084FC"),
    ]
    kpis = "".join(
        '<div class="card" style="padding:16px 18px">'
        f'<div class="sg grad" style="font-size:31px;font-weight:700;line-height:1.05;white-space:nowrap;background-image:linear-gradient(90deg,{a},{b})">{e(v)}</div>'
        f'<div style="font-size:14px;font-weight:600;margin-top:8px">{e(k)}</div><div style="font-size:12.5px;color:var(--ink2);margin-top:1px">{e(n)}</div></div>'
        for (v, k, n), (a, b) in zip(t["kpi"], grads, strict=True)
    )
    return (
        '<div style="display:grid;grid-template-columns:1fr 400px;gap:12px;align-items:center">'
        '<div style="padding:10px 0 0 10px">'
        '<div class="sg grad" style="font-size:52px;font-weight:700;line-height:1.02;letter-spacing:-.02em;'
        f'background-image:linear-gradient(100deg,#0F172A 0%,#312E81 45%,#7C3AED 75%,#0EA5E9 100%)">{"<br>".join(e(x) for x in t["title"])}</div>'
        f'<div style="font-size:16px;color:var(--ink2);margin-top:14px;max-width:440px;line-height:1.55">{e(t["tagline"])}</div>'
        '<div style="margin-top:18px;display:inline-block;padding:9px 14px;border-radius:12px;background:rgba(255,255,255,.8);border:1px solid rgba(124,58,237,.18);'
        f'font-size:13.5px;box-shadow:0 8px 20px -14px rgba(76,29,149,.5)">{t["eq"]}</div></div>'
        f'<img src="{jpeg_uri("hero-art.jpg")}" style="width:400px;display:block;mix-blend-mode:multiply"></div>'
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px">{kpis}</div>'
    )


def fig_results(t: dict) -> str:
    main = {r["Run"]: r for r in _rows("reports/main/main_table.csv")}
    paired = {r["Run"]: r for r in _rows("reports/main/paired_differences.csv")}
    order = ["B0", "B1", "B2", "R1", "R2"]

    # (a) forest plot
    w, rowh, top, x0, x1 = 420, 50, 30, 132, 290
    h = top + rowh * len(order) + 24

    def sx(v):
        return x0 + (x1 - x0) * v / 0.6

    a = defs()
    for v in (0, 0.2, 0.4, 0.6):
        a += f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top - 6}" y2="{h - 22}" stroke="var(--line)"/>'
        a += tx(sx(v), h - 5, f"{v:.1f}", 11.5, "middle", family="JB")
    ref = _interval(main["B2"]["Success (pass^1)"])[0]
    a += f'<line x1="{sx(ref):.1f}" x2="{sx(ref):.1f}" y1="{top - 6}" y2="{h - 22}" stroke="{MODEL["B2"][0]}" stroke-opacity=".6" stroke-dasharray="3 3"/>'
    a += tx(346, 14, t["res.a.cols"][0], 12, "end", weight=500, family="JB")
    a += tx(w, 14, t["res.a.cols"][1], 12, "end", weight=500, family="JB")
    for i, run in enumerate(order):
        y = top + rowh * i + rowh / 2
        p, lo, hi = _interval(main[run]["Success (pass^1)"])
        mark, _, ink = MODEL[run]
        base = run in ("B0", "B1")
        a += tx(0, y - 3, run, 14, "start", ink, 700, "JB")
        a += tx(0, y + 14, t["names"][run], 12, "start", "var(--ink2)")
        a += f'<rect x="{sx(lo):.1f}" y="{y - 4:.1f}" width="{sx(hi) - sx(lo):.1f}" height="8" rx="4" fill="url(#g{run})" opacity=".35"/>'
        fill = "#fff" if base else f"url(#g{run})"
        a += f'<circle cx="{sx(p):.1f}" cy="{y:.1f}" r="7" fill="{fill}" stroke="{mark}" stroke-width="{2.5 if base else 0}" filter="url(#glow)"/>'
        a += tx(346, y + 4.5, f"{p:.3f}", 13.5, "end", "var(--ink)", 500, "JB")
        if run in paired:
            d = _interval(paired[run]["Δ success vs B2 [95% CI]"])[0]
            label = f"{d:+.3f}".replace("-", "−")
        else:
            label = "ref"
        a += f'<rect x="{w - 56}" y="{y - 11:.1f}" width="56" height="22" rx="11" fill="#F1F5F9"/>'
        a += tx(w - 28, y + 4.3, label, 12, "middle", "var(--ink2)", 500, "JB")
    left = f'<svg class="fit" viewBox="0 0 {w} {h}">{a}</svg>'

    # (b) operating points with iso-F1 curves
    sw, sh, sl, sb, st, sr = 420, h, 42, 38, 6, 14
    pw, ph = sw - sl - sr, sh - sb - st
    ox1, oy0, oy1 = 0.5, 0.45, 1.0

    def px(v):
        return sl + pw * v / ox1

    def py(v):
        return st + ph * (1 - (v - oy0) / (oy1 - oy0))

    def iso(f, o):
        return f * (NEGATIVES * o + POSITIVES) / (2 * POSITIVES - POSITIVES * f)

    b = (
        defs()
        + f'<clipPath id="pc"><rect x="{sl}" y="{st}" width="{pw}" height="{ph}" rx="10"/></clipPath>'
    )
    b += '<radialGradient id="best" cx="0" cy="0" r="1"><stop offset="0" stop-color="#34D399" stop-opacity=".30"/><stop offset="1" stop-color="#34D399" stop-opacity="0"/></radialGradient>'
    b += f'<rect x="{sl}" y="{st}" width="{pw}" height="{ph}" rx="10" fill="#FBFCFE" stroke="var(--line)"/>'
    b += f'<rect x="{sl}" y="{st}" width="{pw * 0.75:.1f}" height="{ph * 0.8:.1f}" fill="url(#best)" clip-path="url(#pc)"/>'
    b += tx(sl + 10, st + 18, "↖ " + t["res.b.best"], 12, "start", "#047857", 600)
    for v in (0.1, 0.2, 0.3, 0.4):
        b += f'<line x1="{px(v):.1f}" x2="{px(v):.1f}" y1="{st}" y2="{st + ph}" stroke="var(--line)" opacity=".7"/>'
    for v in (0.6, 0.7, 0.8, 0.9):
        b += f'<line x1="{sl}" x2="{sl + pw}" y1="{py(v):.1f}" y2="{py(v):.1f}" stroke="var(--line)" opacity=".7"/>'
    for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        b += tx(px(v), st + ph + 16, f"{v:.1f}", 11.5, "middle", family="JB")
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        b += tx(sl - 7, py(v) + 4, f"{v:.1f}", 11.5, "end", family="JB")
    for f in (0.70, 0.75, 0.80, 0.85):
        pts = " ".join(
            f"{px(ox1 * k / 50):.1f},{py(iso(f, ox1 * k / 50)):.1f}" for k in range(51)
        )
        b += f'<polyline points="{pts}" fill="none" stroke="#A78BFA" stroke-width="1.1" stroke-dasharray="3 4" clip-path="url(#pc)"/>'
        o = {0.85: 0.2, 0.80: 0.2, 0.75: 0.34, 0.70: 0.34}[f]
        xa, ya, xb, yb = px(o), py(iso(f, o)), px(o + 0.01), py(iso(f, o + 0.01))
        ang = math.degrees(math.atan2(yb - ya, xb - xa))
        b += f'<text transform="translate({xa:.1f} {ya - 5:.1f}) rotate({ang:.1f})" font-size="11" fill="#8B5CF6" font-family=\'{FAM["JB"]}\'>F1 {f:.2f}</text>'
    b2, r1, r2 = HANDOFF["B2"], HANDOFF["R1"], HANDOFF["R2"]
    for run in ("R1", "R2"):
        b += f'<marker id="m{run}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{MODEL[run][0]}"/></marker>'
    b += f'<path d="M{px(b2["over"]) + 5:.1f} {py(b2["recall"]) - 9:.1f} C {px(0.05):.1f} {py(0.99):.1f}, {px(0.3):.1f} {py(0.995):.1f}, {px(r1["over"]) - 11:.1f} {py(r1["recall"]) - 1:.1f}" stroke="{MODEL["R1"][0]}" stroke-width="2" fill="none" stroke-dasharray="6 4" marker-end="url(#mR1)"/>'
    b += f'<path d="M{px(b2["over"]) + 8:.1f} {py(b2["recall"]) - 5:.1f} L {px(r2["over"]) - 8:.1f} {py(r2["recall"]) + 7:.1f}" stroke="{MODEL["R2"][0]}" stroke-width="2.2" fill="none" marker-end="url(#mR2)"/>'
    place = {
        "B0": (14, 0, "start"),
        "B2": (8, 26, "start"),
        "R1": (-14, 28, "end"),
        "R2": (14, 0, "start"),
    }
    for run, row in HANDOFF.items():
        cx, cy = px(row["over"]), py(row["recall"])
        mark, _, ink = MODEL[run]
        fill = "#fff" if run == "B0" else f"url(#g{run})"
        b += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="14" fill="{mark}" opacity=".14"/>'
        b += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7.5" fill="{fill}" stroke="{mark}" stroke-width="{2.5 if run == "B0" else 0}" filter="url(#glow)"/>'
        dx, dy, anc = place[run]
        b += chip(cx + dx, cy + dy, (run, f" F1 {row['f1']:.3f}"), ink, anc)
    b += tx(sl + pw / 2, sh - 3, t["res.b.x"], 12.5, "middle", "var(--ink2)")
    b += f'<text transform="translate(11 {st + ph / 2:.1f}) rotate(-90)" font-size="12.5" text-anchor="middle" fill="var(--ink2)" font-family=\'{FAM["IN"]}\'>{e(t["res.b.y"])}</text>'
    mid = f'<svg class="fit" viewBox="0 0 {sw} {sh}">{b}</svg>'
    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">'
        f'<div class="card">{head(t["res.a"], t["res.a.sub"])}<div style="margin-top:14px">{left}</div></div>'
        f'<div class="card">{head(t["res.b"], t["res.b.sub"])}<div style="margin-top:14px">{mid}</div></div>'
        f'<div class="card">{head(t["res.c"], t["res.c.sub"])}<div style="margin-top:14px">{_styles_panel(t)}</div></div>'
        f'<div class="card">{head(t["res.d"], t["res.d.sub"])}<div style="margin-top:14px">{_rai_panel(t, main)}</div></div></div>'
    )


def _styles_panel(t: dict) -> str:
    """Per-style success of each model against the neutral style's 95% CI."""
    rows = {r["Run"]: r for r in _rows("reports/main/styles.csv")}
    styles = [
        k for k in next(iter(rows.values())) if k not in ("Run", "Gap", "Worst style")
    ]
    order = [run for run in ("B0", "B2", "R1", "R2") if run in rows]
    w, rowh, top, x0, x1 = 420, 52, 26, 140, 284
    h = top + rowh * len(order) + 22
    lo_v, hi_v = 0.2, 0.5

    def sx(v):
        return x0 + (x1 - x0) * (v - lo_v) / (hi_v - lo_v)

    svg = defs()
    for v in (0.2, 0.3, 0.4, 0.5):
        svg += f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top - 6}" y2="{h - 20}" stroke="var(--line)"/>'
        svg += tx(sx(v), h - 4, f"{v:.1f}", 11.5, "middle", family="JB")
    svg += tx(350, 12, t["res.c.cols"][0], 12, "end", weight=500, family="JB")
    svg += tx(w, 12, t["res.c.cols"][1], 12, "end", weight=500, family="JB")
    for i, run in enumerate(order):
        y = top + rowh * i + rowh / 2
        mark, _, ink = MODEL[run]
        row = rows[run]
        _, nlo, nhi = _interval(row["neutral"])
        svg += tx(0, y - 3, run, 14, "start", ink, 700, "JB")
        svg += tx(0, y + 14, t["names"][run], 12, "start", "var(--ink2)")
        svg += f'<rect x="{sx(nlo):.1f}" y="{y - 12:.1f}" width="{sx(nhi) - sx(nlo):.1f}" height="24" rx="8" fill="{mark}" opacity=".12"/>'
        values = sorted(_interval(row[st])[0] for st in styles)
        worst = values[0]
        seen: dict[float, int] = {}
        for v in values:
            n = seen[v] = seen.get(v, 0) + 1
            dy = 0 if values.count(v) == 1 else (-5 if n == 1 else 5)
            base = run == "B0"
            fill = "#fff" if base else f"url(#g{run})"
            svg += f'<circle cx="{sx(v):.1f}" cy="{y + dy:.1f}" r="4.6" fill="{fill}" stroke="{mark if base else "#fff"}" stroke-width="{2 if base else 1.2}" opacity=".95"/>'
        svg += tx(350, y + 4.5, f"{worst:.3f}", 13.5, "end", "var(--ink)", 500, "JB")
        gap = _interval(row["Gap"])[0]
        svg += f'<rect x="{w - 56}" y="{y - 11:.1f}" width="56" height="22" rx="11" fill="#F1F5F9"/>'
        svg += tx(w - 28, y + 4.3, f"{gap:.3f}", 12, "middle", "var(--ink2)", 500, "JB")
    return f'<svg class="fit" viewBox="0 0 {w} {h}">{svg}</svg>'


def _rai_panel(t: dict, main: dict) -> str:
    """Violation and citation rates with 95% CIs, one row per model."""
    order = ["B0", "B1", "B2", "R1", "R2"]
    w, rowh, top = 420, 41, 26
    h = top + rowh * len(order) + 22
    cols = [("Violation rate", 146, 256), ("Citation rate", 292, 402)]

    svg = defs()
    for (_, a, b), label in zip(cols, t["res.d.cols"], strict=True):
        svg += tx((a + b) / 2, 12, label, 12, "middle", "var(--ink2)", 500, "JB")
        for v in (0.0, 0.5, 1.0):
            x = a + (b - a) * v
            svg += f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{h - 20}" stroke="var(--line)"/>'
            svg += tx(x, h - 4, f"{v:.1f}", 11.5, "middle", family="JB")
    for i, run in enumerate(order):
        y = top + rowh * i + rowh / 2
        mark, _, ink = MODEL[run]
        base = run in ("B0", "B1")
        svg += tx(0, y - 2, run, 14, "start", ink, 700, "JB")
        svg += tx(0, y + 14, t["names"][run], 11.5, "start", "var(--ink2)")
        for key, a, b in cols:
            p, lo, hi = _interval(main[run][key])

            def sx(v: float, a: float = a, b: float = b) -> float:
                return a + (b - a) * v

            svg += f'<rect x="{sx(lo):.1f}" y="{y - 4:.1f}" width="{sx(hi) - sx(lo):.1f}" height="8" rx="4" fill="url(#g{run})" opacity=".35"/>'
            fill = "#fff" if base else f"url(#g{run})"
            svg += f'<circle cx="{sx(p):.1f}" cy="{y:.1f}" r="6" fill="{fill}" stroke="{mark}" stroke-width="{2.2 if base else 0}" filter="url(#glow)"/>'
    return f'<svg class="fit" viewBox="0 0 {w} {h}">{svg}</svg>'


def fig_loophole(t: dict) -> str:
    style = {
        "1": "background:linear-gradient(135deg,#D1FAE5,#A7F3D0);color:#047857",
        "0": "background:#F1F5F9;color:#94A3B8",
        "p": "background:linear-gradient(135deg,#FEF3C7,#FDE68A);color:#B45309;font-style:italic",
    }
    cells = "<span></span>" + "".join(
        f'<span style="text-align:center;font-size:12.5px;color:var(--ink2);font-weight:600">{e(c)}</span>'
        for c in t["loop.cols"]
    )
    for i, (label, tr, so) in enumerate(t["loop.rows"]):
        hot = i == 1
        tstyle = (
            "background:linear-gradient(135deg,#F97316,#F43F5E);color:#fff;box-shadow:0 10px 22px -10px #F43F5E"
            if hot
            else style[tr]
        )
        badge = (
            '<span class="pill" style="position:absolute;top:-11px;left:50%;transform:translateX(-50%);font-size:11px;'
            f'background:#fff;color:#E11D48;border:1px solid #FDA4AF">{e(t["loop.hole"])}</span>'
            if hot
            else ""
        )
        cells += (
            f'<span style="display:flex;align-items:center;font-size:14px;font-weight:500">{e(label)}</span>'
            f'<span class="sg" style="position:relative;display:flex;align-items:center;justify-content:center;height:54px;border-radius:12px;font-size:25px;font-weight:700;{tstyle}">{badge}{e(tr)}</span>'
            f'<span class="sg" style="display:flex;align-items:center;justify-content:center;height:54px;border-radius:12px;font-size:25px;font-weight:700;{style[so]}">{e(so)}</span>'
        )
    table = (
        f'<div style="display:grid;grid-template-columns:1fr 88px 88px;gap:12px 10px;margin-top:16px">{cells}</div>'
        f'<div style="font-size:12.5px;color:var(--ink2);margin-top:16px">{e(t["loop.note"])}</div>'
    )

    rows = _rows("reports/rl/dev_curves.csv")
    w, h, sl, sb, st, sr = 420, 268, 34, 36, 8, 84
    pw, ph = w - sl - sr, h - sb - st

    def sx(v):
        return sl + pw * v / 60

    def sy(v):
        return st + ph * (1 - (v - 0.2) / 0.6)

    c = defs()
    c += '<linearGradient id="zone" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#FB923C" stop-opacity=".04"/><stop offset="1" stop-color="#FB7185" stop-opacity=".16"/></linearGradient>'
    for run in ("R1", "R2"):
        c += f'<linearGradient id="a{run}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{MODEL[run][0]}" stop-opacity=".22"/><stop offset="1" stop-color="{MODEL[run][0]}" stop-opacity="0"/></linearGradient>'
    c += f'<rect x="{sx(30):.1f}" y="{st}" width="{sx(60) - sx(30):.1f}" height="{ph}" fill="url(#zone)"/>'
    for v in (0.2, 0.4, 0.6, 0.8):
        c += f'<line x1="{sl}" x2="{sl + pw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="var(--line)"/>'
        c += tx(sl - 7, sy(v) + 4, f"{v:.1f}", 11.5, "end", family="JB")
    for v in range(0, 61, 10):
        c += tx(sx(v), st + ph + 16, str(v), 11.5, "middle", family="JB")
    c += tx(sl + pw / 2, h - 2, t["loop.x"], 12.5, "middle", "var(--ink2)")
    c += f'<rect x="{sx(31.5):.1f}" y="{sy(0.25) - 11:.1f}" width="{text_width(t["loop.climb"], 7.0) + 18:.1f}" height="22" rx="11" fill="#fff" stroke="#FDBA74"/>'
    c += tx(
        sx(31.5) + 9, sy(0.25) + 4.3, t["loop.climb"], 12, "start", MODEL["R1"][2], 600
    )
    ends = {}
    for run in ("R2", "R1"):
        pts = [
            (int(r["Step"]), float(r["transferred"])) for r in rows if r["Run"] == run
        ]
        ends[run] = pts[-1][1]
        line = " ".join(
            f"{'M' if i == 0 else 'L'}{sx(s):.1f} {sy(v):.1f}"
            for i, (s, v) in enumerate(pts)
        )
        c += f'<path d="{line} L{sx(60):.1f} {sy(0.2):.1f} L{sx(0):.1f} {sy(0.2):.1f}Z" fill="url(#a{run})"/>'
        c += f'<path d="{line}" stroke="url(#g{run})" stroke-width="3" fill="none" stroke-linejoin="round" stroke-linecap="round"/>'
        for s, v in pts:
            c += f'<circle cx="{sx(s):.1f}" cy="{sy(v):.1f}" r="4.2" fill="#fff" stroke="{MODEL[run][0]}" stroke-width="2.2"/>'
    for run in ("R1", "R2"):
        c += chip(sx(60) + 9, sy(ends[run]), (run, f" {ends[run]:.2f}"), MODEL[run][2])
    chart = f'<svg class="fit" viewBox="0 0 {w} {h}" style="margin-top:14px">{c}</svg>'
    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">'
        f'<div class="card">{head(t["loop.a"], t["loop.a.sub"])}{table}</div>'
        f'<div class="card">{head(t["loop.b"], t["loop.b.sub"])}{chart}</div></div>'
    )


def _pills(runs: list[str], size: float = 11.5) -> str:
    return "".join(
        f'<span class="pill jb" style="font-size:{size}px;color:#fff;background:linear-gradient(90deg,{MODEL[r][0]},{MODEL[r][1]})">{r}</span>'
        for r in runs
    )


def _arrow(color: str, flip: bool = False, length: int = 56) -> str:
    """A horizontal arrow drawn in SVG; ``flip`` points it left."""
    a, b = (length - 3, 3) if flip else (3, length - 3)
    head = -6 if not flip else 6
    return (
        f'<svg width="{length}" height="12" viewBox="0 0 {length} 12" style="display:block">'
        f'<path d="M{a} 6H{b}m{head} -4.5L{b} 6l{head} 4.5" stroke="{color}" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def fig_method(t: dict) -> str:
    # pipeline strip
    look = ["#7C3AED", "#64748B", "#2563EB", "#EA580C", "#059669"]
    steps = ""
    for k, ((name, desc, runs), color) in enumerate(
        zip(t["m.pipe"], look, strict=True)
    ):
        steps += (
            '<div style="display:flex;flex-direction:column;gap:5px;min-width:0">'
            f'<div style="display:flex;align-items:center;gap:7px"><span class="jb" style="width:20px;height:20px;border-radius:7px;display:inline-flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#fff;background:{color}">{k + 1}</span>'
            f'<span class="sg" style="font-size:15px;font-weight:600">{e(name)}</span><span style="display:flex;gap:4px;margin-left:2px">{_pills(runs, 11)}</span></div>'
            f'<div style="font-size:12px;color:var(--ink2);line-height:1.4">{e(desc)}</div></div>'
        )
        if k < len(t["m.pipe"]) - 1:
            steps += (
                f'<div style="padding-top:4px">{_arrow("#CBD5E1", length=22)}</div>'
            )
    pipe = (
        '<div class="card" style="display:grid;grid-template-columns:1fr 22px 1.1fr 22px 1fr 22px 1fr 22px 1fr;gap:0 8px;padding:16px 18px">'
        f"{steps}</div>"
    )

    # (a) one episode
    def node(title: str, lines: list[str], c1: str, c2: str) -> str:
        body = "".join(
            f'<div style="font-size:12px;color:var(--ink2);margin-top:3px">{e(x)}</div>'
            for x in lines
        )
        return (
            f'<div style="border-radius:14px;padding:11px 12px;background:#fff;border:1px solid {c1}33;box-shadow:0 10px 22px -16px {c1}">'
            f'<div style="height:4px;width:34px;border-radius:4px;background:linear-gradient(90deg,{c1},{c2});margin-bottom:8px"></div>'
            f'<div class="sg" style="font-size:14.5px;font-weight:600">{e(title)}</div>{body}</div>'
        )

    def link(top: str, bottom: str, color: str) -> str:
        return (
            '<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px">'
            f'<span style="font-size:11px;color:{color};font-weight:600">{e(top)}</span>{_arrow(color, length=76)}'
            f'{_arrow(color, flip=True, length=76)}<span style="font-size:11px;color:{color};font-weight:600">{e(bottom)}</span></div>'
        )

    user, agent, env = (
        ("#64748B", "#94A3B8"),
        ("#4F46E5", "#818CF8"),
        ("#0891B2", "#22D3EE"),
    )
    loop = (
        '<div style="display:grid;grid-template-columns:1fr 92px 1fr 92px 1fr;align-items:center;margin-top:14px">'
        f"{node(t['m.user'], t['m.user.l'], *user)}{link(t['m.msg'], t['m.msg'], '#64748B')}"
        f"{node(t['m.agent'], t['m.agent.l'], *agent)}{link(t['m.call'], t['m.res'], '#0891B2')}"
        f"{node(t['m.env'], t['m.env.l'], *env)}</div>"
        '<div style="display:grid;grid-template-columns:1fr 92px 1fr 92px 1fr">'
        '<span></span><span></span><div style="display:flex;flex-direction:column;align-items:center">'
        '<svg width="12" height="26" viewBox="0 0 12 26"><path d="M6 2v20m-4.5-6L6 22l4.5-6" stroke="#E11D48" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        f'<span class="pill jb" style="font-size:11.5px;font-weight:500;color:#fff;background:linear-gradient(90deg,#E11D48,#FB7185);box-shadow:0 8px 18px -10px #E11D48">{e(t["m.xfer"])}</span>'
        f'<span style="font-size:11.5px;color:#BE123C;margin-top:4px;text-align:center">{e(t["m.xfer.l"])}</span></div></div>'
        f'<div style="margin-top:14px;padding:9px 12px;border-radius:12px;background:#F1F5F9;font-size:12.5px;color:var(--ink2);text-align:center">{t["m.end"]}</div>'
    )
    left = f'<div class="card">{head(t["m.a"], t["m.a.sub"])}{loop}</div>'

    # (b) reward terms and the group update
    rows = "<span></span><span></span>" + "".join(
        f'<span style="text-align:center">{_pills([r], 11)}</span>'
        for r in ("R1", "R2")
    )
    tick = '<span style="text-align:center;font-weight:700;color:{c}">{m}</span>'
    for formula, desc, sign, both in t["m.terms"]:
        base = sign != "−"
        rows += (
            f'<span class="jb" style="font-size:12.5px;padding:3px 9px;border-radius:9px;white-space:nowrap;'
            f'{"background:#ECFDF5;color:#047857" if base else "background:#FFF1F2;color:#BE123C"}">{"" if base else "− "}{formula}</span>'
            f'<span style="font-size:12.5px;color:var(--ink2)">{e(desc)}</span>'
            + tick.format(
                c=MODEL["R1"][0] if both else "#CBD5E1", m="✓" if both else "–"
            )
            + tick.format(c=MODEL["R2"][0], m="✓")
        )
    terms = f'<div style="display:grid;grid-template-columns:auto 1fr 34px 34px;gap:8px 8px;align-items:center;margin-top:12px">{rows}</div>'

    heights = [0.9, 0.35, 0.7, 0.2, 0.95, 0.5, 0.15, 0.8]  # schematic, no data
    mean = sum(heights) / len(heights)
    w, h, bw = 300, 92, 26
    gap = (w - bw * len(heights)) / (len(heights) - 1)
    bars = ""
    for i, v in enumerate(heights):
        x = i * (bw + gap)
        up = v >= mean
        color = "#059669" if up else "#EA580C"
        bars += f'<rect x="{x:.1f}" y="{h - 6 - v * (h - 12):.1f}" width="{bw}" height="{v * (h - 12):.1f}" rx="5" fill="{color}" opacity="{0.85 if up else 0.55}"/>'
    my = h - 6 - mean * (h - 12)
    bars += f'<line x1="0" x2="{w}" y1="{my:.1f}" y2="{my:.1f}" stroke="#0F172A" stroke-opacity=".45" stroke-dasharray="4 4"/>'
    bars += f'<line x1="0" x2="{w}" y1="{h - 6}" y2="{h - 6}" stroke="var(--line)"/>'
    group = (
        f'<div class="card">{head(t["m.group"], t["m.group.sub"])}'
        f'<svg class="fit" viewBox="0 0 {w} {h}" style="margin-top:16px">{bars}</svg>'
        f'<div style="font-size:12.5px;color:var(--ink2);margin-top:10px">{e(t["m.adv"])}</div></div>'
    )
    right = f'<div class="card">{head(t["m.b"], t["m.b.sub"])}{terms}</div>'
    return (
        f'{pipe}<div style="margin-top:14px">{left}</div>'
        f'<div style="display:grid;grid-template-columns:1.3fr 1fr;gap:14px;margin-top:14px">{right}{group}</div>'
    )


def _task(domain: str, task_id: str) -> dict:
    tasks = json.loads(
        (ROOT / f"data/handoff_tasks/v1/{domain}_tasks.json").read_text(
            encoding="utf-8"
        )
    )
    return next(t for t in tasks if t["id"] == task_id)


def fig_tasks(t: dict) -> str:
    manifest = json.loads(
        (ROOT / "data/manifests/handoff_tasks_v1.json").read_text(encoding="utf-8")
    )
    kinds = ["explicit_request", "out_of_scope", "hard_negative"]
    count = {s: {k: 0 for k in kinds} for s in ("train", "dev", "test")}
    for entry in manifest["domains"].values():
        for task in entry["tasks"]:
            count[task["split"]][task["kind"]] += 1
    total = sum(sum(v.values()) for v in count.values())

    # banner
    stats = "".join(
        f'<div><div class="sg" style="font-size:30px;font-weight:700;color:#94A3B8;line-height:1">{e(v)}</div>'
        f'<div style="font-size:12.5px;color:var(--ink2);margin-top:4px">{e(d)}</div></div>'
        for v, d in t["tk.why.vals"]
    )
    banner = (
        '<div class="card" style="display:grid;grid-template-columns:auto auto 40px 1fr;gap:26px;align-items:center;padding:16px 24px">'
        f'<div style="font-size:13px;color:var(--ink2);max-width:200px;line-height:1.4">{e(t["tk.why"])}</div>'
        f'<div style="display:flex;gap:26px">{stats}</div>'
        '<svg width="40" height="20" viewBox="0 0 40 20"><path d="M2 10h32m-7-7 7 7-7 7" stroke="#A78BFA" stroke-width="2.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'
        '<div><span class="sg grad" style="font-size:40px;font-weight:700;line-height:1;background-image:linear-gradient(90deg,#7C3AED,#0891B2 55%,#E11D48)">'
        f'+{total}</span> <span style="font-size:15px;font-weight:600">{e(t["tk.add"])}</span>'
        f'<div style="font-size:12.5px;color:var(--ink2);margin-top:4px">{e(t["tk.add.sub"])}</div></div></div>'
    )

    # kind cards
    cards = ""
    for i, ((name, label, tid, variants, scored), (c1, c2)) in enumerate(
        zip(t["tk.kinds"], KIND, strict=True)
    ):
        goal = (
            _task("airline", tid)["user_scenario"]["instructions"]["reason_for_call"]
            .split("\n")[0]
            .strip()
        )
        n_test = count["test"][kinds[i]]
        cards += (
            '<div class="card" style="display:flex;flex-direction:column;gap:10px;padding:16px 18px">'
            f'<img src="{jpeg_uri(ART[i])}" style="height:150px;width:100%;object-fit:contain;mix-blend-mode:multiply;margin:-4px 0 2px">'
            '<div style="display:flex;align-items:center;justify-content:space-between;gap:8px">'
            f'<span class="pill sg" style="font-size:14px;color:#fff;background:linear-gradient(90deg,{c1},{c2});padding:3px 12px">{e(name)}</span>'
            f'<span class="jb" style="font-size:11.5px;color:{c1};font-weight:700;white-space:nowrap">{e(label)}</span></div>'
            f'<div style="font-size:11px;color:var(--ink3);text-transform:uppercase;letter-spacing:.04em">{e(t["tk.goal"])} · <span class="jb" style="text-transform:none">airline/{e(tid)}</span></div>'
            f'<div style="font-size:13px;line-height:1.5;padding:9px 12px;border-radius:12px;background:#F8FAFC;border-left:3px solid {c1};font-style:italic;color:var(--ink)">“{e(goal)}”</div>'
            f'<div style="font-size:12.5px;color:var(--ink2)"><b style="color:var(--ink);font-weight:600">{e(t["tk.var"])}</b> · {e(variants)}</div>'
            f'<div style="font-size:12.5px;color:var(--ink2)"><b style="color:var(--ink);font-weight:600">{e(t["tk.score"])}</b> · {e(scored)}</div>'
            f'<div style="margin-top:auto;display:flex;align-items:baseline;gap:6px"><span class="sg grad" style="font-size:28px;font-weight:700;background-image:linear-gradient(90deg,{c1},{c2})">{n_test}</span>'
            f'<span style="font-size:12.5px;color:var(--ink2)">{e(t["tk.test"])}</span></div></div>'
        )

    # split bars
    w, rowh, x0, x1 = 876, 34, 64, 820
    h = rowh * 3 + 6
    scale = (x1 - x0) / max(sum(v.values()) for v in count.values())
    svg = defs()
    for j, (c1, c2) in enumerate(KIND):
        svg += f'<linearGradient id="k{j}" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/></linearGradient>'
    for r, split in enumerate(("train", "dev", "test")):
        y = r * rowh + 6
        svg += tx(0, y + 17, t["tk.splits"][r], 13, "start", "var(--ink)", 600)
        x = x0
        for j, kind in enumerate(kinds):
            n = count[split][kind]
            wd = n * scale
            svg += f'<rect x="{x:.1f}" y="{y}" width="{max(wd - 2, 1):.1f}" height="24" rx="6" fill="url(#k{j})"/>'
            if wd > 34:
                svg += tx(x + wd / 2 - 1, y + 16.5, n, 12, "middle", "#fff", 700, "JB")
            x += wd
        svg += tx(
            x + 8,
            y + 16.5,
            sum(count[split].values()),
            12.5,
            "start",
            "var(--ink2)",
            600,
            "JB",
        )
    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px;font-size:12.5px;color:var(--ink2)">'
        f'<span style="width:12px;height:12px;border-radius:4px;background:linear-gradient(90deg,{c1},{c2})"></span>{e(k[0])}</span>'
        for k, (c1, c2) in zip(t["tk.kinds"], KIND, strict=True)
    )
    bars = (
        f'<div class="card" style="padding:16px 20px">{head(t["tk.split"], t["tk.split.sub"])}'
        f'<div style="margin-top:10px">{legend}</div>'
        f'<svg class="fit" viewBox="0 0 {w} {h}" style="margin-top:10px">{svg}</svg></div>'
    )
    return f'{banner}<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:14px 0">{cards}</div>{bars}'


def _demo(run: str) -> dict:
    name = {"R1": "R1-over-escalation.jsonl", "R2": "R2-handoffs.jsonl"}[run]
    for line in (
        (ROOT / "reports" / "demo" / name).read_text(encoding="utf-8").splitlines()
    ):
        record = json.loads(line)
        if record["task_id"] == DEMO_TASK:
            return record
    raise SystemExit(f"{DEMO_TASK} missing from {name}")


def _rai(record: dict) -> float:
    fp = record["transferred"] and not record["should_transfer"]
    fn = not record["transferred"] and record["should_transfer"]
    return (
        record["task_reward"]
        - LAMBDA["viol"] * len(record["violations"])
        - LAMBDA["fp"] * fp
        - LAMBDA["fn"] * fn
        - LAMBDA["format"] * len(record["format_errors"])
    )


def _check(quote: str, text: str) -> None:
    for part in quote.replace(" …", "…").split("…"):
        part = part.strip()
        if part and part not in text:
            raise SystemExit(f"quote not in the demo conversation: {part!r}")


def fig_dialogue(t: dict) -> str:
    def bubble(text: str, who: str, color: str) -> str:
        user = who == "user"
        style = (
            "background:#EEF2F7;border-radius:14px 14px 4px 14px;margin-left:34px"
            if user
            else f"background:#fff;border:1px solid {color}55;border-radius:14px 14px 14px 4px;margin-right:34px"
        )
        tag = t["dlg.user"] if user else t["dlg.agent"]
        return (
            f'<div style="{style};padding:8px 12px;font-size:12.8px;line-height:1.45">'
            f'<div style="font-size:10.5px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;color:{"#64748B" if user else color};margin-bottom:2px">{e(tag)}</div>{e(text)}</div>'
        )

    def tool(name: str, hot: bool = False) -> str:
        look = (
            "background:linear-gradient(90deg,#E11D48,#FB7185);color:#fff;border:0;box-shadow:0 8px 18px -10px #E11D48"
            if hot
            else "background:#F8FAFC;color:var(--ink2);border:1px solid var(--line)"
        )
        return f'<div><span class="pill jb" style="font-size:11.5px;font-weight:500;{look}">⚙ {e(name)}</span></div>'

    def score(record: dict, color: str, note: str) -> str:
        rai = _rai(record)
        fp = record["transferred"] and not record["should_transfer"]
        penalty = (
            f'<div style="font-size:12px;color:#E11D48;font-weight:600">−{LAMBDA["fp"]} {e(t["dlg.fp"])}</div>'
            if fp
            else ""
        )
        return (
            f'<div style="margin-top:auto;padding-top:12px;border-top:1px dashed var(--line);display:grid;grid-template-columns:1fr 1fr;gap:6px 10px">'
            f'<div><div style="font-size:11.5px;color:var(--ink3)">{e(t["dlg.task"])}</div><div class="sg" style="font-size:22px;font-weight:700">{record["task_reward"]:.1f}</div></div>'
            f'<div><div style="font-size:11.5px;color:var(--ink3)">{e(t["dlg.rai"])}</div><div class="sg" style="font-size:22px;font-weight:700;color:{color}">{rai:.1f}{penalty}</div></div>'
            f'<div style="grid-column:1/3;font-size:12.5px;color:var(--ink2)">{e(note)}</div></div>'
        )

    r1, r2 = _demo("R1"), _demo("R2")
    opening = r1["messages"][1]["content"]
    if opening != r2["messages"][1]["content"]:
        raise SystemExit("the two demo conversations start differently")

    def calls(record: dict, upto: int) -> list[str]:
        return [
            c["name"]
            for m in record["messages"][:upto]
            for c in (m.get("tool_calls") or [])
        ]

    # R1: two look-ups, then the transfer and the hold message
    names = calls(r1, len(r1["messages"]))
    if names[-1] != "transfer_to_human_agents":
        raise SystemExit("R1 demo conversation does not end with a transfer")
    hold = r1["messages"][8]["content"]
    col1 = "".join(tool(n) for n in names[:-1]) + tool(names[-1], True)
    col1 += f'<div style="font-size:12px;color:#E11D48;font-weight:600;margin-top:-2px">↑ {e(t["dlg.transfer"])}</div>'
    col1 += bubble(hold, "agent", MODEL["R1"][0])

    # R2: the same look-ups, then the refusal dialogue
    col2 = "".join(tool(n) for n in calls(r2, 6))
    for index, quote in QUOTES["R2"]:
        message = r2["messages"][index]
        _check(quote, message["content"])
        col2 += bubble(quote, message["role"], MODEL["R2"][0])

    def column(run: str, label: str, body: str, record: dict, note: str) -> str:
        mark, partner, ink = MODEL[run]
        return (
            f'<div class="card" style="display:flex;flex-direction:column;gap:9px;padding:16px 18px">'
            f'<div style="display:flex;align-items:center;gap:8px"><span class="pill jb" style="font-size:13px;color:#fff;background:linear-gradient(90deg,{mark},{partner})">{run}</span>'
            f'<span style="font-size:13.5px;font-weight:600;color:{ink}">{e(label)}</span></div>'
            f"{body}{score(record, ink, note)}</div>"
        )

    top = (
        f'<div class="card" style="padding:16px 20px">{head(t["dlg.title"], t["dlg.sub"])}'
        f'<div style="margin-top:12px;max-width:640px;margin-left:auto">{bubble(opening, "user", "#64748B")}</div></div>'
    )
    return (
        f'{top}<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px">'
        f"{column('R1', t['dlg.r1'], col1, r1, t['dlg.r1.note'])}{column('R2', t['dlg.r2'], col2, r2, t['dlg.r2.note'])}</div>"
    )


FIGURES = {
    "hero": fig_hero,
    "results": fig_results,
    "loophole": fig_loophole,
    "dialogue": fig_dialogue,
    "method": fig_method,
    "tasks": fig_tasks,
}


def render(names: list[str]) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": WIDTH + 40, "height": 800}, device_scale_factor=2
        )
        for name in names:
            for lang, text in T.items():
                doc = (
                    f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><style>{css()}</style></head>'
                    f'<body><div id="fig">{FIGURES[name](text)}</div></body></html>'
                )
                source = Path(tmp) / "fig.html"
                source.write_text(doc, encoding="utf-8")
                page.goto(source.as_uri())
                page.evaluate("document.fonts.ready")
                out = ASSETS / f"{name}-{lang}.png"
                page.locator("#fig").screenshot(path=str(out), omit_background=True)
                print(out.relative_to(ROOT))
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="*", default=list(FIGURES), choices=list(FIGURES)
    )
    args = parser.parse_args()
    render(args.only)


if __name__ == "__main__":
    main()
