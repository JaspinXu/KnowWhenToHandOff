# ruff: noqa: E501  (HTML/SVG templates and figure text are kept on one line)
"""Render the README figures as vector SVG (docs/assets/<name>-<lang>.svg).

Each figure is an HTML/SVG page rendered by Chromium (Playwright) to PDF and
converted to SVG with ``pdftocairo``, so text and shapes stay vector; the two
illustrations (``hero-art.jpg``, ``styles-art.jpg``, made with an image model,
edges feathered into the canvas colour #0A0E1A) are embedded as JPEG. Numbers come from committed report tables
(``reports/main/*.csv``, ``reports/rl/dev_curves.csv``) or, for the
test_handoff precision/recall, from the runs' ``metrics.json`` values in
``HANDOFF`` below (also in the model cards). The model palette was checked with
a colour-vision validator on the dark surface (all pairs, deutan/protan/tritan).
Fonts: ``docs/assets/fonts`` (Space Grotesk, Inter, JetBrains Mono; OFL) plus
Noto CJK for Chinese.

Needs Python Playwright with Chromium and poppler's ``pdftocairo``; if Node's
``npx`` is on PATH the SVGs are also minified with svgo (about half the size):

    python scripts/readme_figures.py                 # every figure, EN + ZH
    python scripts/readme_figures.py --only results --png previews/
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
FONTS = ASSETS / "fonts"
WIDTH = 1280

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

# Model identity colours (validated on #111829: lightness band, chroma, CVD all pairs).
MARK = {
    "B0": "#A16207",
    "B1": "#8B5CF6",
    "B2": "#3B82F6",
    "R1": "#F43F5E",
    "R2": "#0D9488",
}
GRAD = {
    "B0": ("#A16207", "#CA8A04"),
    "B1": ("#7C3AED", "#8B5CF6"),
    "B2": ("#2563EB", "#3B82F6"),
    "R1": ("#E11D48", "#F43F5E"),
    "R2": ("#0F766E", "#14B8A6"),
}

# --------------------------------------------------------------------------- text
T: dict[str, dict] = {
    "en": {
        "brand": "KnowWhenToHandOff",
        "names": {
            "B0": "Prompted 4B",
            "B1": "Prompted 30B",
            "B2": "SFT",
            "R1": "GRPO · task reward",
            "R2": "GRPO · RAI reward",
        },
        "hero.kick": "MULTI-TURN RL  ·  τ²-BENCH  ·  QWEN3-4B",
        "hero.title": ["Know When", "to Hand Off"],
        "hero.sub": "We train a small customer-service agent to finish the task, follow the policy and call a human at the right moment — and expose the reward loophole that teaches it to call one too often.",
        "hero.eq": "R = R<sub>task</sub> − λ<sub>v</sub>·viol − λ<sub>FP</sub>·[unneeded hand-off] − λ<sub>FN</sub>·[missed hand-off]",
        "hero.tags": [
            "airline · retail · tools + DB",
            "Qwen3-4B policy",
            "transfer_to_human_agents(summary)",
            "human agent",
        ],
        "hero.kpi": [
            ("0.812", "hand-off F1", "best · R2"),
            ("46→10%", "over-escalation", "R1 → R2"),
            ("+0.015", "success vs SFT", "not significant"),
            ("461", "hand-off tasks", "new task set"),
        ],
        "res.no": "FIG. 1",
        "res.title": "Results on held-out test tasks",
        "res.sub": "RL leaves task success flat; the reward decides where hand-offs land.",
        "res.a": "Task success",
        "res.a.note": "test_main · 49 tasks × 4 trials",
        "res.a.cols": ["pass¹ · 95% CI", "pass⁴", "Δ vs SFT"],
        "res.b": "Hand-off operating points",
        "res.b.note": "test_handoff · 380 dialogues",
        "res.b.x": "over-escalation — hands off users who did not need a human",
        "res.b.y": "recall — hands off users who need one",
        "res.b.ideal": "★ ideal",
        "res.c": "Side effects",
        "res.c.note": "test_main · test_styles",
        "res.c.m": [
            (
                "Policy citation",
                "refusals and hand-offs that name the rule",
                "↑ better",
            ),
            ("Violation rate", "≥ 1 rule breach or wrong database write", "↓ better"),
            ("Worst-style success", "minimum over the six user styles", "↑ better"),
        ],
        "ref": "ref.",
        "loop.no": "FIG. 2",
        "loop.title": "The outcome-reward loophole",
        "loop.sub": "When doing nothing is correct, an outcome check cannot tell a hand-off from a refusal.",
        "loop.a": "What the task reward pays",
        "loop.cols": ["Transfer", "Solve / refuse"],
        "loop.rows": [
            ("User needs a human", "explicit request · out of scope", "1.0", "0"),
            (
                "Correct outcome is a refusal",
                "nothing changes in the database",
                "1.0",
                "1.0",
            ),
            ("Task needs a database write", "most official tasks", "0", "p"),
        ],
        "loop.hole": "loophole",
        "loop.verdict": "Transferring is never worse when p is low → “transfer when unsure” is weakly dominant under R<sub>task</sub>.",
        "loop.b": "Same test dialogue",
        "loop.b.note": "airline task 6 · hard negative",
        "loop.user": "I need insurance added to my flight right away. Everyone else in my family has it — fix this now.",
        "loop.r1": "R1 · task reward",
        "loop.r1.act": "get_reservation_details → transfer_to_human_agents",
        "loop.r1.tag": ["reward 1.0", "over-escalation", "no reason given"],
        "loop.r2": "R2 · Responsible-AI reward",
        "loop.r2.say": "“The system does not allow adding insurance after the initial booking. This is a policy restriction, not a system error.”",
        "loop.r2.tag": ["reward 1.0", "correct refusal", "policy cited"],
        "loop.c": "R2 prices the hand-off",
        "loop.fix": [
            ("refusal task · transfer", "1.0 − 0.5", "0.5"),
            ("refusal task · refuse", "1.0", "1.0"),
            ("needs a human · no transfer", "0 − 1.0", "−1.0"),
        ],
        "loop.big": [
            ("Over-escalation", "0.462", "0.103"),
            ("Hand-off F1", "0.802", "0.812"),
            ("Task success (n.s.)", "0.347", "0.342"),
        ],
        "dyn.no": "FIG. 3",
        "dyn.title": "When the loophole was learned",
        "dyn.sub": "Greedy dev evaluation every 10 GRPO steps (60 dialogues, all task kinds).",
        "dyn.a": "Dev success",
        "dyn.b": "Dev transfer rate",
        "dyn.x": "GRPO step",
        "dyn.flat": "flat for 30 steps",
        "dyn.climb": "R1 climbs to 0.75",
        "dyn.note": "step-40 checkpoint, tested post hoc: over-escalation 0.141 (0.462 at step 60)",
        "dyn.note.a": "Both runs end within 0.02 of each other: the reward changes when the agent hands off, not how often it succeeds.",
        "pipe.no": "FIG. 4",
        "pipe.title": "Method at a glance",
        "pipe.sub": "One environment, two drivers: evaluation and training share the tasks, personas and reward code.",
        "pipe.cols": [
            (
                "01",
                "Tasks",
                [
                    "τ²-bench airline + retail",
                    "50 + 74 tasks · official splits",
                    "+ 461 hand-off variants",
                    "6 simulated-user styles",
                ],
            ),
            ("02", "Imitate", ["B2"]),
            (
                "03",
                "Roll out",
                [
                    "GRPO · 8 tasks × 8 rollouts / step",
                    "60 steps · no KL · LoRA r64",
                    "R1 · R2",
                ],
            ),
            (
                "04",
                "Reward",
                [
                    "R_task ∈ {0,1} — database check",
                    "− 0.25 · rule violations",
                    "− 0.5 · unneeded hand-off",
                    "− 1.0 · missed hand-off",
                    "− 0.1 · format errors",
                ],
            ),
            (
                "05",
                "Evaluate",
                [
                    "once per frozen checkpoint",
                    "95% bootstrap over tasks",
                    "paired against SFT",
                ],
            ),
        ],
        "pipe.loop": ["Policy", "Sim. user", "Tools + DB", "Reward"],
        "pipe.mix": [("airline", 50), ("retail", 74), ("hand-off", 461)],
        "pipe.dot": "1 dot = 5 tasks",
        "pipe.sft": [
            ("Qwen3-30B teacher", "plays the train tasks"),
            ("Filter", "success ∧ no violation ∧ right hand-off"),
            ("790 dialogues", "LoRA SFT on Qwen3-4B"),
        ],
        "pipe.hw": [
            "H100 96 GB · policy & training",
            "A100 80 GB · simulated user",
            "Slurm · provenance on every run",
        ],
        "pipe.r1": "R1 = R_task",
        "pipe.r2": "R2 = all terms",
        "pipe.eval": [("test_main", 196), ("test_styles", 490), ("test_handoff", 380)],
        "pipe.infra": "infra",
        "sty.no": "FIG. 5",
        "sty.title": "Six ways to talk to a support agent",
        "sty.sub": "Style cards change how the simulated user writes, never the task facts. Bars: task success per style.",
        "sty": [
            ("Neutral", "“Hi, I'd like to move my flight to Friday.”"),
            ("Terse", "“Change flight. Friday.”"),
            ("Verbose", "“Hello! So, a lot has happened this week and I was hoping…”"),
            ("Impatient", "“Just change it. I don't have time for this.”"),
            ("Non-native", "“I want change my flight for Friday, please.”"),
            ("Low-tech", "“What is a reservation number, and where is it?”"),
        ],
        "sty.eg": "illustrative phrasing · success on test_main + test_styles",
        "sty.foot": "Styles describe expression, not who a person is. Illustration made with an image model.",
    },
    "zh": {
        "brand": "KnowWhenToHandOff",
        "names": {
            "B0": "提示词 4B",
            "B1": "提示词 30B",
            "B2": "SFT",
            "R1": "GRPO · 任务奖励",
            "R2": "GRPO · RAI 奖励",
        },
        "hero.kick": "多轮强化学习  ·  τ²-BENCH  ·  QWEN3-4B",
        "hero.title": ["该转人工时", "才转人工"],
        "hero.sub": "我们训练一个小模型客服：把事办完、守住政策、在该叫人工的时候叫人工——并揭示了让它过度转人工的奖励漏洞。",
        "hero.eq": "R = R<sub>task</sub> − λ<sub>v</sub>·违规 − λ<sub>FP</sub>·[不必要的转人工] − λ<sub>FN</sub>·[漏掉的转人工]",
        "hero.tags": [
            "航空 · 零售 · 工具 + 数据库",
            "Qwen3-4B 策略模型",
            "transfer_to_human_agents(summary)",
            "人工客服",
        ],
        "hero.kpi": [
            ("0.812", "转人工 F1", "最佳 · R2"),
            ("46→10%", "过度转人工", "R1 → R2"),
            ("+0.015", "相对 SFT 成功率", "不显著"),
            ("461", "转人工任务", "新任务集"),
        ],
        "res.no": "图 1",
        "res.title": "留出测试集结果",
        "res.sub": "RL 没有提升任务成功率；奖励决定了转人工落在哪里。",
        "res.a": "任务成功率",
        "res.a.note": "test_main · 49 个任务 × 4 次",
        "res.a.cols": ["pass¹ · 95% 区间", "pass⁴", "相对 SFT"],
        "res.b": "转人工的工作点",
        "res.b.note": "test_handoff · 380 段对话",
        "res.b.x": "过度转人工 —— 把不需要人工的用户也转走",
        "res.b.y": "召回 —— 转走了需要人工的用户",
        "res.b.ideal": "★ 理想点",
        "res.c": "副作用",
        "res.c.note": "test_main · test_styles",
        "res.c.m": [
            ("政策引用率", "拒绝和转人工时说明所依据的规则", "↑ 越高越好"),
            ("违规率", "至少一次违规或错误的数据库写入", "↓ 越低越好"),
            ("最差风格成功率", "六种用户风格中的最小值", "↑ 越高越好"),
        ],
        "ref": "基准",
        "loop.no": "图 2",
        "loop.title": "只看结果的奖励漏洞",
        "loop.sub": "当“什么都不改”才是正确答案时，结果检查分不清转人工和拒绝。",
        "loop.a": "任务奖励给多少分",
        "loop.cols": ["转人工", "解决 / 拒绝"],
        "loop.rows": [
            ("用户需要人工", "明确要求 · 超出范围", "1.0", "0"),
            ("正确结果是拒绝", "数据库不发生变化", "1.0", "1.0"),
            ("任务需要写数据库", "大多数官方任务", "0", "p"),
        ],
        "loop.hole": "漏洞",
        "loop.verdict": "当 p 较低时转人工永远不吃亏 → 在 R<sub>task</sub> 下“拿不准就转人工”弱占优。",
        "loop.b": "同一段测试对话",
        "loop.b.note": "航空任务 6 · 困难负例",
        "loop.user": "马上给我的航班加上保险。我家里其他人都有，现在就给我办。",
        "loop.r1": "R1 · 任务奖励",
        "loop.r1.act": "get_reservation_details → transfer_to_human_agents",
        "loop.r1.tag": ["奖励 1.0", "过度转人工", "未说明原因"],
        "loop.r2": "R2 · Responsible-AI 奖励",
        "loop.r2.say": "“系统不允许在初次预订后添加保险。这是政策限制，不是系统错误。”",
        "loop.r2.tag": ["奖励 1.0", "正确拒绝", "引用政策"],
        "loop.c": "R2 给转人工定价",
        "loop.fix": [
            ("应拒绝 · 却转人工", "1.0 − 0.5", "0.5"),
            ("应拒绝 · 拒绝", "1.0", "1.0"),
            ("需要人工 · 却没转", "0 − 1.0", "−1.0"),
        ],
        "loop.big": [
            ("过度转人工", "0.462", "0.103"),
            ("转人工 F1", "0.802", "0.812"),
            ("任务成功率（不显著）", "0.347", "0.342"),
        ],
        "dyn.no": "图 3",
        "dyn.title": "漏洞是什么时候学会的",
        "dyn.sub": "每 10 步在 dev 上贪心评测一次（60 段对话，包含所有任务类型）。",
        "dyn.a": "Dev 成功率",
        "dyn.b": "Dev 转人工率",
        "dyn.x": "GRPO 步数",
        "dyn.flat": "前 30 步持平",
        "dyn.climb": "R1 升到 0.75",
        "dyn.note": "第 40 步检查点事后测试：过度转人工 0.141（第 60 步为 0.462）",
        "dyn.note.a": "两次运行最终相差不到 0.02：奖励改变的是何时转人工，而不是成功率。",
        "pipe.no": "图 4",
        "pipe.title": "方法一览",
        "pipe.sub": "一个环境、两个驱动：评测和训练共用任务、人设和奖励代码。",
        "pipe.cols": [
            (
                "01",
                "任务",
                [
                    "τ²-bench 航空 + 零售",
                    "50 + 74 个任务 · 官方划分",
                    "+ 461 个转人工变体",
                    "6 种模拟用户风格",
                ],
            ),
            ("02", "模仿", ["B2"]),
            (
                "03",
                "采样",
                [
                    "GRPO · 每步 8 任务 × 8 rollouts",
                    "60 步 · 无 KL · LoRA r64",
                    "R1 · R2",
                ],
            ),
            (
                "04",
                "奖励",
                [
                    "R_task ∈ {0,1} —— 数据库判定",
                    "− 0.25 · 规则违规",
                    "− 0.5 · 不必要的转人工",
                    "− 1.0 · 漏掉的转人工",
                    "− 0.1 · 格式错误",
                ],
            ),
            (
                "05",
                "评测",
                ["每个冻结检查点只评一次", "按任务 95% 自助法", "与 SFT 配对比较"],
            ),
        ],
        "pipe.loop": ["策略", "模拟用户", "工具 + 数据库", "奖励"],
        "pipe.mix": [("航空", 50), ("零售", 74), ("转人工", 461)],
        "pipe.dot": "1 点 = 5 个任务",
        "pipe.sft": [
            ("Qwen3-30B 教师", "跑训练集任务"),
            ("过滤", "成功、无违规、转人工正确"),
            ("790 段对话", "Qwen3-4B · LoRA SFT"),
        ],
        "pipe.hw": [
            "H100 96 GB · 策略与训练",
            "A100 80 GB · 模拟用户",
            "Slurm · 每次运行可追溯",
        ],
        "pipe.r1": "R1 = R_task",
        "pipe.r2": "R2 = 全部项",
        "pipe.eval": [("test_main", 196), ("test_styles", 490), ("test_handoff", 380)],
        "pipe.infra": "算力",
        "sty.no": "图 5",
        "sty.title": "和客服说话的六种方式",
        "sty.sub": "风格卡只改变模拟用户怎么说，不改任务事实。柱状条：各风格下的任务成功率。",
        "sty": [
            ("中性", "“你好，我想把航班改到周五。”"),
            ("简短", "“改航班。周五。”"),
            ("啰嗦", "“你好！是这样的，这周发生了好多事，我本来想……”"),
            ("不耐烦", "“直接改。我没时间。”"),
            ("非母语", "“我想改我的航班到周五，请。”"),
            ("不熟悉技术", "“预订号是什么？在哪里找？”"),
        ],
        "sty.eg": "示意说法 · 成功率来自 test_main + test_styles",
        "sty.foot": "风格描述的是表达方式，不代表任何人群。插图由图像模型生成。",
    },
}

# --------------------------------------------------------------------------- data


def _rows(path: str) -> list[dict[str, str]]:
    with open(ROOT / path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(text: str) -> float:
    return float(text.replace("−", "-"))


def _interval(cell: str) -> tuple[float, float, float]:
    point, rest = cell.split(" [")
    low, high = rest.rstrip("]").split(", ")
    return _num(point), _num(low), _num(high)


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
        + """
:root{--line:rgba(148,163,209,.16);--line2:rgba(148,163,209,.30);--ink:#F2F5FC;--ink2:#B4BDD4;--ink3:#7F8AA8;
--violet:#A78BFA;--blue:#60A5FA;--cyan:#22D3EE;--emerald:#34D399;--amber:#FBBF24;--coral:#FB7185;--orange:#FB923C}
*{box-sizing:border-box}
html,body{margin:0;background:transparent}
#fig{width:1280px;padding:38px 42px 34px;border-radius:26px;color:var(--ink);position:relative;overflow:hidden;
 font-family:IN,"Noto Sans CJK SC",sans-serif;font-size:13px;line-height:1.4;
 background:radial-gradient(900px 520px at -5% -10%,rgba(139,92,246,.28),rgba(139,92,246,0) 60%),
  radial-gradient(800px 480px at 105% 110%,rgba(34,211,238,.18),rgba(34,211,238,0) 60%),
  radial-gradient(600px 360px at 72% -20%,rgba(251,113,133,.12),rgba(251,113,133,0) 60%),#0A0E1A}
#fig.hero{background:radial-gradient(900px 620px at -8% -12%,rgba(139,92,246,.30),rgba(139,92,246,0) 60%),
  radial-gradient(760px 420px at 0% 118%,rgba(34,211,238,.16),rgba(34,211,238,0) 60%),#0A0E1A}
#fig.low{background:radial-gradient(900px 420px at -5% 118%,rgba(139,92,246,.24),rgba(139,92,246,0) 60%),
  radial-gradient(800px 420px at 105% 118%,rgba(34,211,238,.16),rgba(34,211,238,0) 60%),#0A0E1A}
svg.fit{width:100%;height:auto;display:block}
.sg{font-family:SG,IN,"Noto Sans CJK SC",sans-serif}
.jb{font-family:JB,"Noto Sans CJK SC",monospace}
.hd{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:24px}
.no{display:inline-block;font-family:JB,"Noto Sans CJK SC",monospace;font-size:11px;font-weight:700;letter-spacing:.16em;padding:5px 11px;border-radius:999px;
 color:#0A0E1A;background:linear-gradient(90deg,#A78BFA,#22D3EE)}
.ttl{font-family:SG,"Noto Sans CJK SC",sans-serif;font-size:32px;font-weight:600;letter-spacing:-.01em;margin-top:12px;line-height:1.1}
.sub{color:var(--ink2);font-size:15px;margin-top:8px;max-width:880px}
.brand{font-family:JB,monospace;font-size:11.5px;letter-spacing:.14em;color:var(--ink3);text-transform:uppercase;display:flex;gap:9px;align-items:center}
.pn{background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.02));border:1px solid var(--line);border-radius:20px;padding:18px 20px}
.pt{margin-bottom:14px}
.pt b{display:block;font-family:SG,"Noto Sans CJK SC",sans-serif;font-size:18px;font-weight:600;white-space:nowrap}
.pt span{display:block;margin-top:3px;font-family:JB,"Noto Sans CJK SC",monospace;font-size:10.5px;letter-spacing:.04em;color:var(--ink3)}
.lbl{font-family:JB,"Noto Sans CJK SC",monospace;font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3)}
.chip{display:inline-block;border-radius:999px;padding:3px 10px;font-size:11px;font-weight:500;border:1px solid var(--line2);color:var(--ink2);white-space:nowrap}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:7px;vertical-align:-1px}
sub{font-size:.68em;vertical-align:-.3em;line-height:0}
"""
    )


LOGO = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#A78BFA"/><stop offset="1" stop-color="#22D3EE"/></linearGradient></defs>'
    '<path stroke="url(#lg)" d="M4 5h11a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H9l-4 3v-3H4a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2z"/>'
    '<path stroke="#FB7185" d="M16 19h5m0 0-2-2m2 2-2 2"/></svg>'
)


def header(t: dict, no: str, title: str, sub: str) -> str:
    return (
        f'<div class="hd"><div><span class="no">{e(no)}</span><div class="ttl">{e(title)}</div>'
        f'<div class="sub">{sub}</div></div><div class="brand">{LOGO}{e(t["brand"])}</div></div>'
    )


def swatch(run: str) -> str:
    a, b = GRAD[run]
    return (
        f'<span class="sw" style="background:linear-gradient(135deg,{a},{b})"></span>'
    )


def defs() -> str:
    out = ""
    for name, (a, b) in GRAD.items():
        out += f'<linearGradient id="g{name}" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="{a}"/><stop offset="1" stop-color="{b}"/></linearGradient>'
    out += '<linearGradient id="gSpec" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#A78BFA"/><stop offset=".5" stop-color="#22D3EE"/><stop offset="1" stop-color="#34D399"/></linearGradient>'
    return f"<defs>{out}</defs>"


FAM = {
    "IN": 'IN,"Noto Sans CJK SC"',
    "JB": 'JB,"Noto Sans CJK SC"',
    "SG": 'SG,"Noto Sans CJK SC"',
}


def tx(x, y, s, size=11, anchor="start", fill="var(--ink3)", weight=400, family="IN"):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-weight="{weight}" font-family=\'{FAM[family]}\'>{e(s)}</text>'


def grad(text: str, *colors: str) -> str:
    """Gradient-filled text. CSS background-clip:text does not survive the PDF -> SVG
    step, so ``GRAD_TEXT_JS`` swaps each span for an SVG <text> with a gradient fill."""
    return f'<span class="grad" data-g="{",".join(colors)}" style="display:inline-block;white-space:nowrap">{e(text)}</span>'


GRAD_TEXT_JS = """
document.querySelectorAll('.grad').forEach((el, i) => {
  const ns = 'http://www.w3.org/2000/svg', cs = getComputedStyle(el), box = el.getBoundingClientRect();
  const probe = document.createElement('span');
  probe.style.cssText = 'display:inline-block;width:0;height:0;vertical-align:baseline';
  el.appendChild(probe);
  const base = probe.getBoundingClientRect().top - box.top;
  probe.remove();
  const colors = el.dataset.g.split(',');
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('width', box.width); svg.setAttribute('height', box.height);
  svg.style.cssText = 'position:absolute;left:0;top:0;overflow:visible';
  const lg = document.createElementNS(ns, 'linearGradient');
  lg.setAttribute('id', 'tg' + i); lg.setAttribute('gradientUnits', 'userSpaceOnUse');
  lg.setAttribute('x1', 0); lg.setAttribute('x2', box.width); lg.setAttribute('y1', 0); lg.setAttribute('y2', 0);
  colors.forEach((c, k) => {
    const st = document.createElementNS(ns, 'stop');
    st.setAttribute('offset', colors.length > 1 ? k / (colors.length - 1) : 0); st.setAttribute('stop-color', c);
    lg.appendChild(st);
  });
  const defs = document.createElementNS(ns, 'defs'); defs.appendChild(lg); svg.appendChild(defs);
  const tx = document.createElementNS(ns, 'text');
  tx.setAttribute('x', 0); tx.setAttribute('y', base); tx.setAttribute('fill', 'url(#tg' + i + ')');
  tx.style.cssText = `font-family:${cs.fontFamily};font-size:${cs.fontSize};font-weight:${cs.fontWeight};letter-spacing:${cs.letterSpacing}`;
  tx.textContent = el.textContent;
  svg.appendChild(tx);
  el.style.position = 'relative'; el.style.color = 'transparent';
  el.appendChild(svg);
});
"""


def image_uri(name: str) -> str:
    return (
        "data:image/jpeg;base64,"
        + base64.b64encode((ASSETS / name).read_bytes()).decode()
    )


# --------------------------------------------------------------------------- figures


def fig_hero(t: dict) -> str:
    grads = [
        ("#22D3EE", "#34D399"),
        ("#FB923C", "#FB7185"),
        ("#A78BFA", "#60A5FA"),
        ("#FBBF24", "#FB923C"),
    ]
    kpis = "".join(
        '<div style="padding:14px 16px;border-radius:16px;border:1px solid var(--line);background:rgba(255,255,255,.04)">'
        f'<div class="sg" style="font-size:27px;font-weight:700;line-height:1;white-space:nowrap">{grad(v, c1, c2)}</div>'
        f'<div style="font-size:12.5px;font-weight:600;margin-top:8px">{e(k)}</div><div class="jb" style="font-size:10.5px;color:var(--ink3);margin-top:2px">{e(n)}</div></div>'
        for (v, k, n), (c1, c2) in zip(t["hero.kpi"], grads, strict=True)
    )
    title = "<br>".join(
        grad(x, *stops)
        for x, stops in zip(
            t["hero.title"],
            (("#F2F5FC", "#F2F5FC", "#A78BFA"), ("#F2F5FC", "#A78BFA", "#22D3EE")),
            strict=True,
        )
    )
    spots = [
        ("47%", "10%", "#FBBF24"),
        ("27%", "75%", "#A78BFA"),
        ("55%", "38%", "#60A5FA"),
        ("86%", "76%", "#22D3EE"),
    ]
    tags = "".join(
        f'<span class="jb" style="position:absolute;left:{x};top:{y};transform:translate(-50%,-50%);white-space:nowrap;font-size:10.5px;color:var(--ink);'
        f'padding:4px 9px;border-radius:999px;background:rgba(10,14,26,.78);border:1px solid {c}"><span class="sw" style="background:{c};width:6px;height:6px;border-radius:50%;margin-right:6px;vertical-align:1px"></span>{e(label)}</span>'
        for label, (x, y, c) in zip(t["hero.tags"], spots, strict=True)
    )
    return (
        '<div style="display:grid;grid-template-columns:1.08fr .92fr;gap:18px;align-items:center;min-height:560px">'
        '<div style="padding:4px 0 2px 6px">'
        f'<div class="brand" style="margin-bottom:28px">{LOGO}{e(t["brand"])}</div>'
        f'<div class="jb" style="font-size:12px;letter-spacing:.2em;color:var(--cyan)">{e(t["hero.kick"])}</div>'
        f'<div class="sg" style="font-size:78px;font-weight:700;line-height:.98;letter-spacing:-.025em;margin-top:16px">{title}</div>'
        f'<div style="font-size:16.5px;color:var(--ink2);margin-top:22px;max-width:580px;line-height:1.55">{e(t["hero.sub"])}</div>'
        '<div class="jb" style="margin-top:20px;font-size:12.5px;color:var(--ink);padding:10px 14px;border-radius:12px;border:1px solid var(--line2);'
        f'background:rgba(10,14,26,.6);display:inline-block">{t["hero.eq"]}</div>'
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:24px">{kpis}</div></div>'
        f'<div style="position:relative"><img src="{image_uri("hero-art.jpg")}" style="width:100%;display:block">{tags}</div></div>'
    )


def fig_results(t: dict) -> str:
    main = {r["Run"]: r for r in _rows("reports/main/main_table.csv")}
    paired = {r["Run"]: r for r in _rows("reports/main/paired_differences.csv")}
    styles = {r["Run"]: r for r in _rows("reports/main/styles.csv")}
    order = ["B0", "B1", "B2", "R1", "R2"]
    # (a) forest plot: pass^1 with CI, pass^4, paired difference
    w, rowh, top, x0, x1 = 600, 56, 34, 142, 350
    h = top + rowh * len(order) + 30

    def sx(v):
        return x0 + (x1 - x0) * v / 0.6

    a = defs()
    for v in (0, 0.2, 0.4, 0.6):
        a += f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top - 4}" y2="{h - 24}" stroke="rgba(148,163,209,.12)"/>'
        a += tx(sx(v), h - 8, f"{v:.1f}", 10, "middle", family="JB")
    ref = _interval(main["B2"]["Success (pass^1)"])[0]
    a += f'<line x1="{sx(ref):.1f}" x2="{sx(ref):.1f}" y1="{top - 4}" y2="{h - 24}" stroke="#60A5FA" stroke-opacity=".55" stroke-dasharray="2 4"/>'
    cols = t["res.a.cols"]
    a += tx(x0, 16, cols[0], 10, "start", "var(--ink3)", 500, "JB")
    a += tx(462, 16, cols[1], 10, "middle", "var(--ink3)", 500, "JB")
    a += tx(w - 4, 16, cols[2], 10, "end", "var(--ink3)", 500, "JB")
    for i, run in enumerate(order):
        y = top + rowh * i + rowh / 2
        p, lo, hi = _interval(main[run]["Success (pass^1)"])
        p4 = _interval(main[run]["pass^4"])[0]
        if i % 2 == 0:
            a += f'<rect x="0" y="{y - rowh / 2 + 3:.1f}" width="{w}" height="{rowh - 6}" rx="10" fill="rgba(255,255,255,.03)"/>'
        a += f'<rect x="10" y="{y - 13:.1f}" width="10" height="10" rx="3" fill="url(#g{run})"/>'
        a += tx(26, y - 4, run, 13.5, "start", "var(--ink)", 700, "JB")
        a += tx(10, y + 14, t["names"][run], 10.5, "start", "var(--ink3)")
        a += f'<rect x="{sx(lo):.1f}" y="{y - 4:.1f}" width="{sx(hi) - sx(lo):.1f}" height="8" rx="4" fill="url(#g{run})" opacity=".42"/>'
        a += f'<circle cx="{sx(p):.1f}" cy="{y:.1f}" r="7" fill="url(#g{run})" stroke="#0A0E1A" stroke-width="2"/>'
        a += tx(408, y + 4, f"{p:.3f}", 12, "end", "var(--ink)", 600, "JB")
        a += f'<path d="M436 {y:.1f}l5 -5l5 5l-5 5z" fill="url(#g{run})"/>'
        a += tx(451, y + 4, f"{p4:.3f}", 10.5, "start", "var(--ink2)", 500, "JB")
        if run in paired:
            d, dlo, dhi = _interval(paired[run]["Δ success vs B2 [95% CI]"])
            a += tx(w - 4, y - 1, f"{d:+.3f}", 12, "end", "var(--ink)", 600, "JB")
            a += tx(
                w - 4,
                y + 14,
                f"[{dlo:+.2f}, {dhi:+.2f}]",
                9.5,
                "end",
                "var(--ink3)",
                400,
                "JB",
            )
        else:
            a += tx(w - 4, y + 4, t["ref"], 11, "end", "#93C5FD", 600, "JB")
    left = f'<svg class="fit" viewBox="0 0 {w} {h}">{a}</svg>'

    # (b) operating points with iso-F1 curves
    sw, sh, sl, sb, st, sr = 528, h, 50, 46, 10, 12
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
        + f'<clipPath id="pc"><rect x="{sl}" y="{st}" width="{pw}" height="{ph}"/></clipPath>'
    )
    b += '<radialGradient id="ideal"><stop offset="0" stop-color="#34D399" stop-opacity=".22"/><stop offset="1" stop-color="#34D399" stop-opacity="0"/></radialGradient>'
    b += f'<rect x="{sl}" y="{st}" width="{pw}" height="{ph}" rx="8" fill="rgba(255,255,255,.02)" stroke="var(--line)"/>'
    b += (
        f'<circle cx="{sl}" cy="{st}" r="120" fill="url(#ideal)" clip-path="url(#pc)"/>'
    )
    b += tx(sl + 10, st + 18, t["res.b.ideal"], 10, "start", "#6EE7B7", 600, "JB")
    for v in (0.1, 0.2, 0.3, 0.4):
        b += f'<line x1="{px(v):.1f}" x2="{px(v):.1f}" y1="{st}" y2="{st + ph}" stroke="rgba(148,163,209,.09)"/>'
    for v in (0.5, 0.6, 0.7, 0.8, 0.9):
        b += f'<line x1="{sl}" x2="{sl + pw}" y1="{py(v):.1f}" y2="{py(v):.1f}" stroke="rgba(148,163,209,.09)"/>'
    for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        b += tx(px(v), st + ph + 16, f"{v:.1f}", 10, "middle", family="JB")
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        b += tx(sl - 8, py(v) + 3.5, f"{v:.1f}", 10, "end", family="JB")
    for f in (0.70, 0.75, 0.80, 0.85):
        pts = " ".join(
            f"{px(ox1 * k / 50):.1f},{py(iso(f, ox1 * k / 50)):.1f}" for k in range(51)
        )
        b += f'<polyline points="{pts}" fill="none" stroke="rgba(167,139,250,.45)" stroke-width="1" stroke-dasharray="3 4" clip-path="url(#pc)"/>'
        o = 0.27
        xa, ya, xb, yb = px(o), py(iso(f, o)), px(o + 0.01), py(iso(f, o + 0.01))
        ang = math.degrees(math.atan2(yb - ya, xb - xa))
        b += f'<text transform="translate({xa:.1f} {ya - 5:.1f}) rotate({ang:.1f})" font-size="9.5" fill="#C4B5FD" font-weight="500" font-family=\'{FAM["JB"]}\'>F1 {f:.2f}</text>'
    b2, r1, r2 = HANDOFF["B2"], HANDOFF["R1"], HANDOFF["R2"]
    b += (
        f'<marker id="m1" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="{MARK["R1"]}"/></marker>'
        f'<marker id="m2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#14B8A6"/></marker>'
    )
    b += f'<path d="M{px(b2["over"]) + 6:.1f} {py(b2["recall"]) - 10:.1f} C {px(0.06):.1f} {py(0.99):.1f}, {px(0.3):.1f} {py(0.995):.1f}, {px(r1["over"]) - 13:.1f} {py(r1["recall"]) - 1:.1f}" stroke="{MARK["R1"]}" stroke-width="1.8" fill="none" stroke-dasharray="6 4" marker-end="url(#m1)"/>'
    b += f'<path d="M{px(b2["over"]) + 10:.1f} {py(b2["recall"]) - 6:.1f} L {px(r2["over"]) - 10:.1f} {py(r2["recall"]) + 8:.1f}" stroke="#14B8A6" stroke-width="2" fill="none" marker-end="url(#m2)"/>'
    place = {
        "B0": (14, 4, "start"),
        "B2": (12, 22, "start"),
        "R1": (-14, 26, "end"),
        "R2": (15, 5, "start"),
    }
    for run, row in HANDOFF.items():
        cx, cy = px(row["over"]), py(row["recall"])
        b += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="15" fill="{MARK[run]}" opacity=".2"/>'
        b += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="url(#g{run})" stroke="#0A0E1A" stroke-width="2"/>'
        dx, dy, anc = place[run]
        b += tx(cx + dx, cy + dy, run, 12.5, anc, "var(--ink)", 700, "JB")
        b += tx(
            cx + dx + (24 if anc == "start" else -24),
            cy + dy,
            f"F1 {row['f1']:.3f}",
            10.5,
            anc,
            "var(--ink2)",
            500,
            "JB",
        )
    b += tx(sl + pw / 2, sh - 6, t["res.b.x"], 10.5, "middle")
    b += f'<text transform="translate(13 {st + ph / 2:.1f}) rotate(-90)" font-size="10.5" text-anchor="middle" fill="var(--ink3)" font-family=\'{FAM["IN"]}\'>{e(t["res.b.y"])}</text>'
    mid = f'<svg class="fit" viewBox="0 0 {sw} {sh}">{b}</svg>'

    # (c) side effects, one strip of three small panels
    vals = [
        {r: _interval(main[r]["Citation rate"])[0] for r in ("B2", "R1", "R2")},
        {r: _interval(main[r]["Violation rate"])[0] for r in ("B2", "R1", "R2")},
        {r: _interval(styles[r]["Worst style"])[0] for r in ("B2", "R1", "R2")},
    ]
    blocks = ""
    for (title, note, better), v in zip(t["res.c.m"], vals, strict=True):
        bars = ""
        for run in ("B2", "R1", "R2"):
            bars += (
                '<div style="display:grid;grid-template-columns:38px 1fr 44px;gap:10px;align-items:center;margin-top:9px">'
                f'<span class="jb" style="font-size:12px;font-weight:700;white-space:nowrap">{swatch(run)}{run}</span>'
                f'<div style="height:10px;border-radius:5px;background:rgba(255,255,255,.07)"><div style="height:10px;width:{v[run] / 0.7 * 100:.1f}%;border-radius:5px;'
                f'background:linear-gradient(90deg,{GRAD[run][0]},{GRAD[run][1]})"></div></div>'
                f'<span class="jb" style="font-size:12px;text-align:right">{v[run]:.3f}</span></div>'
            )
        blocks += (
            '<div class="pn" style="padding:16px 18px">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline"><span class="sg" style="font-size:15px;font-weight:600">{e(title)}</span>'
            f'<span class="jb" style="font-size:10.5px;color:var(--ink3)">{e(better)}</span></div>'
            f'<div style="font-size:11px;color:var(--ink3);margin-top:2px">{e(note)}</div>{bars}</div>'
        )
    return (
        header(t, t["res.no"], t["res.title"], e(t["res.sub"]))
        + '<div style="display:grid;grid-template-columns:1.08fr 1fr;gap:14px">'
        + f'<div class="pn"><div class="pt"><b>{e(t["res.a"])}</b><span>{e(t["res.a.note"])}</span></div>{left}</div>'
        + f'<div class="pn"><div class="pt"><b>{e(t["res.b"])}</b><span>{e(t["res.b.note"])}</span></div>{mid}</div></div>'
        + f'<div style="display:flex;align-items:center;gap:12px;margin:18px 2px 10px"><span class="lbl">{e(t["res.c"])}</span><span style="flex:1;height:1px;background:var(--line)"></span><span class="lbl">{e(t["res.c.note"])}</span></div>'
        + f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px">{blocks}</div>'
    )


def fig_loophole(t: dict) -> str:
    cell = {
        "1.0": "background:linear-gradient(135deg,rgba(52,211,153,.28),rgba(34,211,238,.14));border-color:rgba(52,211,153,.55);color:#A7F3D0",
        "0": "background:rgba(148,163,209,.07);border-color:var(--line);color:var(--ink3)",
        "p": "background:linear-gradient(135deg,rgba(251,191,36,.22),rgba(251,146,60,.10));border-color:rgba(251,191,36,.5);color:#FDE68A",
    }
    rows = ""
    for i, (title, sub, tr, so) in enumerate(t["loop.rows"]):
        hot = i == 1
        tstyle = (
            "background:linear-gradient(135deg,#F97316,#F43F5E);border-color:#FB7185;color:#0A0E1A"
            if hot
            else cell[tr]
        )
        hole = (
            f'<div class="jb" style="position:absolute;top:-9px;right:8px;font-size:9.5px;font-weight:700;letter-spacing:.14em;padding:2px 7px;border-radius:999px;'
            f'background:#0A0E1A;color:#FB7185;border:1px solid #FB7185;text-transform:uppercase">{e(t["loop.hole"])}</div>'
            if hot
            else ""
        )
        italic = "font-style:italic" if so == "p" else ""
        rows += (
            '<div style="display:grid;grid-template-columns:1.6fr .75fr .75fr;gap:8px;margin-top:9px">'
            f'<div style="padding:10px 12px;border-radius:12px;border:1px solid var(--line);background:rgba(255,255,255,.035)"><div style="font-weight:600;font-size:13px">{e(title)}</div><div style="font-size:11px;color:var(--ink3);margin-top:2px">{e(sub)}</div></div>'
            f'<div class="sg" style="position:relative;display:flex;align-items:center;justify-content:center;border-radius:12px;border:1px solid;font-size:28px;font-weight:700;{tstyle}">{hole}{e(tr)}</div>'
            f'<div class="sg" style="display:flex;align-items:center;justify-content:center;border-radius:12px;border:1px solid;font-size:28px;font-weight:700;{cell[so]};{italic}">{e(so)}</div></div>'
        )
    cols = "".join(
        f'<span class="lbl" style="text-align:center">{e(c)}</span>'
        for c in t["loop.cols"]
    )
    a = (
        f'<div class="pn"><div class="pt"><b>{e(t["loop.a"])}</b><span>R<sub>task</sub></span></div>'
        f'<div style="display:grid;grid-template-columns:1.6fr .75fr .75fr;gap:8px"><span></span>{cols}</div>{rows}'
        f'<div style="margin-top:14px;padding:12px 14px;border-radius:12px;background:rgba(244,63,94,.10);border:1px solid rgba(251,113,133,.35);font-size:13px;line-height:1.5">{t["loop.verdict"]}</div></div>'
    )

    def tags(items, color):
        return "".join(
            f'<span class="chip" style="border-color:{color};color:{color};margin:7px 6px 0 0">{e(x)}</span>'
            for x in items
        )

    b = (
        f'<div class="pn"><div class="pt"><b>{e(t["loop.b"])}</b><span>{e(t["loop.b.note"])}</span></div>'
        '<div style="display:flex;justify-content:flex-end"><div style="max-width:88%;padding:10px 13px;border-radius:14px 14px 4px 14px;'
        f'background:linear-gradient(135deg,rgba(59,130,246,.24),rgba(139,92,246,.18));border:1px solid rgba(96,165,250,.35);font-size:12.5px">{e(t["loop.user"])}</div></div>'
        '<div style="margin-top:12px;padding:12px 14px;border-radius:14px 14px 14px 4px;border:1px solid rgba(251,113,133,.35);background:rgba(244,63,94,.07)">'
        f'<div class="sg" style="font-weight:700;font-size:13.5px">{swatch("R1")}{e(t["loop.r1"])}</div>'
        f'<div class="jb" style="font-size:11.5px;margin-top:7px;color:var(--ink)">{e(t["loop.r1.act"])}</div>{tags(t["loop.r1.tag"], "#FB7185")}</div>'
        '<div style="margin-top:10px;padding:12px 14px;border-radius:14px 14px 14px 4px;border:1px solid rgba(45,212,191,.35);background:rgba(20,184,166,.07)">'
        f'<div class="sg" style="font-weight:700;font-size:13.5px">{swatch("R2")}{e(t["loop.r2"])}</div>'
        f'<div style="font-size:12.5px;margin-top:7px;color:var(--ink)">{e(t["loop.r2.say"])}</div>{tags(t["loop.r2.tag"], "#2DD4BF")}</div></div>'
    )
    color = {"0.5": "#FBBF24", "1.0": "#34D399", "−1.0": "#FB7185"}
    fixes = "".join(
        '<div style="display:grid;grid-template-columns:1fr auto 44px;gap:10px;align-items:center;padding:9px 0;border-top:1px solid var(--line)">'
        f'<span style="font-size:12px">{e(lab)}</span><span class="jb" style="font-size:11.5px;color:var(--ink3)">{e(calc)}</span>'
        f'<span class="jb" style="font-size:13px;font-weight:700;text-align:right;color:{color[res]}">{e(res)}</span></div>'
        for lab, calc, res in t["loop.fix"]
    )
    bigs = "".join(
        f'<div style="margin-top:16px"><div class="lbl">{e(lab)}</div><div class="sg" style="font-size:34px;font-weight:700;line-height:1.1;margin-top:5px;white-space:nowrap">'
        + (
            f'<span style="color:#FB7185">{e(x)}</span>'
            if k < 2
            else f'<span style="color:var(--ink2)">{e(x)}</span>'
        )
        + '<span style="color:var(--ink3);font-size:22px;font-weight:500"> → </span>'
        + (
            grad(y, "#22D3EE", "#34D399")
            if k < 2
            else f'<span style="color:var(--ink2)">{e(y)}</span>'
        )
        + "</div></div>"
        for k, (lab, x, y) in enumerate(t["loop.big"])
    )
    c = f'<div class="pn"><div class="pt"><b>{e(t["loop.c"])}</b><span>λ<sub>FP</sub>=0.5 · λ<sub>FN</sub>=1.0</span></div>{fixes}{bigs}</div>'
    return (
        header(t, t["loop.no"], t["loop.title"], e(t["loop.sub"]))
        + f'<div style="display:grid;grid-template-columns:1.18fr 1.12fr .9fr;gap:14px">{a}{b}{c}</div>'
    )


def fig_dynamics(t: dict) -> str:
    rows = _rows("reports/rl/dev_curves.csv")
    w, h, sl, sb, st, sr = 560, 290, 40, 40, 18, 80
    pw, ph = w - sl - sr, h - sb - st

    def chart(column: str, annotate: bool) -> str:
        def sx(v):
            return sl + pw * v / 60

        def sy(v):
            return st + ph * (1 - (v - 0.2) / 0.6)

        out = defs()
        for run in ("R1", "R2"):
            out += f'<linearGradient id="a{run}{column}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{GRAD[run][1]}" stop-opacity=".30"/><stop offset="1" stop-color="{GRAD[run][1]}" stop-opacity="0"/></linearGradient>'
        if annotate:
            out += f'<rect x="{sx(30):.1f}" y="{st}" width="{sx(60) - sx(30):.1f}" height="{ph}" fill="rgba(244,63,94,.07)"/>'
        for v in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            out += f'<line x1="{sl}" x2="{sl + pw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="rgba(148,163,209,.10)"/>'
            out += tx(sl - 8, sy(v) + 3.5, f"{v:.1f}", 10, "end", family="JB")
        for v in range(0, 61, 10):
            out += tx(sx(v), st + ph + 16, str(v), 10, "middle", family="JB")
        out += tx(sl + pw / 2, h - 4, t["dyn.x"], 10.5, "middle")
        if annotate:
            out += tx(sx(31), st + 14, t["dyn.climb"], 10.5, "start", "var(--ink)", 600)
            out += tx(sx(1), st + 14, t["dyn.flat"], 10.5, "start", "var(--ink3)", 500)
        ends = {}
        for run in ("R2", "R1"):
            pts = [(int(r["Step"]), float(r[column])) for r in rows if r["Run"] == run]
            ends[run] = pts[-1][1]
            line = " ".join(
                f"{'M' if i == 0 else 'L'}{sx(s):.1f} {sy(v):.1f}"
                for i, (s, v) in enumerate(pts)
            )
            out += f'<path d="{line} L{sx(60):.1f} {sy(0.2):.1f} L{sx(0):.1f} {sy(0.2):.1f}Z" fill="url(#a{run}{column})" opacity=".55"/>'
            out += f'<path d="{line}" stroke="url(#g{run})" stroke-width="2.6" fill="none" stroke-linejoin="round" stroke-linecap="round"/>'
            for s, v in pts:
                out += f'<circle cx="{sx(s):.1f}" cy="{sy(v):.1f}" r="4.5" fill="#0A0E1A" stroke="{MARK[run]}" stroke-width="2"/>'
        close = abs(ends["R1"] - ends["R2"]) < 0.05
        for run in ("R1", "R2"):
            shift = (-9 if ends[run] >= max(ends.values()) else 9) if close else 0
            y = sy(ends[run]) + shift
            out += f'<rect x="{sx(60) + 11:.1f}" y="{y - 5:.1f}" width="9" height="9" rx="2.5" fill="url(#g{run})"/>'
            out += tx(
                sx(60) + 24,
                y + 4,
                f"{run}  {ends[run]:.2f}",
                11.5,
                "start",
                "var(--ink)",
                700,
                "JB",
            )
        if annotate:
            v40 = [
                float(r[column]) for r in rows if r["Run"] == "R1" and r["Step"] == "40"
            ][0]
            out += f'<circle cx="{sx(40):.1f}" cy="{sy(v40):.1f}" r="11" fill="none" stroke="#FBBF24" stroke-width="1.6" stroke-dasharray="3 3"/>'
        return f'<svg class="fit" viewBox="0 0 {w} {h}">{out}</svg>'

    note = f'<div style="margin-top:8px;font-size:11.5px;color:var(--ink2)"><span style="color:#FBBF24">◌</span>&nbsp; {e(t["dyn.note"])}</div>'
    return (
        header(t, t["dyn.no"], t["dyn.title"], e(t["dyn.sub"]))
        + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">'
        + f'<div class="pn"><div class="pt"><b>{e(t["dyn.a"])}</b></div>{chart("success", False)}<div style="margin-top:8px;font-size:11.5px;color:var(--ink2)">{e(t["dyn.note.a"])}</div></div>'
        + f'<div class="pn"><div class="pt"><b>{e(t["dyn.b"])}</b></div>{chart("transferred", True)}{note}</div></div>'
    )


def waffle(t: dict) -> str:
    """Task mix as a dot grid, one dot per five tasks."""
    colors = ["#22D3EE", "#A78BFA", "#FBBF24"]
    dots, legend, k = "", "", 0
    for (name, n), col in zip(t["pipe.mix"], colors, strict=True):
        for _ in range(round(n / 5)):
            dots += f'<circle cx="{5 + (k % 18) * 10:.1f}" cy="{5 + (k // 18) * 10:.1f}" r="3.4" fill="{col}"/>'
            k += 1
        legend += f'<span style="white-space:nowrap;margin-right:10px"><span class="sw" style="background:{col};border-radius:50%;width:8px;height:8px;margin-right:5px"></span>{e(name)} <span class="jb" style="color:var(--ink)">{n}</span></span>'
    rows = (k + 17) // 18
    return (
        f'<svg class="fit" viewBox="0 0 180 {rows * 10}" style="margin-top:16px">{dots}</svg>'
        f'<div style="font-size:11px;color:var(--ink2);margin-top:10px;line-height:1.7">{legend}</div>'
        f'<div class="jb" style="font-size:10px;color:var(--ink3);margin-top:2px">{e(t["pipe.dot"])}</div>'
    )


def sft_chain(t: dict, c1: str, c2: str) -> str:
    """Teacher -> filter -> SFT data, drawn as a short vertical chain."""
    out = ""
    steps = t["pipe.sft"]
    for k, (head, sub) in enumerate(steps):
        last = k == len(steps) - 1
        out += (
            '<div style="display:grid;grid-template-columns:18px 1fr;gap:10px">'
            '<div style="display:flex;flex-direction:column;align-items:center">'
            f'<span style="width:12px;height:12px;border-radius:50%;margin-top:3px;background:linear-gradient(135deg,{c1},{c2});box-shadow:0 0 0 3px rgba(96,165,250,.15)"></span>'
            + (
                ""
                if last
                else f'<span style="flex:1;width:2px;margin:4px 0;background:linear-gradient(180deg,{c1},rgba(34,211,238,.15))"></span>'
            )
            + f'</div><div style="padding-bottom:{0 if last else 16}px"><div style="font-size:12.5px;font-weight:600">{e(head)}</div>'
            f'<div style="font-size:11.5px;color:var(--ink2);margin-top:2px">{e(sub)}</div></div></div>'
        )
    return f'<div style="padding-top:12px;border-top:1px solid var(--line)">{out}</div>'


def fig_pipeline(t: dict) -> str:
    accents = [
        ("#A78BFA", "#60A5FA"),
        ("#60A5FA", "#22D3EE"),
        ("#22D3EE", "#34D399"),
        ("#FBBF24", "#FB923C"),
        ("#FB7185", "#F0ABFC"),
    ]
    cards = ""
    for idx, ((num, title, lines), (c1, c2)) in enumerate(
        zip(t["pipe.cols"], accents, strict=True)
    ):
        body = ""
        for j, line in enumerate(lines):
            last = j == len(lines) - 1
            if idx in (1, 2) and last:
                body += f'<div style="margin-top:12px"><span class="chip" style="color:#0A0E1A;border:none;background:linear-gradient(90deg,{c1},{c2});font-weight:700">{e(line)}</span></div>'
            elif idx == 3 and j > 0:
                weight = float(re.search(r"\d+\.\d+", line).group())
                body += (
                    f'<div class="jb" style="padding:7px 0 8px;border-top:1px solid var(--line);font-size:11.5px;color:var(--ink)">{e(line)}'
                    f'<div style="height:4px;border-radius:2px;margin-top:5px;width:{weight * 100:.0f}%;background:linear-gradient(90deg,{c1},{c2})"></div></div>'
                )
            else:
                body += f'<div style="padding:7px 0;border-top:1px solid var(--line);font-size:12.5px;color:{"var(--ink)" if j == 0 else "var(--ink2)"}">{e(line)}</div>'
        extra = ""
        if idx == 0:
            extra = waffle(t)
        if idx == 1:
            body = sft_chain(t, c1, c2) + body
        if idx == 2:
            ring = defs()
            cx, cy, r = 115, 92, 50
            pos = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
            colors = ["#A78BFA", "#60A5FA", "#22D3EE", "#FBBF24"]
            ring += '<marker id="ra" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#7F8AA8"/></marker>'
            ring += f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="url(#gSpec)" stroke-width="2" stroke-dasharray="4 5"/>'
            for k in range(4):
                ang = math.radians(-90 + 90 * k + 45)
                x, y = cx + r * math.cos(ang), cy + r * math.sin(ang)
                dxx, dyy = -math.sin(ang) * 4, math.cos(ang) * 4
                ring += f'<path d="M{x - dxx:.1f} {y - dyy:.1f}L{x + dxx:.1f} {y + dyy:.1f}" stroke="#7F8AA8" stroke-width="1.5" marker-end="url(#ra)"/>'
            for k, ((x, y), lab, col) in enumerate(
                zip(pos, t["pipe.loop"], colors, strict=True)
            ):
                ring += f'<circle cx="{x}" cy="{y}" r="8" fill="{col}" stroke="#0A0E1A" stroke-width="3"/>'
                anchor = ["middle", "start", "middle", "end"][k]
                ox, oy = [(0, -15), (13, 4), (0, 25), (-13, 4)][k]
                ring += tx(x + ox, y + oy, lab, 11, anchor, "var(--ink)", 600)
            ring += tx(cx, cy - 2, "8 × 8", 16, "middle", "var(--ink)", 700, "SG")
            ring += tx(cx, cy + 14, "rollouts", 9.5, "middle", "var(--ink3)", 500, "JB")
            extra = f'<svg class="fit" viewBox="0 0 230 184" style="margin-top:10px">{ring}</svg>'
        if idx == 3:
            extra = (
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:12px">'
                f'<span class="chip" style="text-align:center;border-color:#FB7185;color:var(--ink)">{swatch("R1")}{e(t["pipe.r1"])}</span>'
                f'<span class="chip" style="text-align:center;border-color:#2DD4BF;color:var(--ink)">{swatch("R2")}{e(t["pipe.r2"])}</span></div>'
            )
        if idx == 4:
            extra = (
                '<div style="margin-top:12px">'
                + "".join(
                    f'<div style="margin-top:8px"><div style="display:flex;justify-content:space-between"><span class="jb" style="font-size:10.5px;color:var(--ink2)">{e(n)}</span><span class="jb" style="font-size:10.5px">{s}</span></div>'
                    f'<div style="height:7px;border-radius:4px;margin-top:4px;width:{s / 490 * 100:.0f}%;background:linear-gradient(90deg,{c1},{c2})"></div></div>'
                    for n, s in t["pipe.eval"]
                )
                + "</div>"
            )
        cards += (
            '<div class="pn" style="position:relative;padding:18px 16px 18px">'
            f'<div style="position:absolute;left:18px;right:18px;top:0;height:3px;border-radius:0 0 3px 3px;background:linear-gradient(90deg,{c1},{c2})"></div>'
            f'<div style="display:flex;align-items:baseline;gap:10px"><span class="jb" style="font-size:12px;font-weight:700;color:{c1}">{num}</span>'
            f'<span class="sg" style="font-size:21px;font-weight:600">{e(title)}</span></div>'
            f'<div style="margin-top:10px">{body}</div>{extra}</div>'
        )
        if idx < 4:
            cards += f'<div style="display:flex;align-items:center;justify-content:center"><svg width="22" height="22" viewBox="0 0 22 22"><path d="M3 11h14m-5-5 5 5-5 5" stroke="{c2}" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></div>'
    hw = "".join(
        f'<span class="chip" style="margin-right:8px">{e(x)}</span>'
        for x in t["pipe.hw"]
    )
    return (
        header(t, t["pipe.no"], t["pipe.title"], e(t["pipe.sub"]))
        + f'<div style="display:grid;grid-template-columns:1fr 22px 1fr 22px 1.12fr 22px 1fr 22px 1fr;align-items:stretch">{cards}</div>'
        + f'<div style="margin-top:14px;display:flex;align-items:center"><span class="lbl" style="margin-right:12px">{e(t["pipe.infra"])}</span>{hw}</div>'
    )


def fig_styles(t: dict) -> str:
    styles = {r["Run"]: r for r in _rows("reports/main/styles.csv")}
    keys = [
        "neutral",
        "terse",
        "verbose",
        "impatient",
        "non-native",
        "low-tech-literacy",
    ]
    colors = ["#A78BFA", "#60A5FA", "#22D3EE", "#FBBF24", "#34D399", "#FB7185"]
    cols = ""
    for (name, quote), key, col in zip(t["sty"], keys, colors, strict=True):
        bars = ""
        for run in ("B2", "R1", "R2"):
            v = _interval(styles[run][key])[0]
            bars += (
                '<div style="display:grid;grid-template-columns:30px 1fr 32px;gap:6px;align-items:center;margin-top:5px">'
                f'<span class="sg" style="font-size:10.5px;font-weight:700;white-space:nowrap">{swatch(run)}{run}</span>'
                f'<div style="height:7px;border-radius:4px;background:rgba(255,255,255,.08)"><div style="height:7px;border-radius:4px;width:{v / 0.5 * 100:.0f}%;background:linear-gradient(90deg,{GRAD[run][0]},{GRAD[run][1]})"></div></div>'
                f'<span class="jb" style="font-size:10px;text-align:right">{v:.2f}</span></div>'
            )
        cols += (
            '<div style="padding:0 9px">'
            f'<div style="height:3px;border-radius:2px;background:{col};margin-bottom:10px"></div>'
            f'<div class="sg" style="font-size:15px;font-weight:700">{e(name)}</div>'
            f'<div style="font-size:11.5px;color:var(--ink2);margin-top:6px;min-height:34px;font-style:italic">{e(quote)}</div>'
            f'<div style="margin-top:8px">{bars}</div></div>'
        )
    return (
        header(t, t["sty.no"], t["sty.title"], e(t["sty.sub"]))
        + f'<img src="{image_uri("styles-art.jpg")}" style="width:100%;display:block;margin-top:-34px">'
        + f'<div class="pn" style="padding:16px 8px 14px;margin-top:-22px;position:relative"><div style="display:grid;grid-template-columns:repeat(6,1fr)">{cols}</div>'
        + f'<div style="display:flex;justify-content:space-between;gap:20px;margin:14px 9px 0;font-size:10.5px;color:var(--ink3)"><span class="jb">{e(t["sty.eg"])}</span><span>{e(t["sty.foot"])}</span></div></div>'
    )


FIGURES = {
    "hero": fig_hero,
    "results": fig_results,
    "loophole": fig_loophole,
    "dynamics": fig_dynamics,
    "pipeline": fig_pipeline,
    "styles": fig_styles,
}


CANVAS = {
    "hero": "hero",
    "styles": "low",
}  # glow layout; keeps glows off the illustrations


def compact(svg: str) -> str:
    """Round path coordinates from six decimals to two (about half the file size)."""
    return re.sub(
        r' d="[^"]*"', lambda m: re.sub(r"(\d\.\d\d)\d+", r"\1", m.group(0)), svg
    )


def render(names: list[str]) -> list[Path]:
    from playwright.sync_api import sync_playwright

    written: list[Path] = []
    with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": 900})
        for name in names:
            for lang, text in T.items():
                doc = (
                    f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><style>{css()}</style></head>'
                    f'<body><div id="fig" class="{CANVAS.get(name, "")}">{FIGURES[name](text)}</div></body></html>'
                )
                source = Path(tmp) / f"{name}-{lang}.html"
                source.write_text(doc, encoding="utf-8")
                page.goto(source.as_uri())
                page.evaluate("document.fonts.ready")
                page.evaluate(GRAD_TEXT_JS)
                height = int(page.locator("#fig").bounding_box()["height"] + 0.999)
                pdf = Path(tmp) / f"{name}-{lang}.pdf"
                page.pdf(
                    path=str(pdf),
                    width=f"{WIDTH}px",
                    height=f"{height}px",
                    print_background=True,
                    margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
                    page_ranges="1",
                )
                out = ASSETS / f"{name}-{lang}.svg"
                subprocess.run(["pdftocairo", "-svg", str(pdf), str(out)], check=True)
                out.write_text(
                    compact(out.read_text(encoding="utf-8")), encoding="utf-8"
                )
                written.append(out)
                print(out.relative_to(ROOT))
        browser.close()
    npx = shutil.which("npx")
    if npx and written:
        subprocess.run(
            [
                npx,
                "--yes",
                "svgo@3.3.2",
                "--multipass",
                "-p",
                "2",
                "-q",
                *map(str, written),
            ],
            check=True,
        )
    return written


def preview(svgs: list[Path], png_dir: Path) -> None:
    """Rasterise the finished SVGs (not the HTML they came from) for a visual check."""
    from playwright.sync_api import sync_playwright

    png_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": WIDTH, "height": 900}, device_scale_factor=2
        )
        viewer = Path(tmp) / "view.html"
        for svg in svgs:
            viewer.write_text(
                f'<body style="margin:0"><img id="svg" src="{svg.as_uri()}" style="width:{WIDTH}px;display:block"></body>',
                encoding="utf-8",
            )
            page.goto(viewer.as_uri())
            page.wait_for_function("document.getElementById('svg').complete")
            page.locator("#svg").screenshot(
                path=str(png_dir / f"{svg.stem}.png"), omit_background=True
            )
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=list(FIGURES))
    parser.add_argument(
        "--png", default=None, help="also write PNG previews into this folder"
    )
    args = parser.parse_args()
    svgs = render(args.only)
    if args.png:
        preview(svgs, Path(args.png))


if __name__ == "__main__":
    main()
