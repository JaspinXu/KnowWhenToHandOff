# ruff: noqa: E501  (HTML/SVG templates and figure text are kept on one line)
"""Render the README figures: docs/assets/<name>-<lang>.png.

Four figures (hero, results, loophole, method), each in English and Chinese.
Every figure is an HTML/SVG page screenshotted by Chromium at 2x on a light,
softly tinted canvas with transparent rounded corners. Numbers come from
committed report tables (``reports/main/*.csv``, ``reports/rl/dev_curves.csv``)
or, for the test_handoff precision/recall, from the runs' ``metrics.json`` values
in ``HANDOFF`` below (also in the model cards). Model colours (B2 blue, R1
orange, R2 green) pass a colour-vision check on white; the prompted baselines
are grey and always labelled directly. The hero illustration
(``hero-art.jpg``) was made with an image model. Fonts: ``docs/assets/fonts``
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
        "m": [
            (
                "Tasks",
                [
                    "τ²-bench airline + retail",
                    "+ 461 hand-off variants",
                    "6 user styles",
                ],
                None,
            ),
            (
                "SFT",
                [
                    "teacher: Qwen3-30B",
                    "keep 790 clean dialogues",
                    "LoRA fine-tune Qwen3-4B",
                ],
                ["B2"],
            ),
            (
                "Multi-turn GRPO",
                [
                    "policy ⇄ simulated user",
                    "8 tasks × 8 rollouts × 60 steps",
                    "R1: task reward only",
                    "R2: + penalty terms",
                ],
                ["R1", "R2"],
            ),
            (
                "Evaluate",
                [
                    "frozen checkpoints",
                    "1,066 test dialogues",
                    "paired bootstrap vs SFT",
                ],
                None,
            ),
        ],
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
        "m": [
            (
                "任务",
                ["τ²-bench 航空 + 零售", "+ 461 个转人工变体", "6 种用户沟通风格"],
                None,
            ),
            (
                "SFT",
                ["教师：Qwen3-30B", "保留 790 段干净对话", "LoRA 微调 Qwen3-4B"],
                ["B2"],
            ),
            (
                "多轮 GRPO",
                [
                    "策略 ⇄ 模拟用户",
                    "8 任务 × 8 rollouts × 60 步",
                    "R1：只用任务奖励",
                    "R2：再加惩罚项",
                ],
                ["R1", "R2"],
            ),
            ("评测", ["冻结检查点", "1,066 段测试对话", "与 SFT 配对自助法比较"], None),
        ],
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
        f'<div class="card">{head(t["res.b"], t["res.b.sub"])}<div style="margin-top:14px">{mid}</div></div></div>'
    )


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


def fig_method(t: dict) -> str:
    look = [
        ("tasks", "#7C3AED", "#C084FC"),
        ("sft", "#2563EB", "#60A5FA"),
        ("rl", "#EA580C", "#FB7185"),
        ("eval", "#059669", "#34D399"),
    ]
    boxes = ""
    for k, ((title, lines, chips), (ic, c1, c2)) in enumerate(
        zip(t["m"], look, strict=True)
    ):
        body = "".join(
            f'<div style="font-size:13px;color:var(--ink2);margin-top:6px">{e(x)}</div>'
            for x in lines
        )
        tag = ""
        if chips:
            tag = (
                '<div style="margin-top:12px;display:flex;gap:6px">'
                + "".join(
                    f'<span class="pill jb" style="font-size:12px;color:#fff;background:linear-gradient(90deg,{MODEL[c][0]},{MODEL[c][1]})">{c}</span>'
                    for c in chips
                )
                + "</div>"
            )
        boxes += (
            f'<div class="card" style="padding:16px 14px 18px">'
            f'<div style="display:flex;align-items:center;gap:10px">{icon(ic, c1, c2)}<span class="sg" style="font-size:17px;font-weight:600">{e(title)}</span></div>'
            f'<div style="margin-top:8px">{body}</div>{tag}</div>'
        )
        if k < 3:
            boxes += (
                '<div style="display:flex;align-items:center;justify-content:center"><svg width="20" height="20" viewBox="0 0 20 20">'
                f'<path d="M3 10h12m-4.5-4.5L15 10l-4.5 4.5" stroke="{c2}" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></div>'
            )
    return f'<div style="display:grid;grid-template-columns:1fr 16px 1fr 16px 1.1fr 16px 1fr;gap:0 4px;align-items:stretch">{boxes}</div>'


FIGURES = {
    "hero": fig_hero,
    "results": fig_results,
    "loophole": fig_loophole,
    "method": fig_method,
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
