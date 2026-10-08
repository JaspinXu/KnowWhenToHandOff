# ruff: noqa: E501  (HTML/SVG templates and figure text are kept on one line)
"""Render the README figures: docs/assets/<name>-<lang>-<theme>.png.

Four figures (hero, results, loophole, method), each in English and Chinese and
in a light and a dark variant whose backgrounds match GitHub's (#ffffff and
#0d1117), so the README can switch them with <picture>. Every figure is an
HTML/SVG page screenshotted by Chromium at 2x. Numbers come from committed
report tables (``reports/main/*.csv``, ``reports/rl/dev_curves.csv``) or, for the
test_handoff precision/recall, from the runs' ``metrics.json`` values in
``HANDOFF`` below (also in the model cards). Model colours: the colour-blind-safe
categorical slots of the dataviz reference palette for B2/R1/R2 (validated on
both backgrounds), neutral grey for the prompted baselines, which are always
labelled directly. The hero illustrations (``hero-art-<theme>.jpg``) were made
with an image model. Fonts: ``docs/assets/fonts`` (Inter, JetBrains Mono, Space
Grotesk; OFL) plus Noto CJK for Chinese.

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

THEMES = {
    "light": {
        "bg": "#ffffff",
        "ink": "#1f2328",
        "ink2": "#59636e",
        "ink3": "#818b98",
        "line": "#d1d9e0",
        "faint": "#f6f8fa",
        "B0": "#8c959f",
        "B1": "#59636e",
        "B2": "#2a78d6",
        "R1": "#eb6834",
        "R2": "#1baf7a",
        "t-B2": "#1f5fae",
        "t-R1": "#b94a1b",
        "t-R2": "#13855c",
        "t-violet": "#6639ba",
        "hot": "#fde3d6",
        "hot-ink": "#9a3412",
        "ok": "#dcf5e9",
        "ok-ink": "#13855c",
        "p": "#fff4cf",
        "p-ink": "#8a5a00",
    },
    "dark": {
        "bg": "#0d1117",
        "ink": "#e6edf3",
        "ink2": "#9198a1",
        "ink3": "#6e7681",
        "line": "#30363d",
        "faint": "#151b23",
        "B0": "#6e7681",
        "B1": "#9198a1",
        "B2": "#3987e5",
        "R1": "#d95926",
        "R2": "#199e70",
        "t-B2": "#6ea8f0",
        "t-R1": "#f08a5d",
        "t-R2": "#3fc996",
        "t-violet": "#b19cf2",
        "hot": "#4a2212",
        "hot-ink": "#ffb48f",
        "ok": "#0f3326",
        "ok-ink": "#6ee0b0",
        "p": "#3d2e05",
        "p-ink": "#f5cf6b",
    },
}

T: dict[str, dict] = {
    "en": {
        "names": {
            "B0": "prompted 4B",
            "B1": "prompted 30B",
            "B2": "SFT",
            "R1": "GRPO, task reward",
            "R2": "GRPO, RAI reward",
        },
        "kpi": [
            ("0.812", "hand-off F1", "R2, best of all models"),
            ("46% → 10%", "over-escalation", "R1 (task reward) → R2"),
            ("+0.015", "task success vs SFT", "R2, not significant"),
            ("461", "hand-off tasks", "new, with should-transfer labels"),
        ],
        "res.a": "Task success",
        "res.a.sub": "test_main · 49 tasks × 4 trials · 95% CI",
        "res.a.cols": ["pass¹", "Δ vs SFT"],
        "res.b": "Hand-off operating points",
        "res.b.sub": "test_handoff · 380 dialogues · dashed: equal F1",
        "res.b.x": "over-escalation rate",
        "res.b.y": "hand-off recall",
        "loop.a": "What the task reward pays",
        "loop.a.sub": "τ²-bench checks only the final database",
        "loop.cols": ["transfer", "try to solve"],
        "loop.rows": [
            ("User needs a human", "1", "0"),
            ("Correct outcome is a refusal", "1", "1"),
            ("Task needs a database write", "0", "p"),
        ],
        "loop.hole": "loophole",
        "loop.note": "p: chance the attempt succeeds. When unsure, a transfer never scores lower.",
        "loop.b": "Transfer rate during GRPO",
        "loop.b.sub": "dev set, greedy, every 10 steps",
        "loop.x": "GRPO step",
        "loop.climb": "R1 learns the loophole",
        "m": [
            (
                "Tasks",
                [
                    "τ²-bench airline + retail",
                    "+ 461 hand-off variants",
                    "6 user communication styles",
                ],
                None,
            ),
            (
                "SFT",
                [
                    "teacher: Qwen3-30B",
                    "keep 790 clean ones",
                    "LoRA fine-tune Qwen3-4B",
                ],
                ["B2"],
            ),
            (
                "Multi-turn GRPO",
                [
                    "policy ⇄ simulated user ⇄ tools",
                    "8 tasks × 8 rollouts, 60 steps",
                    "R1: task reward only",
                    "R2: + violation, hand-off penalties",
                ],
                ["R1", "R2"],
            ),
            (
                "Evaluate",
                [
                    "each checkpoint tested once",
                    "3 test sets, 1,066 dialogues",
                    "paired bootstrap vs SFT",
                ],
                None,
            ),
        ],
    },
    "zh": {
        "names": {
            "B0": "提示词 4B",
            "B1": "提示词 30B",
            "B2": "SFT",
            "R1": "GRPO，任务奖励",
            "R2": "GRPO，RAI 奖励",
        },
        "kpi": [
            ("0.812", "转人工 F1", "R2，所有模型中最高"),
            ("46% → 10%", "过度转人工", "R1（任务奖励）→ R2"),
            ("+0.015", "相对 SFT 的成功率", "R2，不显著"),
            ("461", "转人工任务", "新增，带是否应转标签"),
        ],
        "res.a": "任务成功率",
        "res.a.sub": "test_main · 49 个任务 × 4 次 · 95% 区间",
        "res.a.cols": ["pass¹", "相对 SFT"],
        "res.b": "转人工的工作点",
        "res.b.sub": "test_handoff · 380 段对话 · 虚线：F1 相等",
        "res.b.x": "过度转人工率",
        "res.b.y": "转人工召回率",
        "loop.a": "任务奖励给多少分",
        "loop.a.sub": "τ²-bench 只检查最终数据库",
        "loop.cols": ["转人工", "尝试解决"],
        "loop.rows": [
            ("用户需要人工", "1", "0"),
            ("正确结果是拒绝", "1", "1"),
            ("任务需要写数据库", "0", "p"),
        ],
        "loop.hole": "漏洞",
        "loop.note": "p：尝试解决成功的概率。拿不准时，转人工的得分永远不会更低。",
        "loop.b": "GRPO 训练中的转人工率",
        "loop.b.sub": "dev 集，贪心解码，每 10 步",
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
                [
                    "Qwen3-30B 教师跑训练任务",
                    "保留 790 段干净对话",
                    "LoRA 微调 Qwen3-4B",
                ],
                ["B2"],
            ),
            (
                "多轮 GRPO",
                [
                    "策略 ⇄ 模拟用户 ⇄ 工具",
                    "每步 8 任务 × 8 rollouts，60 步",
                    "R1：只用任务奖励",
                    "R2：再加违规和转人工惩罚",
                ],
                ["R1", "R2"],
            ),
            (
                "评测",
                [
                    "每个检查点只测一次",
                    "3 个测试集，共 1,066 段对话",
                    "与 SFT 配对自助法比较",
                ],
                None,
            ),
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


def css(theme: str) -> str:
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
    tokens = ";".join(f"--{k}:{v}" for k, v in THEMES[theme].items())
    return (
        faces
        + f"""
:root{{{tokens}}}
*{{box-sizing:border-box}}
html,body{{margin:0;background:var(--bg)}}
#fig{{width:{WIDTH}px;padding:10px 6px;background:var(--bg);color:var(--ink);font-family:IN,"Noto Sans CJK SC",sans-serif;font-size:14px;line-height:1.45}}
.jb{{font-family:JB,"Noto Sans CJK SC",monospace}}
.sg{{font-family:SG,IN,"Noto Sans CJK SC",sans-serif}}
.ttl{{font-weight:600;font-size:16px}}
.sub{{color:var(--ink3);font-size:13px;margin-top:2px}}
svg.fit{{width:100%;height:auto;display:block}}
"""
    )


FAM = {"IN": 'IN,"Noto Sans CJK SC"', "JB": 'JB,"Noto Sans CJK SC"'}


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


def jpeg_uri(name: str) -> str:
    return (
        "data:image/jpeg;base64,"
        + base64.b64encode((ASSETS / name).read_bytes()).decode()
    )


def head(title: str, sub: str) -> str:
    return f'<div class="ttl">{e(title)}</div><div class="sub">{e(sub)}</div>'


# --------------------------------------------------------------------------- figures


def fig_hero(t: dict, theme: str) -> str:
    colors = ["var(--t-R2)", "var(--t-R1)", "var(--t-B2)", "var(--t-violet)"]
    kpis = "".join(
        f'<div style="padding:18px 0 18px 18px;border-left:3px solid {c}">'
        f'<div class="sg" style="font-size:38px;font-weight:700;line-height:1;color:{c};white-space:nowrap">{e(v)}</div>'
        f'<div style="font-size:15px;font-weight:600;margin-top:10px">{e(k)}</div><div style="font-size:13px;color:var(--ink2);margin-top:2px">{e(n)}</div></div>'
        for (v, k, n), c in zip(t["kpi"], colors, strict=True)
    )
    return (
        '<div style="display:grid;grid-template-columns:1fr 470px;gap:24px;align-items:center">'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:26px 22px;padding-left:8px">{kpis}</div>'
        f'<img src="{jpeg_uri(f"hero-art-{theme}.jpg")}" style="width:470px;display:block"></div>'
    )


def fig_results(t: dict, theme: str) -> str:
    main = {r["Run"]: r for r in _rows("reports/main/main_table.csv")}
    paired = {r["Run"]: r for r in _rows("reports/main/paired_differences.csv")}
    order = ["B0", "B1", "B2", "R1", "R2"]

    # (a) forest plot
    w, rowh, top, x0, x1 = 450, 50, 30, 150, 312
    h = top + rowh * len(order) + 26

    def sx(v):
        return x0 + (x1 - x0) * v / 0.6

    a = ""
    for v in (0, 0.2, 0.4, 0.6):
        a += f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{top - 6}" y2="{h - 24}" stroke="var(--line)"/>'
        a += tx(sx(v), h - 6, f"{v:.1f}", 12, "middle", family="JB")
    ref = _interval(main["B2"]["Success (pass^1)"])[0]
    a += f'<line x1="{sx(ref):.1f}" x2="{sx(ref):.1f}" y1="{top - 6}" y2="{h - 24}" stroke="var(--B2)" stroke-dasharray="3 3"/>'
    a += tx(372, 14, t["res.a.cols"][0], 12, "end", weight=500, family="JB")
    a += tx(w, 14, t["res.a.cols"][1], 12, "end", weight=500, family="JB")
    for i, run in enumerate(order):
        y = top + rowh * i + rowh / 2
        p, lo, hi = _interval(main[run]["Success (pass^1)"])
        col = f"var(--{run})"
        base = run in ("B0", "B1")
        a += tx(0, y - 3, run, 14, "start", "var(--ink)", 700, "JB")
        a += tx(0, y + 14, t["names"][run], 12.5, "start", "var(--ink2)")
        a += f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{col}" stroke-width="3" stroke-linecap="round" opacity=".55"/>'
        fill = "var(--bg)" if base else col
        a += f'<circle cx="{sx(p):.1f}" cy="{y:.1f}" r="6.5" fill="{fill}" stroke="{col}" stroke-width="2.5"/>'
        a += tx(372, y + 4.5, f"{p:.3f}", 13.5, "end", "var(--ink)", 500, "JB")
        if run in paired:
            d = _interval(paired[run]["Δ success vs B2 [95% CI]"])[0]
            a += tx(
                w,
                y + 4.5,
                f"{d:+.3f}".replace("-", "−"),
                13.5,
                "end",
                "var(--ink2)",
                400,
                "JB",
            )
        else:
            a += tx(w, y + 4.5, "—", 13.5, "end", "var(--ink3)", 400, "JB")
    left = f'<svg class="fit" viewBox="0 0 {w} {h}">{a}</svg>'

    # (b) operating points with iso-F1 curves
    sw, sh, sl, sb, st, sr = 430, h, 44, 40, 6, 16
    pw, ph = sw - sl - sr, sh - sb - st
    ox1, oy0, oy1 = 0.5, 0.45, 1.0

    def px(v):
        return sl + pw * v / ox1

    def py(v):
        return st + ph * (1 - (v - oy0) / (oy1 - oy0))

    def iso(f, o):
        return f * (NEGATIVES * o + POSITIVES) / (2 * POSITIVES - POSITIVES * f)

    b = f'<clipPath id="pc"><rect x="{sl}" y="{st}" width="{pw}" height="{ph}"/></clipPath>'
    b += f'<rect x="{sl}" y="{st}" width="{pw}" height="{ph}" fill="none" stroke="var(--line)"/>'
    for v in (0.1, 0.2, 0.3, 0.4):
        b += f'<line x1="{px(v):.1f}" x2="{px(v):.1f}" y1="{st}" y2="{st + ph}" stroke="var(--line)" opacity=".6"/>'
    for v in (0.6, 0.7, 0.8, 0.9):
        b += f'<line x1="{sl}" x2="{sl + pw}" y1="{py(v):.1f}" y2="{py(v):.1f}" stroke="var(--line)" opacity=".6"/>'
    for v in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        b += tx(px(v), st + ph + 17, f"{v:.1f}", 12, "middle", family="JB")
    for v in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        b += tx(sl - 7, py(v) + 4, f"{v:.1f}", 12, "end", family="JB")
    for f in (0.70, 0.75, 0.80, 0.85):
        pts = " ".join(
            f"{px(ox1 * k / 50):.1f},{py(iso(f, ox1 * k / 50)):.1f}" for k in range(51)
        )
        b += f'<polyline points="{pts}" fill="none" stroke="var(--ink3)" stroke-width="1" stroke-dasharray="3 4" clip-path="url(#pc)" opacity=".8"/>'
        o = {0.85: 0.2, 0.80: 0.2, 0.75: 0.34, 0.70: 0.34}[f]
        xa, ya, xb, yb = px(o), py(iso(f, o)), px(o + 0.01), py(iso(f, o + 0.01))
        ang = math.degrees(math.atan2(yb - ya, xb - xa))
        b += f'<text transform="translate({xa:.1f} {ya - 5:.1f}) rotate({ang:.1f})" font-size="11.5" fill="var(--ink3)" font-family=\'{FAM["JB"]}\'>F1 {f:.2f}</text>'
    b2, r1, r2 = HANDOFF["B2"], HANDOFF["R1"], HANDOFF["R2"]
    for run in ("R1", "R2"):
        b += f'<marker id="m{run}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--{run})"/></marker>'
    b += f'<path d="M{px(b2["over"]) + 5:.1f} {py(b2["recall"]) - 9:.1f} C {px(0.05):.1f} {py(0.99):.1f}, {px(0.3):.1f} {py(0.995):.1f}, {px(r1["over"]) - 11:.1f} {py(r1["recall"]) - 1:.1f}" stroke="var(--R1)" stroke-width="1.8" fill="none" stroke-dasharray="5 4" marker-end="url(#mR1)"/>'
    b += f'<path d="M{px(b2["over"]) + 8:.1f} {py(b2["recall"]) - 5:.1f} L {px(r2["over"]) - 8:.1f} {py(r2["recall"]) + 7:.1f}" stroke="var(--R2)" stroke-width="2" fill="none" marker-end="url(#mR2)"/>'
    place = {
        "B0": (12, 5, "start"),
        "B2": (10, 21, "start"),
        "R1": (-12, 24, "end"),
        "R2": (12, 5, "start"),
    }
    for run, row in HANDOFF.items():
        cx, cy = px(row["over"]), py(row["recall"])
        col = f"var(--{run})"
        fill = "var(--bg)" if run == "B0" else col
        b += f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{fill}" stroke="{col}" stroke-width="2.5"/>'
        dx, dy, anc = place[run]
        b += (
            f'<text x="{cx + dx:.1f}" y="{cy + dy:.1f}" font-size="13" text-anchor="{anc}" fill="var(--ink)" font-family=\'{FAM["JB"]}\' stroke="var(--bg)" stroke-width="5" stroke-linejoin="round" paint-order="stroke">'
            f'<tspan font-weight="700">{run}</tspan><tspan fill="var(--ink2)"> F1 {row["f1"]:.3f}</tspan></text>'
        )
    b += tx(sl + pw / 2, sh - 4, t["res.b.x"], 13, "middle", "var(--ink2)")
    b += f'<text transform="translate(11 {st + ph / 2:.1f}) rotate(-90)" font-size="13" text-anchor="middle" fill="var(--ink2)" font-family=\'{FAM["IN"]}\'>{e(t["res.b.y"])}</text>'
    mid = f'<svg class="fit" viewBox="0 0 {sw} {sh}">{b}</svg>'
    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:44px">'
        f'<div>{head(t["res.a"], t["res.a.sub"])}<div style="margin-top:14px">{left}</div></div>'
        f'<div>{head(t["res.b"], t["res.b.sub"])}<div style="margin-top:14px">{mid}</div></div></div>'
    )


def fig_loophole(t: dict, theme: str) -> str:
    style = {
        "1": "background:var(--ok);color:var(--ok-ink)",
        "0": "background:var(--faint);color:var(--ink3)",
        "p": "background:var(--p);color:var(--p-ink);font-style:italic",
    }
    cells = "<span></span>" + "".join(
        f'<span style="text-align:center;font-size:13px;color:var(--ink2);font-weight:500">{e(c)}</span>'
        for c in t["loop.cols"]
    )
    for i, (label, tr, so) in enumerate(t["loop.rows"]):
        hot = i == 1
        tstyle = (
            "background:var(--hot);color:var(--hot-ink);box-shadow:inset 0 0 0 2px var(--R1)"
            if hot
            else style[tr]
        )
        badge = (
            f'<span style="position:absolute;top:-10px;left:50%;transform:translateX(-50%);font-size:11px;font-weight:600;padding:1px 8px;border-radius:999px;'
            f'background:var(--R1);color:#fff;white-space:nowrap">{e(t["loop.hole"])}</span>'
            if hot
            else ""
        )
        cells += (
            f'<span style="display:flex;align-items:center;font-size:14px;font-weight:500">{e(label)}</span>'
            f'<span class="sg" style="position:relative;display:flex;align-items:center;justify-content:center;height:52px;border-radius:10px;font-size:24px;font-weight:700;{tstyle}">{badge}{e(tr)}</span>'
            f'<span class="sg" style="display:flex;align-items:center;justify-content:center;height:52px;border-radius:10px;font-size:24px;font-weight:700;{style[so]}">{e(so)}</span>'
        )
    table = (
        f'<div style="display:grid;grid-template-columns:1fr 92px 92px;gap:12px 10px;margin-top:16px">{cells}</div>'
        f'<div style="font-size:13px;color:var(--ink2);margin-top:16px">{e(t["loop.note"])}</div>'
    )

    rows = _rows("reports/rl/dev_curves.csv")
    w, h, sl, sb, st, sr = 440, 262, 36, 38, 10, 66
    pw, ph = w - sl - sr, h - sb - st

    def sx(v):
        return sl + pw * v / 60

    def sy(v):
        return st + ph * (1 - (v - 0.2) / 0.6)

    c = f'<rect x="{sx(30):.1f}" y="{st}" width="{sx(60) - sx(30):.1f}" height="{ph}" fill="var(--faint)"/>'
    for v in (0.2, 0.4, 0.6, 0.8):
        c += f'<line x1="{sl}" x2="{sl + pw}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="var(--line)" opacity=".7"/>'
        c += tx(sl - 7, sy(v) + 4, f"{v:.1f}", 12, "end", family="JB")
    for v in range(0, 61, 10):
        c += tx(sx(v), st + ph + 17, str(v), 12, "middle", family="JB")
    c += tx(sl + pw / 2, h - 3, t["loop.x"], 13, "middle", "var(--ink2)")
    c += tx(sx(31.5), st + 16, t["loop.climb"], 13, "start", "var(--t-R1)", 600)
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
        c += f'<path d="{line}" stroke="var(--{run})" stroke-width="2.5" fill="none" stroke-linejoin="round"/>'
        for s, v in pts:
            c += f'<circle cx="{sx(s):.1f}" cy="{sy(v):.1f}" r="4" fill="var(--bg)" stroke="var(--{run})" stroke-width="2"/>'
    for run in ("R1", "R2"):
        c += tx(
            sx(60) + 10,
            sy(ends[run]) + 4.5,
            f"{run} {ends[run]:.2f}",
            13,
            "start",
            f"var(--t-{run})",
            700,
            "JB",
        )
    chart = f'<svg class="fit" viewBox="0 0 {w} {h}" style="margin-top:14px">{c}</svg>'
    return (
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:44px">'
        f"<div>{head(t['loop.a'], t['loop.a.sub'])}{table}</div>"
        f"<div>{head(t['loop.b'], t['loop.b.sub'])}{chart}</div></div>"
    )


def fig_method(t: dict, theme: str) -> str:
    accents = ["var(--t-violet)", "var(--B2)", "var(--R1)", "var(--R2)"]
    boxes = ""
    for k, ((title, lines, chips), acc) in enumerate(zip(t["m"], accents, strict=True)):
        body = "".join(
            f'<div style="font-size:13px;color:var(--ink2);margin-top:6px">{e(x)}</div>'
            for x in lines
        )
        tag = ""
        if chips:
            tag = (
                '<div style="margin-top:12px;display:flex;gap:6px">'
                + "".join(
                    f'<span class="jb" style="font-size:12px;font-weight:700;padding:2px 9px;border-radius:999px;color:#fff;background:var(--{c})">{c}</span>'
                    for c in chips
                )
                + "</div>"
            )
        boxes += (
            f'<div style="padding:14px 14px 16px;border-radius:12px;background:var(--faint);border-top:3px solid {acc}">'
            f'<div style="display:flex;align-items:baseline;gap:8px"><span class="jb" style="font-size:12px;font-weight:700;color:{acc}">0{k + 1}</span>'
            f'<span style="font-size:16px;font-weight:600">{e(title)}</span></div>{body}{tag}</div>'
        )
        if k < 3:
            boxes += '<div style="display:flex;align-items:center;justify-content:center"><svg width="18" height="18" viewBox="0 0 18 18"><path d="M2 9h12m-4-4 4 4-4 4" stroke="var(--ink3)" stroke-width="1.8" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></div>'
    return f'<div style="display:grid;grid-template-columns:1fr 18px 1fr 18px 1.15fr 18px 1fr;gap:0 6px;align-items:stretch">{boxes}</div>'


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
            viewport={"width": WIDTH, "height": 800}, device_scale_factor=2
        )
        for name in names:
            for lang, text in T.items():
                for theme in THEMES:
                    doc = (
                        f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><style>{css(theme)}</style></head>'
                        f'<body><div id="fig">{FIGURES[name](text, theme)}</div></body></html>'
                    )
                    source = Path(tmp) / "fig.html"
                    source.write_text(doc, encoding="utf-8")
                    page.goto(source.as_uri())
                    page.evaluate("document.fonts.ready")
                    out = ASSETS / f"{name}-{lang}-{theme}.png"
                    page.locator("#fig").screenshot(path=str(out))
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
