"""Render trajectories as one self-contained HTML page for slides and the report.

Hand-offs are highlighted, policy citations in refusals and hand-offs are
marked, and rule violations appear next to the turn that caused them. The page
has no external assets, so it can be opened offline or embedded in slides.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence

from handoff.eval.citations import CITATION_PATTERN
from handoff.rollout.schema import TRANSFER_TOOL, Trajectory

_CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1f;--muted:#6b6b70;--user:#e8f0fe;
--agent:#f1f3f4;--tool:#fafafa;--handoff:#fff4e5;--handoff-ink:#b45309;
--viol:#fde8e8;--viol-ink:#b91c1c;--cite:#fef9c3}
@media (prefers-color-scheme:dark){:root{--bg:#121214;--card:#1c1c1f;--ink:#ececf1;
--muted:#9a9aa3;--user:#1e2a44;--agent:#26262b;--tool:#202024;--handoff:#3a2a12;
--handoff-ink:#fbbf24;--viol:#3b1515;--viol-ink:#f87171;--cite:#4a4214}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 24px}
section{background:var(--card);border-radius:12px;padding:16px;margin:0 0 24px;
box-shadow:0 1px 3px rgba(0,0,0,.08)}
h2{font-size:16px;margin:0 0 8px}
.meta{color:var(--muted);font-size:13px;margin-bottom:12px}
.msg{border-radius:10px;padding:8px 12px;margin:6px 0;white-space:pre-wrap;
overflow-wrap:anywhere}.user{background:var(--user);margin-left:15%}
.assistant{background:var(--agent);margin-right:15%}.tool{background:var(--tool);
font:12px/1.4 ui-monospace,Menlo,Consolas,monospace;color:var(--muted);margin-right:15%}
.role{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.call{font:13px/1.4 ui-monospace,Menlo,Consolas,monospace}
.handoff{background:var(--handoff);border-left:4px solid var(--handoff-ink)}
.badge{display:inline-block;border-radius:6px;padding:0 6px;font-size:12px;
margin-right:6px}
.b-handoff{background:var(--handoff);color:var(--handoff-ink)}
.b-viol{background:var(--viol);color:var(--viol-ink)}
mark{background:var(--cite);color:inherit}
"""


def _highlight(text: str) -> str:
    escaped = html.escape(text)
    return CITATION_PATTERN.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)


def _section(trajectory: Trajectory) -> str:
    violations: dict[int, list[str]] = {}
    for violation in trajectory.violations:
        violations.setdefault(violation.turn, []).append(violation.rule)
    outcome = "success" if trajectory.success else "failure"
    parts = [
        "<section>",
        f"<h2>{html.escape(trajectory.domain)} · task {html.escape(trajectory.task_id)}"
        f" · {html.escape(trajectory.persona)}</h2>",
        f'<div class="meta">{html.escape(trajectory.kind)} · should transfer: '
        f"{trajectory.should_transfer} · transferred: {trajectory.transferred} · "
        f"{outcome} (reward {trajectory.task_reward:g}) · "
        f"{html.escape(trajectory.policy_ckpt)}</div>",
    ]
    for message in trajectory.messages:
        role = str(message.get("role"))
        turn = int(message.get("turn", 0))
        classes = ["msg", role]
        badges = []
        body = ""
        calls = message.get("tool_calls") or []
        if any(call.get("name") == TRANSFER_TOOL for call in calls):
            classes.append("handoff")
            badges.append('<span class="badge b-handoff">hand-off</span>')
        for rule in violations.get(turn, []):
            badges.append(f'<span class="badge b-viol">{html.escape(rule)}</span>')
        text = message.get("content") or ""
        if role == "tool":
            body = html.escape(str(text)[:600])
        else:
            body = _highlight(str(text)) if text else ""
            for call in calls:
                arguments = json.dumps(call.get("arguments"), ensure_ascii=False)
                body += (
                    f'<div class="call">{html.escape(str(call.get("name")))}'
                    f"({html.escape(arguments)})</div>"
                )
        parts.append(
            f'<div class="{" ".join(classes)}"><div class="role">{role} · {turn}'
            f"</div>{''.join(badges)}{body}</div>"
        )
    parts.append("</section>")
    return "\n".join(parts)


def render_html(trajectories: Sequence[Trajectory], title: str) -> str:
    """Return a complete HTML document for the given trajectories."""
    sections = "\n".join(_section(t) for t in trajectories)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body><main>"
        f"<h1>{html.escape(title)}</h1><p class='sub'>Hand-offs are outlined; "
        "policy citations are highlighted; red badges mark rule violations.</p>"
        f"{sections}</main></body></html>\n"
    )
