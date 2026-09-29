"""Small queries about Slurm jobs (``squeue``), used by the serving helpers.

Every function returns ``None`` when Slurm is not available or the answer is
unknown, so the callers work the same on a laptop.
"""

from __future__ import annotations

import shutil
import subprocess


def _squeue(job_id: str, field: str) -> subprocess.CompletedProcess[str] | None:
    if not job_id or shutil.which("squeue") is None:
        return None
    try:
        return subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", field],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def job_alive(job_id: str) -> bool | None:
    """Whether a Slurm job is still pending or running (``None``: cannot tell)."""
    result = _squeue(job_id, "%T")
    if result is None:
        return None
    # squeue fails with "Invalid job id specified" once a job has ended.
    return result.returncode == 0 and bool(result.stdout.strip())


def parse_duration(text: str) -> float | None:
    """Seconds in a Slurm time string (``D-HH:MM:SS``, ``HH:MM:SS``, ``MM:SS``).

    ``None`` for ``UNLIMITED``, ``NOT_SET`` and anything unparsable.
    """
    text = text.strip()
    days: int | None = None
    if "-" in text:
        head, text = text.split("-", 1)
        if not head.isdigit():
            return None
        days = int(head)
    parts = text.split(":")
    if not 1 <= len(parts) <= 3 or not all(part.isdigit() for part in parts):
        return None
    values = [int(part) for part in parts]
    if days is not None:  # D-HH, D-HH:MM, D-HH:MM:SS
        hours, minutes, seconds = (values + [0, 0])[:3]
    elif len(values) == 3:  # HH:MM:SS
        hours, minutes, seconds = values
    else:  # MM or MM:SS
        hours, minutes, seconds = 0, values[0], (values + [0])[1]
    return float((days or 0) * 86400 + hours * 3600 + minutes * 60 + seconds)


def time_left(job_id: str) -> float | None:
    """Seconds until a Slurm job reaches its time limit (``None``: unknown)."""
    result = _squeue(job_id, "%L")
    if result is None or result.returncode != 0 or not result.stdout.strip():
        return None
    return parse_duration(result.stdout.splitlines()[0])
