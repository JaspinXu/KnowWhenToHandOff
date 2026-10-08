# Report (T4.2)

LaTeX source of the course report (≤ 10 pages excluding the title page and
references, 12 pt, 1-inch margins, Vancouver references; NUS CS4246/5446
project guidelines).

| File | What |
| --- | --- |
| `main.tex` | The report; references are a hand-written Vancouver list (numbered by first citation) |
| `make_figures.py` | Draws `figures/*.pdf` from `reports/main/main_table.csv` and `reports/rl/dev_curves.csv` |
| `main.pdf` | Public build: team names only |
| `team.local.tex` | **Gitignored.** Names, matriculation numbers and e-mails for the submitted copy |
| `submission.pdf` | **Gitignored.** The copy uploaded to Canvas |

Every number in the report comes from `reports/main/`, `reports/rl/`,
`docs/experiment-log.md` or `docs/rl-audit.md`; the test_handoff precision,
recall and over-escalation values are the runs' `metrics.json` values,
recorded in AGENTS.md §2.

## Build

```bash
bash slurm/run.sh python report/make_figures.py   # repo root; needs matplotlib (cluster env)
cd report
latexmk -pdf main.tex                  # public copy (names only)
latexmk -pdf -jobname=submission main.tex   # Canvas copy, reads team.local.tex
latexmk -c                             # remove intermediates
```

Never commit `team.local.tex` or `submission.pdf` (AGENTS §18: no student
numbers or e-mails in the repository).
