# Reports

Generated from run IDs only (T4.1), never edited by hand:

```bash
python scripts/make_report.py --name main --baseline B2 --runs B0=<run id> B2=<run id> R2=<run id>
```

`reports/<name>/` holds the tables (`.md`, `.csv`, `.tex`), figures (`.png`) and
`sources.json`, which lists the run ID, git SHA, tau2 commit, model revisions and
seeds behind every number. Final versions are committed.

`reports/review/` (human-review packets with full dialogues, from
`scripts/sample_review.py`) is not committed. `reports/demo/` holds the offline
HTML demo pages from `scripts/render_demo.py`.
