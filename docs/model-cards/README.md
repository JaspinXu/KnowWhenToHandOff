# Model cards (T4.4)

Filled-in cards for the released checkpoints, from the template in
[../model-card.md](../model-card.md). Every number comes from a committed
report table or a named run's `metrics.json`.

| Card | Checkpoint | Why release it |
| --- | --- | --- |
| [R2.md](R2.md) | `R2-s1-20261006`, step 60 | Best-calibrated hand-offs (the main result) |
| [R1.md](R1.md) | `R1-s1-20261006`, step 60 | The over-escalation case study (RQ4) |

## Upload (owner, cluster login node)

Each export is a merged bf16 Hugging Face directory (~8 GB). Log in with your
own write token when `hf` asks (never put it in the repository or a script),
upload privately, check the page, then make it public after the report is
submitted. `huggingface-hub` 0.36 in the cluster environment provides `hf`.

```bash
bash slurm/run.sh hf auth login
for r in R1 R2; do
  bash slurm/run.sh hf upload <hf-user>/KnowWhenToHandOff-$r \
    "$HANDOFF_SCRATCH/runs/$r-s1-20261006/export/global_step_60" . --private
  bash slurm/run.sh hf upload <hf-user>/KnowWhenToHandOff-$r \
    docs/model-cards/$r.md README.md --private
done
```
