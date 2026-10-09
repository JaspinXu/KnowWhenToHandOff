# Release

What is published, where it comes from, and how the dataset folder is built.
Everything is public; the Hugging Face copies are primary.

| Repository | Type | Source on the cluster | Card | DOI |
| --- | --- | --- | --- | --- |
| [`JaspinXu/KnowWhenToHandOff-Qwen3-4B-R2`](https://huggingface.co/JaspinXu/KnowWhenToHandOff-Qwen3-4B-R2) | model | `runs/R2-s1-20261006/export/global_step_60` | [model-cards/R2.md](model-cards/R2.md) | [10.57967/hf/10823](https://doi.org/10.57967/hf/10823) |
| [`JaspinXu/KnowWhenToHandOff-Qwen3-4B-R1`](https://huggingface.co/JaspinXu/KnowWhenToHandOff-Qwen3-4B-R1) | model | `runs/R1-s1-20261006/export/global_step_60` | [model-cards/R1.md](model-cards/R1.md) | [10.57967/hf/10824](https://doi.org/10.57967/hf/10824) |
| [`JaspinXu/KnowWhenToHandOff-Qwen3-4B-SFT`](https://huggingface.co/JaspinXu/KnowWhenToHandOff-Qwen3-4B-SFT) | model | `runs/B2-s0-20261005/final` | [model-cards/B2.md](model-cards/B2.md) | [10.57967/hf/10825](https://doi.org/10.57967/hf/10825) |
| [`JaspinXu/KnowWhenToHandOff-data`](https://huggingface.co/datasets/JaspinXu/KnowWhenToHandOff-data) | dataset | `scripts/export_release.py` output | [dataset-card.md](dataset-card.md) | [10.57967/hf/10822](https://doi.org/10.57967/hf/10822) |

The four are grouped in the collection
[`JaspinXu/knowwhentohandoff`](https://huggingface.co/collections/JaspinXu/knowwhentohandoff).
The dataset is mirrored on Kaggle as
[`zhaobinxu/knowwhentohandoff-data`](https://www.kaggle.com/datasets/zhaobinxu/knowwhentohandoff-data)
(DOI [10.34740/kaggle/dsv/20472451](https://doi.org/10.34740/kaggle/dsv/20472451);
the dataset only, never weights), with the CPU-only example notebook
[`notebooks/kaggle_quickstart.ipynb`](../notebooks/kaggle_quickstart.ipynb)
published as
[`zhaobinxu/knowwhentohandoff-quickstart`](https://www.kaggle.com/code/zhaobinxu/knowwhentohandoff-quickstart).

## Models

Each model is a merged bf16 Hugging Face directory (~8 GB) exported during
training (`handoff.train.export`); the card in `docs/model-cards/` is its
`README.md`. Bookkeeping files with cluster paths are left out.

## Dataset folder

The hand-off task set and manifests (`tasks/`), every completed real run's
records (`runs/<run id>/`: trajectories, metrics, summary, provenance, resolved
config, and for RL runs the per-episode training rollouts), the SFT selection
metadata (`sft/<dataset>/`), a run index (`runs.jsonl`) and `SHA256SUMS`. No
weights or checkpoints. Absolute paths are scrubbed and machine names replaced
by `<host>` before anything is written (`src/handoff/release.py`).

Build it on a CPU node (read the dry-run listing first):

```bash
bash slurm/submit.sh cpu CMD="python scripts/export_release.py --out $HANDOFF_SCRATCH/release/data --dry-run"
bash slurm/submit.sh cpu CMD="python scripts/export_release.py --out $HANDOFF_SCRATCH/release/data && cp docs/dataset-card.md $HANDOFF_SCRATCH/release/data/README.md"
```

## Updating a release

Uploads and page edits are done by hand (`hf upload`, or the Hugging Face and
Kaggle web pages). The DOIs are fixed: the repositories can no longer be
deleted, renamed or made private, so publish changes as new versions. After
editing a card here, paste it into the repository's `README.md` on Hugging
Face; before any upload, check the text files for paths, host names, tokens
and personal identifiers.
