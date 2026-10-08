# Release on Hugging Face (T4.4)

Four repositories under the Hugging Face account `JaspinXu`, plus a Kaggle
mirror of the dataset. Create
them **private**, check each page, and make them public together with the
GitHub repository.

| Repository | Type | Source on the cluster | Card |
| --- | --- | --- | --- |
| `JaspinXu/KnowWhenToHandOff-Qwen3-4B-R2` | model | `runs/R2-s1-20261006/export/global_step_60` | [model-cards/R2.md](model-cards/R2.md) |
| `JaspinXu/KnowWhenToHandOff-Qwen3-4B-R1` | model | `runs/R1-s1-20261006/export/global_step_60` | [model-cards/R1.md](model-cards/R1.md) |
| `JaspinXu/KnowWhenToHandOff-Qwen3-4B-SFT` | model | `runs/B2-s0-20261005/final` | [model-cards/B2.md](model-cards/B2.md) |
| `JaspinXu/KnowWhenToHandOff-data` | dataset | `scripts/export_release.py` output | [dataset-card.md](dataset-card.md) |

Each model is a merged bf16 Hugging Face directory (~8 GB). The dataset folder
holds the hand-off task set, every completed real run's records, the RL
training rollouts and the SFT selection metadata; never weights or
checkpoints.

## Steps (cluster, repository root, after `git pull` and `source slurm/cluster.env`)

1. Log in once with **your own write token** (typed into the prompt; never in a
   file of this repository or a script):

   ```bash
   bash slurm/run.sh hf auth login
   ```

2. Build the dataset folder on a CPU node; read the listing in the log first
   (file count, size, one line per run), then build it:

   ```bash
   bash slurm/submit.sh cpu CMD="python scripts/export_release.py --out $HANDOFF_SCRATCH/release/data --dry-run"
   bash slurm/submit.sh cpu CMD="python scripts/export_release.py --out $HANDOFF_SCRATCH/release/data && cp docs/dataset-card.md $HANDOFF_SCRATCH/release/data/README.md"
   ```

3. Upload everything from a CPU node (large uploads do not belong on the login
   node):

   ```bash
   bash slurm/submit.sh cpu CMD="python scripts/upload_release.py --user JaspinXu"
   ```

   The script creates the four private repositories, uploads the folders and
   the cards (`README.md` in each), and prints the URLs. `--only R2 data`
   re-runs part of it.

4. Open each page and check the card, the file list and that nothing private
   is there. Make them public from each page's Settings when the GitHub
   repository goes public.

## Kaggle mirror (dataset only)

Kaggle gets the same dataset folder, never the weights; the Hugging Face copy
stays primary. Your Kaggle *handle* is the last part of your profile URL
(`kaggle.com/<handle>`), not the display name.

1. Put your API token in `~/.kaggle/access_token` on the cluster (or a legacy
   `kaggle.json` in `~/.kaggle/`), readable only by you:

   ```bash
   mkdir -p ~/.kaggle && chmod 700 ~/.kaggle
   nano ~/.kaggle/access_token        # paste the token, save
   chmod 600 ~/.kaggle/access_token
   ```

2. Install the Kaggle CLI in its own small environment (it must not touch the
   GPU environment's pinned packages):

   ```bash
   bash slurm/run.sh uv venv "$HANDOFF_SCRATCH/kaggle-venv"
   bash slurm/run.sh uv pip install --python "$HANDOFF_SCRATCH/kaggle-venv/bin/python" "kaggle==2.2.4"
   ```

3. After the Hugging Face steps above (the dataset folder must exist), check
   the staging and metadata, then upload privately from a CPU node:

   ```bash
   bash slurm/submit.sh cpu CMD="python scripts/upload_kaggle.py --user <handle> --dry-run"
   bash slurm/submit.sh cpu CMD="python scripts/upload_kaggle.py --user <handle> --kaggle $HANDOFF_SCRATCH/kaggle-venv/bin/kaggle"
   ```

   The script stages `$HANDOFF_SCRATCH/release/kaggle` with hard links and a
   `dataset-metadata.json` (licence `other`, stated as MIT in the description,
   since Kaggle's list has no MIT), then runs `kaggle datasets create` with
   sub-folders as zip archives. Later fixes: add `--version "what changed"`.

4. Check the dataset page, then make it public from its Settings together with
   the rest.
