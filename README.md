# L2RW CIFAR-10 Paired Data Collection

Runs Learning to Reweight Examples (L2RW) on CIFAR-10 with noisy labels and logs the inputs and outputs of each reweighting step as paired data for downstream use.

## What it does

At each L2RW step the algorithm:
1. Forwards a noisy training batch through a virtual copy of the model
2. Takes a differentiable virtual SGD step weighted by learnable per-example scalars (`eps`)
3. Forwards a clean validation batch through the virtually updated model
4. Differentiates the validation loss back through the virtual update to get `d(val_loss)/d(eps)`
5. Normalises `clamp(-eps_grads, min=0)` into example weights and applies them to the real update

Every sampled step writes `(train_embeddings, train_losses, val_embeddings, val_losses, l2rw_weights)` to HDF5 for later use.

## Project layout

```
train.py        Main training loop — warmup + L2RW + logging
model.py        ResNet-32 for CIFAR-10; forward() optionally returns penultimate embeddings
dataset.py      CIFAR-10 with uniform or asymmetric label noise; balanced clean val split
storage.py      HDF5 writer — flush + fsync after every write
requirements.txt
.env.example    Template for wandb credentials
```

## Setup

> **Python note:** use `/opt/homebrew/bin/python3` (Homebrew Python 3.14) to create the venv. The pyenv 3.13 build on this machine is missing `_lzma` and cannot import torchvision.

```bash
/opt/homebrew/bin/python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## wandb

Get your API key at https://wandb.ai/authorize, then log in once:

```bash
wandb login
```

The key is saved to `~/.netrc` and reused automatically. Alternatively, set `WANDB_API_KEY` in your environment (see `.env.example`).

## Running

```bash
source venv/bin/activate

python train.py \
  --noise_type    uniform \
  --noise_rate    0.4 \
  --epochs        120 \
  --warmup_epochs 10 \
  --batch_size    100 \
  --val_size      1000 \
  --log_sample_rate 0.1 \
  --output_dir    ./pairs_data/ \
  --wandb_project l2rw-cifar10 \
  --wandb_run_name uniform_nr0.4_seed42 \
  --seed          42
```

Asymmetric noise:

```bash
python train.py --noise_type asymmetric --noise_rate 0.4 --wandb_project l2rw-cifar10
```

## Command-line arguments

| Argument | Default | Description |
|---|---|---|
| `--noise_rate` | `0.4` | Fraction of training labels to corrupt |
| `--noise_type` | `uniform` | `uniform` or `asymmetric` |
| `--epochs` | `120` | Total training epochs |
| `--batch_size` | `100` | Training batch size |
| `--val_size` | `1000` | Clean validation set size (100 per class) |
| `--val_batch_size` | `100` | Validation batch size per L2RW step |
| `--warmup_epochs` | `10` | Plain training epochs before L2RW starts |
| `--log_sample_rate` | `0.1` | Fraction of L2RW batches written to HDF5 |
| `--output_dir` | `./pairs_data/` | Directory for HDF5 output files |
| `--data_root` | `./data` | Where CIFAR-10 is downloaded |
| `--wandb_project` | *(disabled)* | wandb project name; omit to skip wandb |
| `--wandb_run_name` | `None` | Optional run label in wandb |
| `--seed` | `42` | Random seed |

## Output format

HDF5 file at `pairs_data/pairs_<noise_type>_nr<rate>_seed<seed>.h5`:

```
/epoch_11/batch_0/
    train_embeddings   float32  [100, 64]
    train_losses       float32  [100]
    val_embeddings     float32  [100, 64]
    val_losses         float32  [100]
    l2rw_weights       float32  [100]
    attrs: epoch, batch_idx
```

Groups are written and synced atomically; re-running from a crash safely skips already-written groups.

## Model

ResNet-32 (He et al., CIFAR variant): 3 stages of 5 BasicBlocks each, 64-d global average pooling before the linear head. LR schedule: 0.1 → 0.01 at epoch 80 → 0.001 at epoch 100, SGD with momentum 0.9 and weight decay 1e-4.

## Noise types

**Uniform (symmetric):** each label is independently replaced with a uniform random class with probability `noise_rate`.

**Asymmetric:** labels are flipped to a visually similar class with probability `noise_rate` — airplane↔bird, automobile↔truck, cat↔dog, deer↔horse.

## Reference

Ren et al., *Learning to Reweight Examples for Robust Deep Learning*, ICML 2018.
