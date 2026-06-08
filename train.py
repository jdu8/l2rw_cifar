"""L2RW training on CIFAR-10 with noisy labels, with paired-data collection.

Algorithm (after warmup):
  For each training batch (x_train, y_train):
    1. [fwd_train]   Forward x_train through a virtual copy of the model.
    2. [virtual_upd] Virtually update the copy with eps-weighted loss (SGD step).
    3. [fwd_val]     Forward x_val through the virtually updated copy.
    4. [2nd_order]   Differentiate val_loss w.r.t. eps → eps_grads.
    5. Compute weights: w = clamp(-eps_grads, min=0); w /= w.sum() (or uniform if all zero).
    6. [real_bwd]    Real forward + backward on the main model with weights w.
  Periodically log the paired data (embeddings, losses, weights) to HDF5.
"""
import argparse
import itertools
import os
import random
import time

import higher
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_cifar10_datasets
from model import ResNet32
from storage import PairStorage


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Synchronize GPU/MPS before recording wall-clock time."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    # MPS doesn't expose synchronize; we accept some timing imprecision there


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = correct = n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        n += len(labels)
    model.train()
    return total_loss / n, correct / n


# ------------------------------------------------------------------
# Training steps
# ------------------------------------------------------------------

def warmup_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Plain cross-entropy step (no L2RW)."""
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def l2rw_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    device: torch.device,
    log_timing: bool = False,
):
    """One full L2RW update.

    Returns
    -------
    result : dict with keys:
        weighted_loss, raw_weights,
        train_embeddings, train_losses,
        val_embeddings, val_losses,
        timings (only populated when log_timing=True)
    """
    timings = {}

    # ----------------------------------------------------------------
    # Phase 1 — virtual forward on training batch
    # ----------------------------------------------------------------
    if log_timing:
        sync(device)
        t0 = time.perf_counter()

    # eps is the per-example weighting variable that L2RW optimises over.
    # Initialised to zero so the first virtual gradient is computed at w=0.
    eps = torch.zeros(len(train_images), requires_grad=True, device=device)

    # higher creates a differentiable copy of model + optimizer so that
    # gradients can flow back through the virtual SGD step to eps.
    with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (meta_model, meta_opt):

        train_logits, train_embeddings = meta_model(train_images, return_embedding=True)
        train_losses = F.cross_entropy(train_logits, train_labels, reduction="none")

        if log_timing:
            sync(device)
            timings["fwd_train"] = time.perf_counter() - t0

        # ----------------------------------------------------------------
        # Phase 2 — virtual parameter update weighted by eps
        # ----------------------------------------------------------------
        if log_timing:
            sync(device)
            t0 = time.perf_counter()

        # eps-weighted loss: gradient through this step w.r.t. eps is what
        # we eventually differentiate to find optimal sample weights.
        eps_weighted_loss = (eps * train_losses).sum()
        # meta_opt.step propagates gradients through the optimizer update
        # (higher patches the optimizer so all ops stay in the autograd graph)
        meta_opt.step(eps_weighted_loss)

        if log_timing:
            sync(device)
            timings["virtual_upd"] = time.perf_counter() - t0

        # ----------------------------------------------------------------
        # Phase 3 — forward on validation batch with virtually updated params
        # ----------------------------------------------------------------
        if log_timing:
            sync(device)
            t0 = time.perf_counter()

        val_logits, val_embeddings = meta_model(val_images, return_embedding=True)
        val_losses = F.cross_entropy(val_logits, val_labels, reduction="none")
        val_loss = val_losses.mean()

        if log_timing:
            sync(device)
            timings["fwd_val"] = time.perf_counter() - t0

        # ----------------------------------------------------------------
        # Phase 4 — second-order gradient: d(val_loss) / d(eps)
        #
        # val_loss depends on eps through the chain:
        #   eps → eps_weighted_loss → meta_opt.step → meta_params → val_loss
        # higher has tracked this graph, so autograd works normally here.
        # ----------------------------------------------------------------
        if log_timing:
            sync(device)
            t0 = time.perf_counter()

        eps_grads = torch.autograd.grad(val_loss, eps)[0].detach()

        if log_timing:
            sync(device)
            timings["second_order"] = time.perf_counter() - t0

    # ----------------------------------------------------------------
    # Compute normalised weights from eps gradients.
    #
    # An example with a large positive eps_grad would, if upweighted,
    # increase val_loss → bad. We therefore clamp to negatives and flip sign:
    #   high -eps_grad  ⟹  high weight (this example helps val performance).
    # ----------------------------------------------------------------
    w_tilde = torch.clamp(-eps_grads, min=0.0)
    w_sum = w_tilde.sum()
    if w_sum > 0:
        weights = w_tilde / w_sum
    else:
        # All gradients were non-negative → no example is clearly helpful;
        # fall back to uniform weights so training still makes progress.
        weights = torch.ones_like(w_tilde) / len(w_tilde)

    # ----------------------------------------------------------------
    # Phase 5 — real forward + backward with learned weights
    # ----------------------------------------------------------------
    if log_timing:
        sync(device)
        t0 = time.perf_counter()

    real_logits, _ = model(train_images, return_embedding=True)
    real_losses = F.cross_entropy(real_logits, train_labels, reduction="none")
    weighted_loss = (weights.detach() * real_losses).sum()
    optimizer.zero_grad()
    weighted_loss.backward()
    optimizer.step()

    if log_timing:
        sync(device)
        timings["real_bwd"] = time.perf_counter() - t0

    return {
        "weighted_loss": weighted_loss.item(),
        "raw_weights": weights.detach(),
        "train_embeddings": train_embeddings.detach(),
        "train_losses": train_losses.detach(),
        "val_embeddings": val_embeddings.detach(),
        "val_losses": val_losses.detach(),
        "timings": timings,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="L2RW on CIFAR-10 with noisy labels")
    p.add_argument("--noise_rate",      type=float, default=0.4)
    p.add_argument("--noise_type",      type=str,   default="uniform", choices=["uniform", "asymmetric"])
    p.add_argument("--epochs",          type=int,   default=120)
    p.add_argument("--batch_size",      type=int,   default=100)
    p.add_argument("--val_size",        type=int,   default=1000)
    p.add_argument("--val_batch_size",  type=int,   default=100)
    p.add_argument("--warmup_epochs",   type=int,   default=10,
                   help="Epochs of plain training before L2RW starts")
    p.add_argument("--log_sample_rate", type=float, default=0.1,
                   help="Fraction of L2RW batches to write to HDF5")
    p.add_argument("--output_dir",      type=str,   default="./pairs_data/")
    p.add_argument("--data_root",       type=str,   default="./data")
    p.add_argument("--wandb_project",   type=str,   default=None)
    p.add_argument("--wandb_run_name",  type=str,   default=None)
    p.add_argument("--seed",            type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # ----------------------------------------------------------------
    # wandb
    # ----------------------------------------------------------------
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # ----------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------
    train_dataset, val_dataset, test_dataset = build_cifar10_datasets(
        data_root=args.data_root,
        noise_rate=args.noise_rate,
        noise_type=args.noise_type,
        val_size=args.val_size,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True,
    )
    # Cycle the val loader so we can sample one batch per training batch
    val_iter = itertools.cycle(val_loader)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # ----------------------------------------------------------------
    # Model + optimizer + LR schedule
    # ----------------------------------------------------------------
    model = ResNet32(num_classes=10).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True
    )
    # Decay LR at 80 and 100 epochs (common schedule for CIFAR-10 / 120 epoch runs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.1)

    # ----------------------------------------------------------------
    # Storage
    # ----------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    hdf5_path = os.path.join(
        args.output_dir,
        f"pairs_{args.noise_type}_nr{args.noise_rate}_seed{args.seed}.h5",
    )
    storage = PairStorage(hdf5_path)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        is_warmup = epoch <= args.warmup_epochs

        epoch_loss = 0.0
        epoch_weight_mean = 0.0
        epoch_weight_max = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            if is_warmup:
                loss = warmup_step(model, optimizer, images, labels)
                if use_wandb:
                    wandb.log({"train/loss": loss, "step": global_step})
                epoch_loss += loss
            else:
                # Sample one val batch (cycled, balanced)
                val_images, val_labels = next(val_iter)
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                should_log = random.random() < args.log_sample_rate
                result = l2rw_step(
                    model, optimizer,
                    images, labels,
                    val_images, val_labels,
                    device,
                    log_timing=True,
                )

                loss = result["weighted_loss"]
                weights = result["raw_weights"]
                timings = result["timings"]

                epoch_loss += loss
                epoch_weight_mean += weights.mean().item()
                epoch_weight_max  += weights.max().item()

                # Batch-level wandb logs
                if use_wandb:
                    log_dict = {
                        "train/weighted_loss": loss,
                        "train/weight_mean":   weights.mean().item(),
                        "train/weight_max":    weights.max().item(),
                        "train/weight_std":    weights.std().item(),
                        "train/nonzero_frac":  (weights > 0).float().mean().item(),
                        "step": global_step,
                    }
                    for k, v in timings.items():
                        log_dict[f"timing/{k}_ms"] = v * 1000
                    wandb.log(log_dict)

                # Persist paired data when sampled
                if should_log:
                    storage.write(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        train_embeddings=result["train_embeddings"],
                        train_losses=result["train_losses"],
                        val_embeddings=result["val_embeddings"],
                        val_losses=result["val_losses"],
                        l2rw_weights=weights,
                    )

            global_step += 1
            n_batches += 1

        scheduler.step()

        # ----------------------------------------------------------------
        # Epoch-level evaluation
        # ----------------------------------------------------------------
        train_loss_avg = epoch_loss / n_batches
        val_loss, val_acc   = evaluate(model, val_loader,  device)
        test_loss, test_acc = evaluate(model, test_loader, device)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | lr={lr_now:.4f} | "
            f"train_loss={train_loss_avg:.4f} | "
            f"val_acc={val_acc:.4f} | test_acc={test_acc:.4f}"
            + (f" | wt_mean={epoch_weight_mean/n_batches:.4f}" if not is_warmup else "")
        )

        if use_wandb:
            log_dict = {
                "epoch": epoch,
                "epoch/train_loss": train_loss_avg,
                "epoch/val_loss":   val_loss,
                "epoch/val_acc":    val_acc,
                "epoch/test_loss":  test_loss,
                "epoch/test_acc":   test_acc,
                "epoch/lr":         lr_now,
            }
            if not is_warmup:
                log_dict["epoch/weight_mean"] = epoch_weight_mean / n_batches
                log_dict["epoch/weight_max"]  = epoch_weight_max  / n_batches
            wandb.log(log_dict)

    # ----------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------
    storage.close()
    if use_wandb:
        wandb.finish()
    print(f"\nPaired data saved to: {hdf5_path}")


if __name__ == "__main__":
    main()
