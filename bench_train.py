"""Lean L2RW trainer — epoch-level output only, best checkpoint saving.

Differences from train.py:
- No per-batch tqdm or wandb step logs
- One print line per epoch (prefixed with run name for parallel readability)
- Saves best-val-acc checkpoint to --checkpoint_dir
- --model resnet32 | resnet20
"""
import argparse
import itertools
import os
import random
import time

import higher
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from dataset import build_cifar10_datasets
from model import ResNet20, ResNet32
from storage import PairStorage


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
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


def warmup_step(model, optimizer, images, labels):
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def l2rw_step(model, optimizer, train_images, train_labels, val_images, val_labels, device):
    eps = torch.zeros(len(train_images), requires_grad=True, device=device)
    with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fmodel, fopt):
        train_logits, train_emb = fmodel(train_images, return_embedding=True)
        train_losses = F.cross_entropy(train_logits, train_labels, reduction="none")
        fopt.step((eps * train_losses).sum())
        val_logits, val_emb = fmodel(val_images, return_embedding=True)
        val_losses = F.cross_entropy(val_logits, val_labels, reduction="none")
        eps_grads = torch.autograd.grad(val_losses.mean(), eps)[0].detach()

    w = torch.clamp(-eps_grads, min=0.0)
    w_sum = w.sum()
    weights = w / w_sum if w_sum > 0 else torch.ones_like(w) / len(w)

    real_logits, _ = model(train_images, return_embedding=True)
    real_losses = F.cross_entropy(real_logits, train_labels, reduction="none")
    loss = (weights.detach() * real_losses).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "weights": weights.detach(),
        "train_emb": train_emb.detach(),
        "train_losses": train_losses.detach(),
        "val_emb": val_emb.detach(),
        "val_losses": val_losses.detach(),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           default="resnet32", choices=["resnet32", "resnet20"])
    p.add_argument("--noise_rate",      type=float, default=0.4)
    p.add_argument("--noise_type",      type=str,   default="uniform", choices=["uniform", "asymmetric"])
    p.add_argument("--epochs",          type=int,   default=10)
    p.add_argument("--batch_size",      type=int,   default=100)
    p.add_argument("--val_size",        type=int,   default=1000)
    p.add_argument("--val_batch_size",  type=int,   default=100)
    p.add_argument("--warmup_epochs",   type=int,   default=0)
    p.add_argument("--log_sample_rate", type=float, default=0.1)
    p.add_argument("--output_dir",      type=str,   default="./pairs_data/")
    p.add_argument("--checkpoint_dir",  type=str,   default="./checkpoints/")
    p.add_argument("--data_root",       type=str,   default="./data")
    p.add_argument("--wandb_project",   type=str,   default=None)
    p.add_argument("--wandb_run_name",  type=str,   default=None)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--lr",              type=float, default=0.1)
    p.add_argument("--weight_decay",    type=float, default=2e-4)
    p.add_argument("--lr_milestones",   type=int,   nargs="+", default=[82, 123])
    p.add_argument("--baseline",        action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    run_name = args.wandb_run_name or (
        f"{'baseline' if args.baseline else 'l2rw'}"
        f"_{args.model}_{args.noise_type}_nr{args.noise_rate}_s{args.seed}"
    )

    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    train_ds, val_ds, test_ds = build_cifar10_datasets(
        data_root=args.data_root,
        noise_rate=args.noise_rate,
        noise_type=args.noise_type,
        val_size=args.val_size,
        seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.val_batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                              num_workers=4, pin_memory=True)
    val_iter = itertools.cycle(val_loader)

    model_cls = ResNet32 if args.model == "resnet32" else ResNet20
    model = model_cls(num_classes=10).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=0.1,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{run_name}_best.pt")
    best_val_acc = 0.0

    storage = None
    if not args.baseline:
        os.makedirs(args.output_dir, exist_ok=True)
        hdf5_path = os.path.join(
            args.output_dir,
            f"pairs_{args.model}_{args.noise_type}_nr{args.noise_rate}_s{args.seed}.h5",
        )
        storage = PairStorage(hdf5_path)

    t_start = time.perf_counter()

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            is_warmup = epoch <= args.warmup_epochs

            epoch_loss = epoch_w_mean = n_batches = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                if is_warmup or args.baseline:
                    epoch_loss += warmup_step(model, optimizer, images, labels)
                else:
                    val_images, val_labels = next(val_iter)
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)
                    result = l2rw_step(
                        model, optimizer,
                        images, labels,
                        val_images, val_labels,
                        device,
                    )
                    epoch_loss   += result["loss"]
                    epoch_w_mean += result["weights"].mean().item()
                    if storage is not None and random.random() < args.log_sample_rate:
                        storage.write(
                            epoch=epoch, batch_idx=batch_idx,
                            train_embeddings=result["train_emb"],
                            train_losses=result["train_losses"],
                            val_embeddings=result["val_emb"],
                            val_losses=result["val_losses"],
                            l2rw_weights=result["weights"],
                        )

                n_batches += 1

            scheduler.step()

            val_loss, val_acc   = evaluate(model, val_loader,  device)
            test_loss, test_acc = evaluate(model, test_loader, device)
            lr_now  = optimizer.param_groups[0]["lr"]
            elapsed = time.perf_counter() - t_start

            w_str = f" | w={epoch_w_mean/n_batches:.4f}" if not is_warmup and not args.baseline else ""
            print(
                f"[{run_name}] {epoch:3d}/{args.epochs} | {elapsed:6.1f}s"
                f" | lr={lr_now:.4f} | loss={epoch_loss/n_batches:.4f}"
                f" | val={val_acc:.4f} | test={test_acc:.4f}{w_str}",
                flush=True,
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {"epoch": epoch, "val_acc": val_acc,
                     "model_state": model.state_dict(), "args": vars(args)},
                    ckpt_path,
                )

            if use_wandb:
                log = {
                    "epoch": epoch,
                    "epoch/train_loss": epoch_loss / n_batches,
                    "epoch/val_loss":   val_loss,
                    "epoch/val_acc":    val_acc,
                    "epoch/test_loss":  test_loss,
                    "epoch/test_acc":   test_acc,
                    "epoch/lr":         lr_now,
                }
                if not is_warmup and not args.baseline:
                    log["epoch/weight_mean"] = epoch_w_mean / n_batches
                wandb.log(log)

    finally:
        if storage is not None:
            storage.close()
        if use_wandb:
            wandb.finish()

    print(f"[{run_name}] done | best_val_acc={best_val_acc:.4f} | ckpt={ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
