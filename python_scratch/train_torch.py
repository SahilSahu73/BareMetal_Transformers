"""Train the PyTorch GPT (CPU or CUDA). Used for the `small` config; see notebooks/colab_train.ipynb."""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import decode, encode, get_batch, load_tinyshakespeare
from optimizers import cosine_lr
from torch_model import TorchGPT
from transformer import CONFIGS


@torch.no_grad()
def evaluate(model, data, cfg, device, batch_size, n_batches=50):
    model.eval()
    rng = np.random.default_rng(2)
    losses = []
    for _ in range(n_batches):
        x, y = (torch.from_numpy(a).to(device) for a in get_batch(data, batch_size, cfg.max_seq, rng))
        losses.append(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def sample(model, cfg, device, prompt, n_new, temperature=0.8):
    ids = encode(prompt).tolist()
    for _ in range(n_new):
        ctx = torch.tensor([ids[-cfg.max_seq:]], device=device)
        logits = model(ctx)[0, -1] / temperature
        ids.append(torch.multinomial(torch.softmax(logits.float(), -1), 1).item())
    return decode(ids)


def save(model, cfg, step, val_loss, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step,
                "val_loss": val_loss}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="small", choices=list(CONFIGS))
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast (CUDA / modern CPUs)")
    ap.add_argument("--out", default="checkpoints/torch_small.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = CONFIGS[args.config]
    model = TorchGPT(cfg).to(device)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    train, val = load_tinyshakespeare()
    rng = np.random.default_rng(1)
    print(f"{args.config}: {model.num_params():,} params on {device} | "
          f"batch {args.batch_size} x ctx {cfg.max_seq}", flush=True)

    t0, best = time.time(), float("inf")
    for step in range(args.steps):
        lr = cosine_lr(step, args.lr, args.lr / 10, args.warmup, args.steps)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = (torch.from_numpy(a).to(device) for a in get_batch(train, args.batch_size, cfg.max_seq, rng))
        with torch.autocast(device, dtype=torch.bfloat16, enabled=args.bf16):
            logits = model(x)
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | gnorm {gnorm:.2f} | lr {lr:.2e} "
                  f"| {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            vl = evaluate(model, val, cfg, device, args.batch_size)
            print(f"  >> val loss {vl:.4f} | val ppl {math.exp(vl):.2f}", flush=True)
            if vl < best:
                best = vl
                save(model, cfg, step + 1, vl, args.out)

    print(f"\nbest val loss {best:.4f} (ppl {math.exp(best):.2f}) -> {args.out}")
    print("\n--- sample ---\n" + sample(model, cfg, device, "ROMEO:\n", 400))


if __name__ == "__main__":
    main()
