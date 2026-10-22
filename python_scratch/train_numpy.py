"""Train the `tiny` config end to end with the pure-NumPy stack (no autograd)."""
import argparse
import math
import time

import numpy as np

from data import decode, encode, get_batch, load_tinyshakespeare
from optimizers import AdamW, clip_grad_norm, cosine_lr
from transformer import CONFIGS, NumpyGPT


def evaluate(model, data, batch_size, seq_len, rng, n_batches=20):
    losses = [model.loss(*get_batch(data, batch_size, seq_len, rng)) for _ in range(n_batches)]
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--out", default="checkpoints/numpy_tiny.npz")
    args = ap.parse_args()

    cfg = CONFIGS["tiny"]
    train, val = load_tinyshakespeare()
    model = NumpyGPT(cfg, seed=0)
    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.1)
    rng = np.random.default_rng(1)
    print(f"tiny: {model.num_params():,} params | train {len(train):,} bytes | val {len(val):,} bytes")

    t0 = time.time()
    for step in range(args.steps):
        opt.lr = cosine_lr(step, args.lr, args.lr / 10, args.warmup, args.steps)
        x, y = get_batch(train, args.batch_size, cfg.max_seq, rng)
        loss = model.loss(x, y)
        model.backward()
        gnorm = clip_grad_norm(model.gradients(), 1.0)
        opt.step(model.gradients())
        if step % 50 == 0:
            print(f"step {step:5d} | loss {loss:.4f} | gnorm {gnorm:.2f} | lr {opt.lr:.2e} "
                  f"| {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            vl = evaluate(model, val, args.batch_size, cfg.max_seq, np.random.default_rng(2))
            print(f"  >> val loss {vl:.4f} | val ppl {math.exp(vl):.2f}", flush=True)

    np.savez(args.out, **model.parameters())
    prompt = "ROMEO:\n"
    out = model.generate(encode(prompt), 300, temperature=0.8)
    print("\n--- sample ---\n" + decode(out))


if __name__ == "__main__":
    main()
