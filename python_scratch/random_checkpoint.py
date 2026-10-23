"""Write a checkpoint with random weights in the shape of a real config.

Kernel correctness and speed do not depend on the weights being trained, so this lets the C++
engine be validated against PyTorch at the full `small` size (6 layers, d=512) before - or
without - a training run. Weights are scaled up from the training init so logits have a
meaningful spread. The output is NOT a trained model and produces gibberish.
"""
import argparse

import torch

from torch_model import TorchGPT
from transformer import CONFIGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="small", choices=list(CONFIGS))
    ap.add_argument("--out", default="checkpoints/random_small.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = CONFIGS[args.config]
    model = TorchGPT(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim > 1:
                p.copy_(torch.randn_like(p) * (0.3 if name == "emb.E" else 0.08))
            else:
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": 0, "val_loss": float("nan")}, args.out)
    print(f"{args.out}: {model.num_params():,} random parameters")


if __name__ == "__main__":
    main()
