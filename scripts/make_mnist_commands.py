#!/usr/bin/env python3
"""
Generate commands file for MNIST experiments.

One run = one command (suitable for SLURM array submission using scripts/launch_slurm_array.sh).
"""
import itertools
import argparse
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="commands_mnist.txt")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--gmm-k", type=int, default=10)
    ap.add_argument("--out-dir", type=str, default="logs_mnist")
    ap.add_argument("--n-latent", type=int, default=10)
    args = ap.parse_args()

    seeds = args.seeds
    gmm_k = args.gmm_k
    out_dir = args.out_dir
    n_latent = args.n_latent

    priors = ["standard", "gmm", "vamp"]

    lines = []
    for prior, seed in itertools.product(priors, seeds):
        # (A) density reg disabled -> omit the flag entirely
        cmd = f"python scripts/exp_mnist_runner.py --prior {prior} --aligner none --gmm-k {gmm_k}  --seed {seed} --epochs 50 --log-root {out_dir} --n-latent {n_latent}"
        lines.append(cmd.strip())

        # (B) density reg enabled -> include the boolean flag `--density-reg`
        for aln in ["direct", "flow"]:
            cmd = f"python scripts/exp_mnist_runner.py --prior {prior} --density-reg --aligner {aln} --gmm-k {gmm_k} --seed {seed} --epochs 50 --log-root {out_dir} --n-latent {n_latent}"
            lines.append(cmd.strip())

    # write
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} commands to {out_path}")

if __name__ == "__main__":
    main()
