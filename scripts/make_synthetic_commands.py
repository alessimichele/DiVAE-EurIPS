#!/usr/bin/env python3
"""
Generate commands file for synthetic GMMDataset experiments.

One run = one command (suitable for SLURM array submission using scripts/launch_slurm_array.sh).
"""
import itertools
import argparse
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="commands_synthetic")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--gmm-k", type=int, default=8)
    ap.add_argument("--gmm-dim", type=int, default=50)
    ap.add_argument("--gmm-seed", type=int, default=123)
    ap.add_argument("--log-root", type=str, default="logs_synthetic")
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()

    seeds = args.seeds
    log_root = args.log_root + f"_{args.gmm_seed}"


    # Datasets: symmetric False/True
    priors = ["standard", "gmm"]

    lines = []
    for prior, seed in itertools.product(priors, seeds):
        # (A) density reg disabled -> omit the flag entirely
        common_args = f"--gmm-k {args.gmm_k} --gmm-dim {args.gmm_dim} --gmm-seed {args.gmm_seed} --seed {seed} --epochs {args.epochs} --log-root {log_root}"

        cmd = f"python scripts/exp_synthetic_runner.py --prior {prior} --aligner none --external-density oracle {common_args}"
        lines.append(cmd.strip())

        # (B) density reg enabled -> include the boolean flag `--density-reg`
        for aln in ["direct", "flow"]:
            for ext in ["oracle", "dpa"]:
                cmd = f"python scripts/exp_synthetic_runner.py --prior {prior} --density-reg --aligner {aln} --external-density {ext} {common_args}"
                lines.append(cmd.strip())

    # write
    out_path = args.out + f"_{args.gmm_seed}.txt"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(lines)} commands to {out_path}")

if __name__ == "__main__":
    main()
