#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Generating MNIST commands..."
python3 scripts/make_mnist_commands.py \
	--out commands_mnist.txt \
	--seeds 1 2 3

echo "Generating synthetic commands (setup 1: k=8, dim=50, seed=6032000)..."
python3 scripts/make_synthetic_commands.py \
	--seeds 1 2 3 \
	--gmm-k 8 \
	--gmm-dim 50 \
	--gmm-seed 6032000 \
	--epochs 100

echo "Generating synthetic commands (setup 2: k=4, dim=50, seed=2031965)..."
python3 scripts/make_synthetic_commands.py \
	--seeds 1 2 3 \
	--gmm-k 4 \
	--gmm-dim 50 \
	--gmm-seed 2031965 \
	--epochs 100

echo "Generating synthetic commands (setup 3: k=12, dim=30, seed=14082005)..."
python3 scripts/make_synthetic_commands.py \
	--seeds 1 2 3 \
	--gmm-k 12 \
	--gmm-dim 30 \
	--gmm-seed 14082005 \
	--epochs 100

echo "Done. Generated command files in:"
echo "  - commands_mnist.txt"
echo "  - commands_synthetic_6032000.txt"
echo "  - commands_synthetic_2031965.txt"