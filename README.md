Code for the paper: **"Density-Informed VAE (DiVAE): Reliable Log-Prior Probability via Density Alignment Regularization"** (PriGM Workshop @ EurIPS 2025).  
This repository contains the implementation of DiVAE and the scripts to reproduce the experiments on synthetic datasets and MNIST.

## Installation

Tested with Python 3.9+ and PyTorch 2.0.

```bash
git clone https://github.com/alessimichele/DiVAE-EurIPS.git
cd DiVAE-EurIPS

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # on Linux/macOS
# .venv\Scripts\activate         # on Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```
If you use conda
```bash
conda create -n divae python=3.9
conda activate divae
pip install -r requirements.txt
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{alessi2025divae,
  title     = {Density-Informed VAE (DiVAE): Reliable Log-Prior Probability via Density Alignment Regularization},
  author    = {Alessi, Michele and Ansuini, Alessio and Rodriguez, Alex},
  booktitle = {Proceedings of the PriGM Workshop at EurIPS 2025}
  year      = {2025}
}
