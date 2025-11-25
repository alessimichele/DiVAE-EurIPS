# Density-Informed VAE (DiVAE)
**Reliable Log-Prior Probability via Density Alignment Regularization**  
_Code release for the PriGM Workshop @ EurIPS 2025_

This repository contains the official implementation of **DiVAE**, a Variational Autoencoder with **Density Alignment Regularization** that improves the reliability of the log-prior term.  
It includes all code necessary to reproduce the experiments on **synthetic datasets** and **MNIST**, as presented in the paper.

---

## ⚠️ DISCLAIMER (IMPORTANT)

The current implementation of the **density estimator** includes a **C backend** that is **only supported on Linux**.

- Training and experiments **must be run on a Linux environment**.
- macOS and Windows are **not supported at the moment** for density estimation.
- A pure-Python or cross-platform fallback will be added in future updates.

---

## 🚀 Installation

DiVAE requires **Python 3.9+** and **PyTorch 2.0+**.  
You may use either `venv` or `conda`.

### **Using Python venv**

```bash
git clone https://github.com/alessimichele/DiVAE-EurIPS.git
cd DiVAE-EurIPS

# Create and activate a virtual environment
python -m venv divae
source divae/bin/activate        # Linux/macOS
# divae\Scripts\activate       # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

### **Using conda**

```bash
git clone https://github.com/alessimichele/DiVAE-EurIPS.git
cd DiVAE-EurIPS

conda create -n divae python=3.9
conda activate divae

pip install -r requirements.txt
```

---

## 📄 Reproducing the Experiments

To generate all commands needed to replicate the experiments from the paper, run:

```bash
bash scripts/replicate_exps.sh
```

This script outputs the complete set of experiment commands.  
Each produced line corresponds to one configuration and internally calls:

- `exp_mnist_runner.py`
- `exp_synthetic_runner.py`

You may execute the entire sweep or run experiments individually.

---

## 📦 Citation

If you use this code or build upon DiVAE, please cite:

```bibtex
@inproceedings{alessi2025divae,
  title     = {Density-Informed VAE (DiVAE): Reliable Log-Prior Probability via Density Alignment Regularization},
  author    = {Alessi, Michele and Ansuini, Alessio and Rodriguez, Alex},
  booktitle = {Proceedings of the PriGM Workshop at EurIPS 2025},
  year      = {2025}
}
```

---

## 🙌 Acknowledgements

This work was presented at the **PriGM Workshop @ EurIPS 2025**.  

---

## 📝 License

This project is released under the **MIT License**.  
See the `LICENSE` file for details.
