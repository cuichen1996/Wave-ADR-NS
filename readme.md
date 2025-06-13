# WaveADR Neural Solver for Helmholtz Equations

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

Official implementation of the paper **"A Neural Multigrid Solver for Helmholtz Equations with High Wavenumber and Heterogeneous Media"**.([arXiv:2404.02493](https://arxiv.org/abs/2404.02493))

## Description

This repository contains the source code for a novel neural multigrid solver designed to efficiently solve Helmholtz equations with high wavenumbers and heterogeneous media. The method combines traditional multigrid techniques with deep learning to address the challenges of:

- High-frequency wave propagation (wavenumber up to 2000)
- Complex heterogeneous media configurations
- Multigrid optimization through neural networks

Key features:
- Hybrid architecture combining conventional multigrid with neural networks
- Adaptive smoothing operators learned from data
- Enhanced convergence in high-wavenumber regimes
- Robust performance across heterogeneous media



### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy
- SciPy
- Matplotlib
- timm

## Citation

```bibtex
@article{cui2025neural,
  title={A neural multigrid solver for helmholtz equations with high wavenumber and heterogeneous media},
  author={Cui, Chen and Jiang, Kai and Shu, Shi},
  journal={SIAM Journal on Scientific Computing},
  volume={47},
  number={3},
  pages={C655--C679},
  year={2025},
  doi={https://doi.org/10.1137/24M1654397}
}
```



