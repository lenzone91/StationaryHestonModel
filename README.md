# Stationary Heston Model

This repository contains a reproduction project based on the article on the *Stationary Heston model* ([arXiv:2001.03101](https://arxiv.org/abs/2001.03101)).

The goal of this project is to reproduce and analyze the main results of the paper by comparing the Stationary Heston model, an extension of the standard Heston model, with the original model, especially in terms of short-maturity implied volatility smiles. The project also implements numerical pricing methods for exotic derivatives, in particular Bermudan and Barrier options.

The notebook includes a short Gamma-law simulation study with:

- direct sampling with NumPy,
- interactive parameter exploration with Plotly and notebook sliders,
- and a rejection-based simulation method together with a discussion of its limitations.

## Dependencies

The project uses a `mamba` environment defined in `dependencies.yml`.
The same file is compatible with `conda` if you replace the `mamba` commands below with their `conda` equivalents.

### Setup

Create the environment:

```bash
mamba env create -f dependencies.yml
```

Activate it:

```bash
mamba activate stationary-heston-model
```

Start JupyterLab:

```bash
jupyter lab
```

Then open [`gamma_law_simulation.ipynb`](/home/enzo/code/el_karoui/StationaryHestonModel/gamma_law_simulation.ipynb).

The interactive Plotly section uses `ipywidgets`.

The environment includes:

- Python `~=3.11.15`
- NumPy `~=2.4.3`
- SciPy `~=1.17.1`
- pandas `~=3.0.2`
- matplotlib `~=3.10.8`
- Plotly `~=6.5`
- ipywidgets `~=8.1.8`
- JupyterLab `~=4.5.6`
- ipykernel `~=7.2.0`
