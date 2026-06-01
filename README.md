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
- Jupyter `~=1.1.1`

## First simulator/pricer implementation

The `src/stationary_heston` package contains a first object-oriented version of
the non-calibration part of the paper:

- `HestonParameters` stores the risk-neutral Heston parameters and exposes the
  invariant CIR Gamma law.
- `CIRStationarySimulator` simulates the standalone CIR variance, with initial
  variance set by either the stationary Gamma law, the last available variance,
  or the CIR mean.
- `HestonPathSimulator` simulates Heston paths with the paper's positive
  boosted-Milstein variance step and an Euler step on the log spot.
- `EuropeanMonteCarloPricer`, `BermudanLSMPricer`, and
  `BarrierMonteCarloPricer` price European, Bermudan, and barrier products.
- `compare_initialisations` prices the same product under the three variance
  initialisation strategies.

Run the example from the repository root:

```bash
PYTHONPATH=src python src/main.py
```

This first version deliberately uses plain Monte Carlo and Longstaff-Schwartz
regression for Bermudan options. It is meant as a readable baseline; the paper's
recursive quantization machinery can be added behind the same product/pricer
interfaces later.
