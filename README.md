# QEEP (Quadratic Estimators for Equivalence Principle)

This is a code for general reconstruction of a long-density mode of large-scale structure using Quadratic Estimators (QE).

 It has been originally written to accompany the work: "Testing the Equivalence Principle with Quadratic Estimators", arxiv: [2510.13803](https://arxiv.org/pdf/2510.13803).

Currently, the code can reconstruct density modes with applications to gravitational non-linearity, the equivalence principle, and primordial non-Gaussianity.

It still can not be applied to data, but pull requests are welcomed! 

Please, get in touch if you find bugs/errors.

# Installation

`pip install -e .`

Optional extras: `pip install -e ".[theory]"` for the symbolic kernels used by `scripts/thy.py` and the paper notebooks, and `pip install -e ".[abacus]"` for running on Abacus simulations.

# Usage

The quickest check that the install works is the minimal example, which loads the power spectra in `data/` and prints the theory normalization of the growth estimator on a few modes:

```
python scripts/minimal_example.py
```

From there, the main entry points are the notebooks and the config-driven scripts.

1. For a worked example, look at the notebooks in `notebooks/paper/`: start with `spectra.ipynb` (theory and reconstruction spectra) and `forecasts.ipynb` (forecasts). These were run on a cluster, so a few input/output paths may need editing, and they need the `[theory]` extra. The `notebooks/example/` folder is still a placeholder.

2. To compute theory curves (normalizations, variances, noise) for a pair of tracers, run a script from inside `scripts/` with a YAML config:

   ```
   cd scripts
   python thy.py --config config_abacus_thy.yaml --config_dir ../configs/abacus/
   ```

   The config sets the biases, number densities, `k`-range and the estimator pairs to compute. See `scripts/README.md` for more examples, including pulling biases and shot noise from an Abacus HOD run (`--config_hod`).

3. To run a forecast, use the same convention:

   ```
   python forecast.py --config config_desi_example_forecast_base.yaml --config_dir ../configs/abacus/
   ```

4. To build your own pipeline, import the package directly. The main modules are `qeep.rec` (reconstruction on a density grid, via `rec.get_rec`), `qeep.qeutils` (kernels, weights, normalizations and variances), and `qeep.fisher` / `qeep.forecast` (Fisher matrix and forecasting).

# Log

The code used for the paper (arxiv: [2510.13803](https://arxiv.org/abs/2510.13803)) is essentially the one in commit [`a64b148`](https://github.com/Saladino93/qeep/commit/a64b148) on the `main` branch, with the relevant notebooks under `notebooks/paper/`. Some of the figures were made with minor offline tweaks that may not be tracked here, so if you want to reproduce something specific and it doesn't match, please get in touch.