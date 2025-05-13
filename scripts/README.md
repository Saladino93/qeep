# Scripts

This directory contains Python scripts for quadratic estimator analyses:

- `abacus_galaxies.py`, process galaxy catalogs generated from Abacaus (see below)

- **run_recs.py**: Run quadratic estimator reconstruction algorithms
- **analyze_abacus.py**: Tools for analyzing quadratic estimator results on Abacus data
- **generate_abacus_products.py**: Generate products for the Abacus simulations, such as power spectra, to cross-check results
- **run_theory.py**: Calculates normalization, variance, shot reconstruction noise


## Running with Abacus

### Galaxies

To run with Abacus galaxies, first you need to populate with some HOD. Following https://abacusutils.readthedocs.io/en/latest/hod.html these are the steps:

* `python -m abacusnbody.hod.prepare_sim --path2config PATH2CONFIG`
* `python /users/odarwish/abacusutils/scripts/hod/run_hod.py --path2config PATH2CONFIG`

Then, you need to run some preparation steps for the ZCV method.

Following https://abacusutils.readthedocs.io/en/latest/tutorials/analysis/zcv.html:

* `python -m abacusnbody.hod.zcv.zenbu_window --path2config PATH2CONFIG`
* `python -m abacusnbody.hod.zcv.ic_fields --path2config PATH2CONFIG`
* `python -m abacusnbody.hod.zcv.advect_fields --path2config PATH2CONFIG`



### Matter