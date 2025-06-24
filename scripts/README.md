# Scripts

## Quick start

To get some theory curves with custom biases and shot noises:

`python thy.py --config $name_config_file.yaml$ --config_dir $name_config_dir$`

Example:

`python thy.py --config config_abacus_thy.yaml --config_dir ../configs/abacus/`


If you want to include results from Abacus simulations, then:

`python thy.py --config $name_config_file.yaml$ --config_dir $name_config_dir$ --config_hod $name_config_hod.yaml$`

To run results with abacus simulations

### Running with Abacus

#### Galaxies

To run with Abacus galaxies, first you need to populate halos with some HOD.

Following https://abacusutils.readthedocs.io/en/latest/hod.html these are the steps:

* `python -m abacusnbody.hod.prepare_sim --path2config PATH2CONFIG`
* `python /users/odarwish/abacusutils/scripts/hod/run_hod.py --path2config PATH2CONFIG`

Then, you need to run some preparation steps for the ZCV method. This is needed to get bias parameters (but you can use other methods) useful for theory predictions.

Following https://abacusutils.readthedocs.io/en/latest/tutorials/analysis/zcv.html:

* `python -m abacusnbody.hod.zcv.zenbu_window --path2config PATH2CONFIG`
* `python -m abacusnbody.hod.zcv.ic_fields --path2config PATH2CONFIG`
* `python -m abacusnbody.hod.zcv.advect_fields --path2config PATH2CONFIG`


## Forecasts