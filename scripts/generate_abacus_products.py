from classy import Class
from abacusnbody.metadata import get_meta

import yaml

import numpy as np

with open('config_abacus.yaml', 'r') as f:
    config = yaml.safe_load(f)

sim_name = config['sim_params']['sim_name']
z = config['sim_params']['z_mock']

meta = get_meta(sim_name, redshift=z)
Lbox = meta['BoxSize']
z_ic = meta['InitialRedshift']
Ndim = int(meta['ppd'])

k_max = 20.0
n_points = 2000
k_min = 1e-4

# set up cosmology
boltz = Class()
cosmo = {}
cosmo['output'] = 'mPk'
cosmo['P_k_max_h/Mpc'] = k_max
int(sim_name.split('ph')[-1])
for k in (
        'H0',
        'omega_b',
        'omega_cdm',
        'omega_ncdm',
        'N_ncdm',
        'N_ur',
        'n_s',
        'A_s',
        'alpha_s',
        #'wa', 'w0',
):
    cosmo[k] = meta[k]
cosmo["z_max_pk"] = z+0.1
boltz.set(cosmo)
boltz.compute()

# Create logarithmically spaced k values
k_arr = np.logspace(np.log10(k_min), np.log10(k_max), n_points)
pk_arr = np.zeros_like(k_arr)

# Fill the power spectrum array
for i, k in enumerate(k_arr):
    pk_arr[i] = boltz.pk(k * boltz.h(), z)*boltz.h()**3  # k in 1/Mpc, z is the redshift

np.savetxt(config['power_spectrum']['linear'], np.vstack((k_arr, pk_arr)).T)

cosmo['non linear'] = 'halofit' 
boltz.set(cosmo)
boltz.compute()

# Fill the power spectrum array
for i, k in enumerate(k_arr):
    pk_arr[i] = boltz.pk(k * boltz.h(), z)*boltz.h()**3  # k in 1/Mpc, z is the redshift

np.savetxt(config['power_spectrum']['nonlinear'], np.vstack((k_arr, pk_arr)).T)
