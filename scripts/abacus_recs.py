import numpy as np

import jax

from abacusnbody.data.read_abacus import read_asdf
from abacusnbody.analysis import tsc
from abacusnbody.metadata import get_meta
from abacusnbody.hod.zcv.ic_fields import load_dens
from abacusnbody.analysis import power_spectrum as ps

import astropy

from classy import Class

import yaml

import argparse

import os

from scipy.fft import rfftn, irfftn

from pathlib import Path

from qeep import rec_utils as utils, rec
from qeep import qeutils


parser = argparse.ArgumentParser(description='Run reconsturction using configuration from a YAML file.')
parser.add_argument('--config', type=str, help='Path to the YAML configuration file', default='config_abacus.yaml')
parser.add_argument('--config_dir', type=str, help='Path to the configuration directory', default='../configs/abacus/')
args = parser.parse_args()


scratch = "/capstor/scratch/cscs/odarwish/ABACUS/"
ic_dir = "/users/odarwish/scratch/ABACUS/ic/"

with open(args.config_dir + "/" + args.config, 'r') as file:
    config = yaml.safe_load(file)

    
# Extract parameters from config
output_config = config['output']
filename_prefix = output_config['filename_prefix']
output_dir = Path(output_config['directory'])/config['name']

ps_config = config['power_spectrum']

sim_params = config['sim_params']
sim_name_base = sim_params['sim_name_base']
z = sim_params['z_mock']


box = sim_params['box']
nthread = sim_params['nthread']
ngrid = sim_params['ngrid']
interlaced = sim_params['interlaced']
compensated = sim_params['compensated']
paste = sim_params['paste']

sim_name_meta = f"{sim_name_base}{0:03}"
meta = get_meta(sim_name_meta, redshift=z)
z_ic = meta['InitialRedshift']
D_ratio = meta['GrowthTable'][meta['Redshift']] / meta['GrowthTable'][z_ic]

kmin, kmax = config['k_range']['kmin'], config['k_range']['kmax']

keys = config['estimator_keys']
print("We will apply the following estimators:", keys)


W = ps.get_W_compensated(box, ngrid, paste, interlaced)
f = (
    W[:, np.newaxis, np.newaxis]
    * W[np.newaxis, :, np.newaxis]
    * W[np.newaxis, np.newaxis, : (ngrid // 2 + 1)]
)

z_mock = sim_params['z_mock']

knl, pnl = np.loadtxt(ps_config['main_directory']+config['name']+"/"+ps_config['nonlinear']).T
kl, pl = np.loadtxt(ps_config['main_directory']+config['name']+"/"+ps_config['linear']).T

interpolate_function = qeutils.get_interpolated(knl, pnl)
interpolate_function_lin = qeutils.get_interpolated(kl, pl)

samples = sim_params["samples"]


for i in sim_params['sim_list']:


    sim_name = f"{sim_name_base}{i:03}"


    scratch = f"/users/odarwish/scratch/ABACUS/abacus_out/{sim_name}/z{z_mock:.3f}/galaxies/"


    print("Working on simulation index", i)

    out_deltas = [np.load(scratch+f"{kind}_delta_g.npy") for kind in samples]
    out_infos = [np.load(scratch+f"{kind}_out_info.npy", allow_pickle=True).item() for kind in samples]

    #_, kmag = utils.get_kgrid_kmag(box, out_deltas[0].shape[0])
    #del _

    #Ptot_interp = np.interp(kmag, knl, pnl)
    #Plin_interp = np.interp(kmag, kl, pl)

    Ptot_interp = qeutils.get_interpolated(knl, pnl)
    Plin_interp = qeutils.get_interpolated(kl, pl)

    kA = out_infos[0]['k']
    kB = out_infos[1]['k']
    P_AA = qeutils.get_interpolated(kA, out_infos[0]['Ptot'])
    P_BB = qeutils.get_interpolated(kB, out_infos[1]['Ptot'])

    P_AB = qeutils.get_interpolated(kl, out_infos[0]['b1']*out_infos[1]['b1']*pl)
    P_linear = jax.jit(lambda k: interpolate_function_lin(k))


    ic = load_dens(ic_dir, sim_name, ngrid)*D_ratio
    ic_fft = rfftn(ic, overwrite_x=False, workers=nthread).astype(np.complex128)
    ic_fft /= ic.size

    results = {}

    k_values, sim_linear_power = utils.calc_power_mu0_x_axis(
            ic_fft,     # Your FFT field
            BoxSize=box,   # Box size
        )
    
    results["sim_linear_power"] = (k_values, sim_linear_power)
    
    delta_shifted_ffts = [rfftn(delta_shifted, overwrite_x=False, workers=nthread).astype(np.complex128)/delta_shifted.size for delta_shifted in out_deltas]
    
    for sample, delta_shifted_fft in zip(samples, delta_shifted_ffts):

        k_values, sim_nonlinear_power = utils.calc_power_mu0_x_axis(
                delta_shifted_fft/f,     # Your FFT field
                BoxSize=box,   # Box size
            )
        results[f"{sample}_nonlinear_power"] = (k_values, sim_nonlinear_power)
    
    print("Will start reconstructions.")

    keys = ["g", "n"]

    for key in keys:

        temp = {}

        reconstruction = rec.get_rec(key, out_deltas[0]/out_deltas[0].size, box, kmin, kmax, P_AA, P_linear, nthread, out_deltas[1]/out_deltas[1].size, P_BB)

        k_values, auto = utils.calc_power_mu0_x_axis(
            reconstruction,     # Your FFT field
            BoxSize=box,   # Box size
        )

        reconstruction = reconstruction.astype(np.complex128)

        temp["auto"] = (k_values, auto)

        k_values, cross_with_linear = utils.calc_power_mu0_x_axis(
            reconstruction,     # Your FFT field
            BoxSize=box,   # Box size
            delta_k2 = ic_fft
        )

        temp["cross"] = (k_values, cross_with_linear)

        reconstruction = rec.get_rec(key, ic/ic.size, box, kmin, kmax, Ptot_interp, Plin_interp)
        k_values, auto_from_linear = utils.calc_power_mu0_x_axis(
            reconstruction,     # Your FFT field
            BoxSize=box,   # Box size
        )

        temp["auto_from_linear"] = (k_values, auto_from_linear)

        results[key] = temp


    np.save(output_dir / f"results_{filename_prefix}_{i}.npy", results)
