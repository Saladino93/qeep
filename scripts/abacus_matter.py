import numpy as np

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
import os


parser = argparse.ArgumentParser(description='Run reconsturction using configuration from a YAML file.')
parser.add_argument('--config', type=str, help='Path to the YAML configuration file', default='../configs/abacus/config_abacus.yaml')
parser.add_argument('--gpu', type=int, help='GPU to use', default=0)
    
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # Only GPU 3 will be visible to JAX
    

scratch = "/capstor/scratch/cscs/odarwish/ABACUS/"
ic_dir = "/users/odarwish/scratch/ABACUS/ic/"

with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

    
# Extract parameters from config
output_config = config['output']
filename_prefix = output_config['filename_prefix']
output_dir = Path(output_config['directory'])/config['name']

ps_config = config['power_spectrum']
#gen_power = np.loadtxt(ps_config['linear'])
#gen_nl_power = np.loadtxt(ps_config['nonlinear'])

sim_params = config['sim_params']
#sim_name = sim_params['sim_name']
z = sim_params['z_mock']


box = sim_params['box']
nthread = sim_params['nthread']
ngrid = sim_params['ngrid']
interlaced = sim_params['interlaced']
compensated = sim_params['compensated']
paste = sim_params['paste']


#meta = get_meta(sim_name, redshift=z)
#z_ic = meta['InitialRedshift']
#D_ratio = meta['GrowthTable'][meta['Redshift']] / meta['GrowthTable'][z_ic]

kmin, kmax = config['k_range']['kmin'], config['k_range']['kmax']

W = ps.get_W_compensated(box, ngrid, paste, interlaced)
f = (
    W[:, np.newaxis, np.newaxis]
    * W[np.newaxis, :, np.newaxis]
    * W[np.newaxis, np.newaxis, : (ngrid // 2 + 1)]
)

for i in sim_params['sim_list']:

    print("Working on simulation index", i)

    out_delta = scratch+f"delta_matter_{i}.npy"

    if os.path.exists(out_delta):

        delta_shifted = np.load(out_delta)

    else:

        print("Building delta from halo positions.")

        out_file = scratch+f"pos_file_{i}.npy"
        if os.path.exists(out_file):
            pos = np.load(out_file)
        else:
            print("Loading halos.")
            NN = 33
            directory = f"{scratch}/AbacusSummit_base_c000_ph{i:03}/halos/z0.500/halo_rv_A/"
            pos = [read_asdf(directory+f"halo_rv_A_{k:03}.asdf", ['pos']) for k in range(NN)]
            directory = f"{scratch}/AbacusSummit_base_c000_ph{i:03}/halos/z0.500/field_rv_A/"
            pos_f = [read_asdf(directory+f"field_rv_A_{k:03}.asdf", ['pos']) for k in range(NN)]
            pos = astropy.table.vstack(pos+pos_f)
            np.save(out_file, pos)

        delta_shifted = tsc.tsc_parallel(pos['pos']+box/2, ngrid, box, nthread = nthread)
        delta_shifted /= np.mean(delta_shifted, dtype = np.float64)
        delta_shifted -= 1.

        np.save(out_delta, delta_shifted)


    """
    _, kmag = utils.get_kgrid_kmag(box, delta_shifted.shape[0])
    del _

    Ptot_interp = np.interp(kmag, gen_nl_power[:,0], gen_nl_power[:,1])
    Plin_interp = np.interp(kmag, gen_power[:,0], gen_power[:,1])


    sim_name = f"{sim_name_base}{i:03}"
    ic = load_dens(ic_dir, sim_name, ngrid)*D_ratio
    ic_fft = rfftn(ic, overwrite_x=False, workers=nthread).astype(np.complex128)
    ic_fft /= ic.size

    results = {}

    k_values, sim_linear_power = utils.calc_power_mu0_x_axis(
            ic_fft,     # Your FFT field
            BoxSize=box,   # Box size
        )
    
    results["sim_linear_power"] = (k_values, sim_linear_power)
    
    delta_shifted_fft = rfftn(delta_shifted, overwrite_x=False, workers=nthread).astype(np.complex128)
    delta_shifted_fft /= delta_shifted.size
    k_values, sim_nonlinear_power = utils.calc_power_mu0_x_axis(
            delta_shifted_fft/f,     # Your FFT field
            BoxSize=box,   # Box size
        )
    
    results["sim_nonlinear_power"] = (k_values, sim_nonlinear_power)
    
    print("Will start reconstructions.")

    for key in keys:

        temp = {}

        reconstruction = rec.get_rec(key, delta_shifted, box, kmin, kmax, Ptot_interp, Plin_interp, nthread)

        k_values, auto = utils.calc_power_mu0_x_axis(
            reconstruction,     # Your FFT field
            BoxSize=box,   # Box size
        )

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
    """
