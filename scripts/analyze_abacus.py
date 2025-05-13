import sys
import numpy as np

import matplotlib.pyplot as plt

from abacusnbody.data.read_abacus import read_asdf

import numpy as np

from abacusnbody.analysis.tsc import tsc_parallel

import matplotlib.pyplot as plt

from abacusnbody.analysis.power_spectrum import calc_power, calc_pk_from_deltak, get_k_mu_edges
from abacusnbody.analysis import power_spectrum as ps

from scipy.fft import rfftn, irfftn

import astropy

from classy import Class

import yaml


from abacusnbody.analysis.power_spectrum import calc_power
from abacusnbody.data import read_abacus
from pathlib import Path


scratch = "/capstor/scratch/cscs/odarwish/ABACUS/"

z = 0.5

box = 2000
nthread = 256
ngrid = 512

with open('config_abacus.yaml', 'r') as f:
    config = yaml.safe_load(f)
gen_nl_power = np.loadtxt(config['power_spectrum']['nonlinear'])

output_config = config['output']
output_dir = Path(output_config['directory'])/config['name']
output_dir.mkdir(exist_ok=True, parents=True)

sim_list = config['sim_params']['sim_list']

print("Working on sims: ", sim_list, "with z = ", config['sim_params']['z_mock'], "\n")

for sim_idx in sim_list:

    print(f"Working on sim {sim_idx:03}")

    sim_name_base = config['sim_params']['sim_name_base']
    z_mock = config['sim_params']['z_mock']
    cleaned_halos = config['sim_params']['cleaned_halos']

    print("Reading dark matter field")

    NN = 33
    directory = f"{scratch}/{sim_name_base}{sim_idx:03}/halos/z{z_mock:.3f}/halo_rv_A/"
    pos = [read_asdf(directory+f"halo_rv_A_{i:03}.asdf", ['pos']) for i in range(NN)]
    directory = f"{scratch}/{sim_name_base}{sim_idx:03}/halos/z{z_mock:.3f}/field_rv_A/"
    pos_f = [read_asdf(directory+f"field_rv_A_{i:03}.asdf", ['pos']) for i in range(NN)]
    pos = astropy.table.vstack(pos+pos_f)

    interlaced = True
    compensated = True
    paste = 'TSC'
    nbins_mu = 4
    logk = False
    k_hMpc_max = np.pi * ngrid / box + 1.0e-6
    nbins_k = ngrid // 2
    nthread = 128
    dtype = np.float32

    poles = [0]

    kmin_vol = 2*np.pi/box

    power_alt = calc_power(
        pos["pos"],
        box,
        nbins_k,
        nbins_mu,
        k_hMpc_max,
        logk,
        paste,
        ngrid,
        compensated,
        interlaced,
        poles=poles,
    )

    np.save(output_dir/f'power_alt_{sim_idx:03}.npy', power_alt)



    from abacusnbody.analysis import tsc
    delta = tsc.tsc_parallel(pos['pos'], ngrid, box, nthread=nthread)
    delta /= np.mean(delta, dtype=np.float64)
    delta -= 1.

    from scipy.fft import rfftn, irfftn, fftfreq, rfftfreq
    def get_kgrid(N):
        return 2 * np.pi * np.stack(
            np.meshgrid(
                fftfreq(N, d=box/N), fftfreq(N, d=box/N), rfftfreq(N, d=box/N),
                indexing='ij',
                )
            )

    #k = (fftfreq(nmesh, d=d) * 2.0 * np.pi).astype(np.float32)  # h/Mpc

    W = ps.get_W_compensated(box, ngrid, paste, interlaced)

    kgrid = get_kgrid(delta.shape[0])
    print(kgrid.shape)
    kmag = (kgrid**2).sum(axis=0)**0.5
    print(kmag.shape)

    field_fft = rfftn(delta, overwrite_x=False, workers=nthread)
    field_fft *= 1 / delta.size

    inv_size = dtype(1 / delta.size)
    field_fft = rfftn(delta, overwrite_x=True, workers=nthread)
    ps._normalize(field_fft, inv_size, nthread=nthread)
    field_fft /= (
            W[:, np.newaxis, np.newaxis]
            * W[np.newaxis, :, np.newaxis]
            * W[np.newaxis, np.newaxis, : (ngrid // 2 + 1)]
        )


    poles = np.array([0])
    k_max = np.pi * ngrid / box
    mubins = nbins_mu
    logk = False
    kbins = ngrid
    kbins, mubins = get_k_mu_edges(box, k_max, kbins, mubins, logk)


    power_delta = calc_pk_from_deltak(
        field_fft,
        box,
        kbins,
        mubins,
        field2_fft=None,
        poles=poles,
        squeeze_mu_axis=True,
        nthread=128,
    )

    np.save(output_dir/f'power_delta_{sim_idx:03}.npy', power_delta)

    print("Will do quadratic reconstruction now.")

    import jax
    import jax.numpy as jnp

    Ptot_interp = jnp.interp(kmag, gen_nl_power[:,0], gen_nl_power[:,1])
    gen_power = np.loadtxt(config['power_spectrum']['linear'])
    Plin_interp = jnp.interp(kmag, gen_power[:,0], gen_power[:,1])

    kmin, kmax = config['k_range']['kmin'], config['k_range']['kmax']
    selection = (kmag>=kmin) & (kmag<=kmax)

    delta_A = field_fft*1/Ptot_interp*selection
    delta_B = field_fft*1/Ptot_interp*Plin_interp*selection*delta.size

    delta_A_real = irfftn(delta_A, overwrite_x=True, workers=nthread)
    delta_B_real = irfftn(delta_B, overwrite_x=True, workers=nthread)

    product = delta_A_real*delta_B_real*17/21

    product_fft = rfftn(product, overwrite_x=False, workers=nthread)
    #product_fft *= 1 / product.size

    power_product = calc_pk_from_deltak(
        product_fft,
        box,
        kbins,
        mubins,
        field2_fft=None,
        poles=poles,
        squeeze_mu_axis=True,
        nthread=128,
    )

    np.save(output_dir/f'power_product_{sim_idx:03}.npy', power_product)