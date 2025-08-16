import argparse

from abacusnbody.analysis import tsc
from abacusnbody.analysis import power_spectrum as ps
from abacusnbody.metadata import get_meta

import yaml

from astropy.io import ascii

from classy import Class

from qeep import rec_utils as ru

import numpy as np

from scipy.fft import rfftn, irfftn

from abacusnbody.hod.abacus_hod import AbacusHOD


def main(config_path, config_path_hod):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    with open(config_path_hod, 'r') as f:
        config_hod = yaml.safe_load(f)

    sim_name = config_hod['sim_params']['sim_name']
    z_mock = config_hod['sim_params']['z_mock']

    samples = config['sim_params']['samples']
    box = config['sim_params']['box']
    interlaced = config['sim_params']['interlaced']
    compensated = config['sim_params']['compensated']
    paste = config['sim_params']['paste']
    ngrid = config['sim_params']['ngrid']
    nthread = config['sim_params']['nthread']

    ps_main_directory = config['power_spectrum']['main_directory']
    name_config = config['name']
    gen_nl_power = np.loadtxt(ps_main_directory+name_config+"/"+config['power_spectrum']['nonlinear'])
    gen_power = np.loadtxt(ps_main_directory+name_config+"/"+config['power_spectrum']['linear'])

    scratch = f"/users/odarwish/scratch/ABACUS/abacus_out/{sim_name}/z{z_mock:.3f}/galaxies/"

    print(f"Working on sim {sim_name}")

    for sample in samples:

        out_info = {}

        print(f"Working on sample {sample}")
        name = scratch+f"{sample}s.dat"
        galaxies = ascii.read(name)

        N = galaxies['x'].size
        nbar = N/(box**3)


        pos = np.vstack([galaxies['x'], galaxies['y'], galaxies['z']]).T

        delta_g = tsc.tsc_parallel(pos+box/2, ngrid, box, nthread=nthread)  #need to shift by box/2 to correlate with input field
        delta_g /= np.mean(delta_g, dtype=np.float64)
        delta_g -= 1.

        np.save(scratch+f"{sample}_delta_g.npy", delta_g)

        field_fft_g = rfftn(delta_g, overwrite_x=False, workers=nthread)
        field_fft_g *= 1 / delta_g.size

        W = ps.get_W_compensated(box, ngrid, paste, interlaced)
        f = (
            W[:, np.newaxis, np.newaxis]
            * W[np.newaxis, :, np.newaxis]
            * W[np.newaxis, np.newaxis, : (ngrid // 2 + 1)]
        )

        sim_params = config_hod['sim_params']
        HOD_params = config_hod['HOD_params']
        clustering_params = config_hod['clustering_params']
        zcv_params = config_hod['zcv_params']

        for k in HOD_params["tracer_flags"].keys():
            HOD_params["tracer_flags"][k] = True
        #HOD_params["tracer_flags"][sample] = True

        newBall = AbacusHOD(sim_params, HOD_params, clustering_params)

        mock_dict = {}
        mock_dict[sample] = galaxies

        for k in config_hod['HOD_params']["tracer_flags"].keys():
            config_hod['HOD_params']["tracer_flags"][k] = False
        config_hod['HOD_params']["tracer_flags"][sample] = True

        load_presaved = False
        zcv_dict = newBall.apply_zcv(mock_dict, config_hod, load_presaved=load_presaved)


        sim_name = config_hod['sim_params']['sim_name']
        z = config_hod['sim_params']['z_mock']
        ztarget = z

        meta = get_meta(sim_name, redshift=z)
        Lbox = meta['BoxSize']
        z_ic = meta['InitialRedshift']
        Ndim = int(meta['ppd'])
        
        Dz = meta['GrowthTable']
        
        pk = meta['CLASS_power_spectrum']
        kk  = pk['k (h/Mpc)']
        
        zpk = meta['ZD_Pk_file_redshift']  # 1.0
        pk = meta['CLASS_power_spectrum']
        k  = pk['k (h/Mpc)']
        input_pk = pk['P (Mpc/h)^3']
        pkk = input_pk * (Dz[ztarget] / Dz[zpk])**2

        k_max = 20.0
        n_points = 2000
        k_min = 1e-4
        
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

        a1, a2 = 1, -17/21
        b_1 = 1+zcv_dict['bias'][0]
        b_2 = zcv_dict['bias'][1]
        b_s = zcv_dict['bias'][2]

        b_2_eul =  2*(a1 + a2)*zcv_dict['bias'][0]+ a2*zcv_dict['bias'][1] #from bias review, also check Abidi and Baldauf, 2.32
        b_s_eul = -2/7*zcv_dict['bias'][0]+zcv_dict['bias'][2]

        out_info['b1'] = b_1
        out_info['b2'] = b_2_eul
        out_info['bs'] = b_s_eul
        out_info['b2_L'] = b_2
        out_info['bs_L'] = b_s
        out_info['nbar'] = nbar
        out_info['z'] = z
        out_info['Ptot'] = gen_nl_power[:,1]*b_1**2+1/nbar
        out_info['Ptot_L'] = gen_power[:,1]*b_1**2+1/nbar
        out_info['PNL'] = gen_nl_power[:,1]
        out_info['PL'] = gen_power[:,1]
        out_info['k'] = gen_nl_power[:,0]

        k_values, power = ru.calc_power_mu0_x_axis(
            field_fft_g/f, 
            BoxSize=box,
        )

        out_info['Pmeasured'] = power
        out_info['k_measured'] = k_values


        np.save(scratch+f"{sample}_out_info.npy", out_info)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--config_hod", type=str, required=True)
    parser.add_argument("--config_dir", type=str, default = "../configs/abacus/")
    args = parser.parse_args()
    config = args.config_dir + "/" + args.config
    config_hod = args.config_dir + "/" + args.config_hod
    main(config, config_hod)

        
        










   
