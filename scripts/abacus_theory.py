from qeep import qeutils
import jax

import numpy as np
import sympy as sp
import sympy2jax
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm


def run_analysis(config_path, config_path_hod):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    with open(config_path_hod, 'r') as file:
        config_hod = yaml.safe_load(file)
    
    # Extract parameters from config
    ps_config = config['power_spectrum']

    kr_config = config['k_range']
    sampling_config = config['sampling']
    output_config = config['output']

    sim_params = config_hod['sim_params']

    skip_variance = config['skip_variance']
    
    # Load power spectrum data
    knl, pnl = np.loadtxt(ps_config['main_directory']+config['name']+"/"+ps_config['nonlinear']).T
    kl, pl = np.loadtxt(ps_config['main_directory']+config['name']+"/"+ps_config['linear']).T
    
    interpolate_function = qeutils.get_interpolated(knl, pnl)
    interpolate_function_lin = qeutils.get_interpolated(kl, pl)
    
    q1, q2, mu = sp.symbols('q1 q2 mu')
    
    sim_name_base = config['sim_params']['sim_name_base']


    for sim_index in config['sim_params']['sim_list']:
        print(f"Processing simulation {sim_index}")

        sim_name = sim_name_base + f"{sim_index:03}"
        z_mock = sim_params['z_mock']
        scratch = f"/users/odarwish/scratch/ABACUS/abacus_out/{sim_name}/z{z_mock:.3f}/galaxies/"

        samples = config["sim_params"]["samples"]
        
        out_info_A = np.load(scratch+f"{samples[0]}_out_info.npy", allow_pickle=True).item()
        out_info_B = np.load(scratch+f"{samples[1]}_out_info.npy", allow_pickle=True).item()
        
        b10_A = out_info_A['b1']
        b10_B = out_info_B['b1']
        b20_A = out_info_A['b2']
        b20_B = out_info_B['b2']
        bs2_A = out_info_A['bs']
        bs2_B = out_info_B['bs']

        # Number density parameters
        nbar_A = out_info_A['nbar']
        nbar_B = out_info_B['nbar']
        
        kA = out_info_A['k']
        kB = out_info_B['k']
        P_AA = qeutils.get_interpolated(kA, out_info_A['Ptot'])
        P_BB = qeutils.get_interpolated(kB, out_info_B['Ptot'])

        P_AB = qeutils.get_interpolated(kl, b10_A*b10_B*pl)
        P_linear = jax.jit(lambda k: interpolate_function_lin(k))

        # Keypairs and estimator keys
        keypairs = config['keypairs']
        estimator_keys = config['estimator_keys']
        
        # k-range parameters
        kmin = kr_config['kmin']
        kmax = kr_config['kmax']
        k_samples = kr_config['k_samples']
        k_min_analysis = kr_config['k_min_analysis']
        k_max_analysis = kr_config['k_max_analysis']
        
        #Ks = np.linspace(k_min_analysis, k_max_analysis, k_samples)
        Ks = np.logspace(np.log10(k_min_analysis), np.log10(k_max_analysis), k_samples)
        
        # Sampling parameters
        Nsamples_base = sampling_config['Nsamples_base']
        
        # Define estimator configs
        estimator_configs = {
            'g': {
                'F': 17/21*q1/q1,
                'ca': 1., 
                'cb': 1.
            },
            'ga': { #antisymmetric part of g
                'F': 17/21*q1/q1,
                'ca': 1., 
                'cb': -1.
            },
            's': {
                'F': 0.5*(q2/q1+q1/q2)*mu,
                'ca': 1, 
                'cb': 1
            },
            'sa': { #antisymmetric part of s
                'F': 0.5*(q2/q1+q1/q2)*mu,
                'ca': 1, 
                'cb': -1
            },
            't': {
                'F': (2./7.)*(mu**2.-1./3.),
                'ca': 1., 
                'cb': 1.
            },
            'ta': { #antisymmetric part of t
                'F': (2./7.)*(mu**2.-1./3.),
                'ca': 1., 
                'cb': -1.
            },
            'n': {
                'F': mu*q2/q1,
                'ca': 1, 
                'cb': 0
            }
        }
        
        estimator_lam_jax = {key: sympy2jax.SymbolicModule(estimator_configs[key]['F']) for key in estimator_configs}
        
        f_jax = {key: qeutils.get_f(estimator_lam_jax[key], P_linear, estimator_configs[key]["ca"], estimator_configs[key]["cb"]) for key in estimator_lam_jax}
        
        # Prepare output directory
        output_dir = Path(output_config['directory'])/config['name']
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Run analysis
        out_normalization_AB = {}
        out_variance_AB = {}
        out_cross_shot_AB = {}

        out_normalization_BA = {}
        out_variance_BA = {}
        out_cross_shot_BA = {}
        
        # Add progress bar for keypair calculations
        print("Running analysis for each keypair...")
        pbar = tqdm(total=len(keypairs), desc="Processing keypairs")
        for keypair in keypairs:
            key1, key2 = keypair
            # Update the progress bar to show current keypair
            pbar.set_description(f"Processing ({key1}, {key2})")
            
            N_single_AB = qeutils.N_per_mode(f_jax[key1], f_jax[key2], P_AA, P_BB, kmin, kmax, Nsamples_base=Nsamples_base, gauss_filter=False)
            out_normalization_AB[tuple(keypair)] = qeutils.integrate(Ks, N_single_AB, batch_size=2)
            out_normalization_AB[(key2, key1)] = out_normalization_AB[tuple(keypair)]

            w_A = qeutils.get_w(f_jax[key1], P_AA, P_BB)

            # N_single_AB_weighted = qeutils.N_per_mode_weighted(w_A, f_jax[key2], kmin, kmax, Nsamples_base=Nsamples_base, gauss_filter = False)
            #print(out_normalization_AB[tuple(keypair)]/qeutils.integrate(Ks, N_single_AB_weighted, batch_size=2))

            cross_single_AB = qeutils.cross_shot_mixed_AAB(w_A, nbar_A, P_AB, kmin=kmin, kmax=kmax, Nsamples_base=Nsamples_base)
            out_cross_shot_AB[tuple(keypair)] = qeutils.integrate(Ks, cross_single_AB, batch_size=2)
            out_cross_shot_AB[(key2, key1)] = out_cross_shot_AB[tuple(keypair)]

            w_B = qeutils.get_w(f_jax[key2], P_AA, P_BB)
            #AB-XY = AB-AB
            #
            #variance_single_AB = qeutils.variance_per_mode(w_A, w_B, P_linear, P_linear, P_linear, P_linear, kmin, kmax, Nsamples_base=Nsamples_base//20)
            if not skip_variance:
                variance_single_AB = qeutils.variance_per_mode(w_A, w_B, P_AA, P_BB, P_AB, P_AB, kmin, kmax, Nsamples_base=Nsamples_base//15, gauss_filter = False)
                out_variance_AB[tuple(keypair)] = qeutils.integrate(Ks, variance_single_AB, batch_size=2)
                out_variance_AB[(key2, key1)] = out_variance_AB[tuple(keypair)]
            # Update progress
            pbar.update(1)
        
        pbar.close()

        
        # Save results to files
        filename_prefix = output_config['filename_prefix']+f"{sim_name}_z{z_mock:.3f}_{samples[0]}_{samples[1]}"
            
        # Save as NPY files
        np.save(output_dir / f"{filename_prefix}_normalization_AB.npy", out_normalization_AB)
        np.save(output_dir / f"{filename_prefix}_cross_shot_AB.npy", out_cross_shot_AB)
        np.save(output_dir / f"{filename_prefix}_variance_AB.npy", out_variance_AB)

        np.save(output_dir / f"{filename_prefix}_Ks.npy", Ks)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run noise and bias analysis using configuration from a YAML file.')
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file', default='config_abacus.yaml')
    parser.add_argument('--config_hod', type=str, help='Path to the HOD configuration file', default='config_hod_0.yaml')
    parser.add_argument('--config_dir', type=str, help='Path to the configuration directory', default='../configs/abacus/')
    args = parser.parse_args()
    
    run_analysis(args.config_dir + "/" + args.config, args.config_dir + "/" + args.config_hod)

