"""
Theory code
"""

#import sys
#sys.path.append('/users/odarwish/lenscarf/lib/python3.12/site-packages')
#sys.path.append('/users/odarwish/qeep/')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Only GPU 3 will be visible to JAX

from qeep import qeutils
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import sympy as sp
import sympy2jax
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm


def get_kernels():

    K, q1, q2, mu = sp.symbols('K q1 q2 mu')

    estimator_configs = {
            'g': {
                'F': 17/21*q1/q1,
                'ca': 1., 
                'cb': 1.
            },
            'ga': { #antisymmetric part of g
                'F': 17/21*q1/q1,
                'ca': -1., 
                'cb': +1.
            },
            's': {
                'F': 0.5*(q2/q1+q1/q2)*mu,
                'ca': 1, 
                'cb': 1
            },
            'sa': { #antisymmetric part of s
                'F': 0.5*(q2/q1+q1/q2)*mu,
                #F(k1+k2, -k1), F(k1+k2, -k2), but here q1 and q2 are the arguments of F
                #in my case, q1 = K, the long-mode and q2 is the small-scale mode = k1
                #hence k2 = K-k1
                'ca': -1, 
                'cb': +1
            },
            't': {
                'F': (2./7.)*(mu**2.-1./3.),
                'ca': 1., 
                'cb': 1.
            },
            'ta': { #antisymmetric part of t
                'F': (2./7.)*(mu**2.-1./3.),
                'ca': -1., 
                'cb': +1.
            },
            'n': {
                #'F': -mu*q2/q1, 
                'F': 2*mu*q1/K, #k short/k long, so q1 is the small-scale mode = k, and K is the long-mode
                'ca': 1,
                'cb': 0
            }
        }
    
    return estimator_configs


def run_analysis(config_path, config_path_hod):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract parameters from config
    ps_config = config['power_spectrum']

    kr_config = config['k_range']
    sampling_config = config['sampling']
    output_config = config['output']
    
    # Load power spectrum data
    knl, pnl = np.loadtxt(ps_config['main_directory']+config['name']+"/"+ps_config['nonlinear']).T
    kl, pl = np.loadtxt(ps_config['main_directory']+config['name']+"/"+ps_config['linear']).T
    
    interpolate_function = qeutils.get_interpolated(knl, pnl)
    interpolate_function_lin = qeutils.get_interpolated(kl, pl)
    
    sim_name_base = config['sim_params']['sim_name_base']


    for sim_index in config['sim_params']['sim_list']:
        print(f"Processing simulation {sim_index}")

        sim_name = sim_name_base + f"{sim_index:03}"


        #if we have a HOD config, we use it to get the bias and shot noise parameters
        if config_path_hod is not None:
            with open(config_path_hod, 'r') as file:
                config_hod = yaml.safe_load(file)
            sim_params = config_hod['sim_params']
            z_mock = sim_params['z_mock']
            scratch = f"/users/odarwish/scratch/ABACUS/abacus_out/{sim_name}/z{z_mock:.3f}/galaxies/"

            samples = config["sim_params"]["samples"]

            same_tracers = (samples[0] == samples[1])
        
            if samples[0] == "matter":
                b10_A = 1
                b20_A = 0.
                bs2_A = 0.
                nbar_A = 1000
                Ptot_A = b10_A**2*pnl
                kA = kl
            else:
                out_info_A = np.load(scratch+f"{samples[0]}_out_info.npy", allow_pickle=True).item()
                b10_A = out_info_A['b1']
                b20_A = out_info_A['b2']
                bs2_A = out_info_A['bs']
                # Number density parameters
                nbar_A = out_info_A['nbar']
                Ptot_A = out_info_A['Ptot']
                kA = out_info_A['k']

            if samples[1] == "matter":
                b10_B = 1
                b20_B = 0.
                bs2_B = 0.
                nbar_B = 1000
                Ptot_B = b10_B**2*pnl
                kB = kl
            else:
                out_info_B = np.load(scratch+f"{samples[1]}_out_info.npy", allow_pickle=True).item()
                b10_B = out_info_B['b1']            
                b20_B = out_info_B['b2']            
                bs2_B = out_info_B['bs']
                nbar_B = out_info_B['nbar']
                Ptot_B = out_info_B['Ptot']
                kB = out_info_B['k']

            # Save results to files
            filename_prefix = output_config['filename_prefix']+f"_{sim_name}_z{z_mock:.3f}_{samples[0]}_{samples[1]}"

            bs2_fid = bs2_A #qeutils.bs2_coev(b10_A)
            b2_fid = b20_A #qeutils.b2_fid(b10_A)

            print("Comparing b20s (sim vs theory), A and B", "A", b20_A, "theory", qeutils.b2_fid(b10_A), "B", b20_B, "theory", qeutils.b2_fid(b10_B))
            print("Comparing bs2s (sim vs theory), A and B", "A", bs2_A, "theory", qeutils.bs2_coev(b10_A), "B", bs2_B, "theory", qeutils.bs2_coev(b10_B))

        else:

            biases_config = config['bias']
            nbar_config = config['number_density']

            b10_A = biases_config['b10_A']
            nbar_A = nbar_config['nbar_A']

            b10_B = biases_config['b10_B']
            nbar_B = nbar_config['nbar_B']
            
            same_tracers = (b10_A == b10_B) and (nbar_A == nbar_B)

            nshot_A = 1/nbar_A
            nshot_B = 1/nbar_B

            Ptot_A = b10_A**2*pnl+nshot_A
            Ptot_B = b10_B**2*pnl+nshot_B

            kA, kB = kl, kl

            filename_prefix = output_config['filename_prefix']+"_theory"

            bs2_fid_A = qeutils.bs2_coev(b10_A)
            b2_fid_A = qeutils.b2_fid(b10_A) if "b2_A" not in biases_config else biases_config['b2_A']

            bs2_fid_B = qeutils.bs2_coev(b10_B)
            b2_fid_B = qeutils.b2_fid(b10_B) if "b2_B" not in biases_config else biases_config['b2_B']

            b2_fid = b2_fid_A
            bs2_fid = bs2_fid_A 

        #b2_fid, bs2_fid are used in the bispectrum for the shot-trispectrum calculations. Ideally, you would want a cross-trispectrum.
        #For computational ease, I just insert the trispectrum for a single tracer here.

        P_AA = qeutils.get_interpolated(kA, Ptot_A)
        P_BB = qeutils.get_interpolated(kB, Ptot_B)

        P_AB = P_AA if same_tracers else qeutils.get_interpolated(kl, b10_A*b10_B*pnl) #here is non-linear
        P_linear = jax.jit(lambda k: interpolate_function_lin(k))

        P_AA_signal = qeutils.get_interpolated(kA, b10_A**2*pl) #needs to be linear
        P_A_signal = qeutils.get_interpolated(kA, b10_A*pl)
        P_BB_signal = qeutils.get_interpolated(kB, b10_B**2*pl)
        P_AB_signal = qeutils.get_interpolated(kl, b10_A*b10_B*pl) 

        # Keypairs and estimator keys
        keypairs = config['keypairs'] #this is where we loop over the keypairs
        
        # k-range parameters
        kmin = kr_config['kmin']
        kmax = kr_config['kmax']
        k_samples = kr_config['k_samples']
        k_min_analysis = kr_config['k_min_analysis']
        k_max_analysis = kr_config['k_max_analysis']
        
        #define modes on which we reconstruct
        #Ks = np.linspace(k_min_analysis, k_max_analysis, k_samples)
        kmin_max = 2*k_min_analysis
        Ks_ = np.linspace(k_min_analysis, kmin_max, 20)
        Ks = np.logspace(np.log10(kmin_max), np.log10(k_max_analysis), k_samples)
        Ks = np.concatenate([Ks_, Ks])
        Ks = np.unique(Ks)
        
        # Sampling parameters for Monte Carlo integration
        Nsamples_base = sampling_config['Nsamples_base']        

        estimator_configs = get_kernels()
        
        estimator_lam_jax = {key: sympy2jax.SymbolicModule(estimator_configs[key]['F']) for key in estimator_configs}
        
        f_jax = {key: qeutils.get_f(estimator_lam_jax[key], P_linear, estimator_configs[key]["ca"], estimator_configs[key]["cb"]) for key in estimator_lam_jax if key != "n"}
        #f_jax["n"] = qeutils.get_f(estimator_lam_jax["n"], P_linear, estimator_configs["n"]["ca"], estimator_configs["n"]["cb"]) 
        f_jax["n"] = qeutils.get_f_squeezed(estimator_lam_jax["n"], P_linear)
        
        Fkernels = [qeutils.Fg, qeutils.Fs, qeutils.Ft]
        Fbiases = [qeutils.bias_g(b10_A, b2_fid), qeutils.bias_s(b10_A), qeutils.bias_t(b10_A, bs2_fid)]

        # Prepare output directory
        output_dir = Path(output_config['directory'])/config['name']
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Run analysis
        out_normalization_AB = {} #this is the inverse of the normalization factor for the estimator
        out_variance_AB = {} #this is the variance of the estimator
        out_cross_shot_AB = {} #this is the cross-shot noise
        out_shot_bispectrum = {} #this is the shot noise of the bispectrum, but assuming one single tracer
        out_shot_trispectrum = {} #this is the shot noise of the trispectrum, but assuming one single tracer

        out_normalization_BA = {} #this is the inverse of the normalization factor for the estimator, swapping A and B
        out_variance_BA = {} #this is the variance of the estimator, swapping A and B
        out_cross_shot_BA = {} #this is the cross-shot noise, swapping A and B


        P_A_signal_jax = jax.jit(lambda k: P_A_signal(k))
        #P_BB_signal_jax = jax.jit(lambda k: P_BB_signal(k))
        #P_AB_signal_jax = jax.jit(lambda k: P_AB_signal(k))
        bispectrum_cont = qeutils.get_bispectrum_XYZ(P_A_signal_jax, P_A_signal_jax, P_A_signal_jax, Fkernels, Fbiases)
        
        # Add progress bar for keypair calculations
        print("Running analysis for each keypair...")
        pbar = tqdm(total=len(keypairs), desc="Processing keypairs")
        for keypair in keypairs:
            keypair, quantity = keypair
            key1, key2 = keypair
            # Update the progress bar to show current keypair
            pbar.set_description(f"Processing ({key1}, {key2})")


            if "Nfull" in quantity:
                w_A = qeutils.get_full_w(f_jax[key1], P_AA, P_BB, P_AB)
            else:
                w_A = qeutils.get_w(f_jax[key1], P_AA, P_BB)
            
            if "N" in quantity or "all" in quantity or "Nfull" in quantity:

                if "full" not in quantity:#in principle I could just use N_per_mode_weighted
                    N_single_AB = qeutils.N_per_mode(f_jax[key1], f_jax[key2], P_AA, P_BB, kmin, kmax, Nsamples_base=Nsamples_base, gauss_filter=False)
                    result_N = qeutils.integrate(Ks, N_single_AB, batch_size=2)
                else:
                    N_per_mode_weighted = qeutils.N_per_mode_weighted(w_A, f_jax[key2], kmin, kmax, Nsamples_base=Nsamples_base, gauss_filter = False)
                    result_N = qeutils.integrate(Ks, N_per_mode_weighted, batch_size=2)

                out_normalization_AB[tuple(keypair)] = result_N
                out_normalization_AB[(key2, key1)] = out_normalization_AB[tuple(keypair)]


            # N_single_AB_weighted = qeutils.N_per_mode_weighted(w_A, f_jax[key2], kmin, kmax, Nsamples_base=Nsamples_base, gauss_filter = False)
            #print(out_normalization_AB[tuple(keypair)]/qeutils.integrate(Ks, N_single_AB_weighted, batch_size=2))

            if "Bcross" in quantity or "all" in quantity:
                cross_single_AB = qeutils.cross_shot_mixed_AAB(w_A, nbar_A, P_AB_signal, kmin=kmin, kmax=kmax, Nsamples_base=Nsamples_base)
                out_cross_shot_AB[tuple(keypair)] = qeutils.integrate(Ks, cross_single_AB, batch_size=2)
                out_cross_shot_AB[(key2, key1)] = out_cross_shot_AB[tuple(keypair)]

            if "Nfull" in quantity:
                w_B = qeutils.get_full_w(f_jax[key2], P_AA, P_BB, P_AB)
            else:
                w_B = qeutils.get_w(f_jax[key2], P_AA, P_BB)


            if "T" in quantity or "all" in quantity:
                torchquad = True
                shot_trispectrum = qeutils.shot_trispectrum(w_A, w_B, P_AA_signal, bispectrum_cont, nbar_A, kmin, kmax, Nsamples_base=400//20, torchquad = True)
                shot_result = qeutils.integrate(Ks, shot_trispectrum, batch_size=2) if not torchquad else qeutils.integrate_vegas(Ks, shot_trispectrum)
                print("Done with shot trispectrum")
                out_shot_trispectrum[tuple(keypair)] = shot_result
                out_shot_trispectrum[(key2, key1)] = out_shot_trispectrum[tuple(keypair)]

            if "B" in quantity or "all" in quantity:
                shot_bispectrum = qeutils.shot_bispectrum(w_A, nbar_A, P_AA_signal, kmin, kmax, Nsamples_base=Nsamples_base)
                shot_bispectrum_result = qeutils.integrate(Ks, shot_bispectrum, batch_size=2)

                #shot_bispectrum_alternative = qeutils.shot_bispectrum_alternative(w_A, nbar_A, P_AA_signal, kmin, kmax, Nsamples_base=Nsamples_base, Norm_K = lambda K: jnp.interp(K, Ks, out_normalization_AB[tuple(keypair)]**-1.))
                #shot_bispectrum_alternative_result = qeutils.integrate(Ks, shot_bispectrum_alternative, batch_size=2)
                #out_shot_bispectrum[tuple(keypair)] = shot_bispectrum_result
                #print("shot_bispectrum_alternative_result", shot_bispectrum_alternative_result/shot_bispectrum_result)

                out_shot_bispectrum[tuple(keypair)] = shot_bispectrum_result
                out_shot_bispectrum[(key2, key1)] = out_shot_bispectrum[tuple(keypair)]

            #AB-XY = AB-AB
            #
            #variance_single_AB = qeutils.variance_per_mode(w_A, w_B, P_linear, P_linear, P_linear, P_linear, kmin, kmax, Nsamples_base=Nsamples_base//20)
            if "V" in quantity or "all" in quantity or "Vfull" in quantity:
                #variance_single_AB_fast = qeutils.variance_per_mode(w_A, w_B, P_AA, P_BB, P_AB, P_AB, kmin, kmax, Nsamples_base=8000//15, gauss_filter = False)
                variance_single_AB_fast = qeutils.variance_per_mode_fast(w_A, w_B, P_AA, P_BB, P_AB, P_AB, kmin, kmax, Nsamples_base=8000//15, gauss_filter = False)
                out_variance_AB[tuple(keypair)] = qeutils.integrate(Ks, variance_single_AB_fast, batch_size=2)
                out_variance_AB[(key2, key1)] = out_variance_AB[tuple(keypair)]
            # Update progress
            pbar.update(1)
        
        pbar.close()
            
        # Save as NPY files
        np.save(output_dir / f"{filename_prefix}_normalization_AB.npy", out_normalization_AB)
        np.save(output_dir / f"{filename_prefix}_cross_shot_AB.npy", out_cross_shot_AB)
        np.save(output_dir / f"{filename_prefix}_variance_AB.npy", out_variance_AB)
        np.save(output_dir / f"{filename_prefix}_shot_trispectrum_AB.npy", out_shot_trispectrum)
        np.save(output_dir / f"{filename_prefix}_shot_bispectrum_AB.npy", out_shot_bispectrum)
        np.save(output_dir / f"{filename_prefix}_Ks.npy", Ks)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run noise and bias analysis using configuration from a YAML file.')
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file', default='config_abacus_thy.yaml')
    parser.add_argument('--config_hod', type=str, help='Path to the HOD configuration file', default = None) #default='config_hod_0.yaml')
    parser.add_argument('--config_dir', type=str, help='Path to the configuration directory', default='../configs/abacus/')
    args = parser.parse_args()
    
    run_analysis(args.config_dir + "/" + args.config, args.config_dir + "/" + args.config_hod if args.config_hod is not None else None)

