"""
Theory code
"""
#import sys
#sys.path.append('/users/odarwish/lenscarf/lib/python3.12/site-packages')
#sys.path.append('/users/odarwish/qeep/')

from qeep import qeutils
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
import sympy as sp
from sympy.utilities.lambdify  import implemented_function
import sympy2jax
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil



def get_quick_M(z = 0.5):
    direc = "/users/odarwish/qeep/notebooks/paper/desi_abacus/data_dir/"
    k, M = np.loadtxt(direc+"M.txt").T
    return qeutils.get_interpolated(k, M)

def get_quick_M_(z = 0.5):
    from abacusnbody.metadata import get_meta

    sim_name = "AbacusSummit_base_c000_ph000"
    meta = get_meta(sim_name, redshift=z)

    k_max = 20.0
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

    h = cosmo['H0']/100
    cosmo['Omega_cdm'] = cosmo["omega_cdm"]/h**2
    cosmo['Omega_b'] = cosmo["omega_b"]/h**2
    cosmo['Omega_m'] = cosmo['Omega_cdm']+cosmo['Omega_b'] 
    cosmo['T_CMB'] = 2.7255
    cosmo['h'] = h

    k_arr = np.geomspace(1E-4, 1E2, 2000)

    import pyccl as ccl
    cosmopy = ccl.Cosmology(Omega_c=cosmo['omega_cdm']/h**2, Omega_b=cosmo['omega_b']/h**2, h=h, A_s=cosmo['A_s'], n_s=cosmo['n_s'], m_nu = 0.06)
    pklin = ccl.linear_matter_power(cosmopy, k_arr, 1.0)
    transfer_function = np.sqrt(pklin/k_arr**cosmo['n_s'])
    transfer_function = transfer_function[0]

    sf = 1/(1+z)
    Dz = ccl.growth_factor(cosmopy, sf)
    #Dz_norm = (1/51.) / ccl.growth_factor(cosmopy, 1/51.)
    Dz_norm = 0.01/ccl.growth_factor(cosmopy, 0.01)
    Dz = Dz * Dz_norm

    Omega_M = cosmo['Omega_m']
    H0 = (cosmo['h']/ccl.physical_constants.CLIGHT_HMPC)
    M = 2*Dz*transfer_function/(3*H0**2*Omega_M)*k_arr**2
    
    M_interp = qeutils.get_interpolated(k_arr, M)

    return M_interp

def get_kernels():

    K, q1, q2, mu = sp.symbols('K q1 q2 mu')

    M = sp.Function('M')

    #expr_c11 = sp.parsing.sympy_parser.parse_expr("0.5*(1/M(q1)+1/M(q2))")
    #expr_c01 = sp.parsing.sympy_parser.parse_expr("0.5 * mu*q1*q2 * (1/(q1**2*M(q2))+1/(q2**2*M(q1)))")
    #expr_c02 = sp.parsing.sympy_parser.parse_expr("1/(M(q1)*M(q2))")
    #expr_phiphi = sp.parsing.sympy_parser.parse_expr("M(sqrt(q1**2+q2**2+2*q1*q2*mu))*1/M(q1)*1/M(q2)")

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
            },
            'c11': { 
                'F': 0.5*(1./M(q1)+1./M(q2)),
                #F': expr_c11,
                'ca': 1,
                'cb': 1
            },
            'c01': {
                'F': 0.5 * mu*q1*q2 * (1./(q1**2.*M(q2))+1./(q2**2.*M(q1))),
                #'F': expr_c01,
                'ca': 1,
                'cb': 1
            },
            'c02': {
                'F': (1./(M(q1)*M(q2))),
                #'F': expr_c02,
                'ca': 1,
                'cb': 1
            },
            'phiphi': {
                'F': M(sp.sqrt(q1**2.+q2**2.+2*q1*q2*mu)) * (1./M(q1)) * (1./M(q2)),
                #'F': expr_phiphi,
                'ca': 1,
                'cb': 1
            },
            'c11a': {
                'F': 0.5*(1./M(q1)+1./M(q2)),
                'ca': -1,
                'cb': 1
            },
            'c01a': {
                'F': 0.5 * mu*q1*q2 * (1./(q1**2.*M(q2))+1./(q2**2.*M(q1))),
                'ca': -1,
                'cb': 1
            },
            'c02a': {
                'F': (1./(M(q1)*M(q2))),
                'ca': -1,
                'cb': 1
            },
            'phiphia': {
                'F': M(sp.sqrt(q1**2.+q2**2.+2*q1*q2*mu)) * (1./M(q1)) * (1./M(q2)),
                'ca': -1,
                'cb': 1
            }
        }
    
    return estimator_configs, M


def run_analysis(config_path, config_path_hod):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract parameters from config
    ps_config = config['power_spectrum']

    kr_config = config['k_range']
    sampling_config = config['sampling']
    output_config = config['output']

    #make directory
    new_directory = Path(ps_config['main_directory']+config['name'])
    new_directory.mkdir(exist_ok=True, parents=True)
    #copy txt files from ../data/ to the new directory
    #ps_config['nonlinear']
    #ps_config['linear']
    #copy ps_config['nonlinear'] in ../data/ to the new directory
    shutil.copy("../data/"+ps_config['nonlinear'], new_directory / ps_config['nonlinear'])
    shutil.copy("../data/"+ps_config['linear'], new_directory / ps_config['linear'])
    
    # Load power spectrum data
    knl, pnl = np.loadtxt(ps_config['main_directory']+config['name']+"/"+ps_config['nonlinear']).T
    kl, pl = np.loadtxt(ps_config['main_directory']+config['name']+"/"+ps_config['linear']).T
    
    interpolate_function = qeutils.get_interpolated(knl, pnl)
    interpolate_function_lin = qeutils.get_interpolated(kl, pl)
    
    #sim_name_base = config['sim_params']['sim_name_base']


    for sim_index in [0]:

        #sim_name = sim_name_base + f"{sim_index:03}"

        #if we have a HOD config, we use it to get the bias and shot noise parameters
        if config_path_hod is not None:
            with open(config_path_hod, 'r') as file:
                config_hod = yaml.safe_load(file)
            sim_params = config_hod['sim_params']
            z_mock = sim_params['z_mock']

            sim_name = sim_params["sim_name"]

            print("Working on sim", sim_name)

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

            bs2_fid_A = bs2_A #qeutils.bs2_coev(b10_A)
            b2_fid_A = b20_A/2 #qeutils.b2_fid(b10_A) #NOTE DIVISION BY 2
            bs2_fid_B = bs2_B #qeutils.bs2_coev(b10_B)
            b2_fid_B = b20_B/2 #qeutils.b2_fid(b10_B)

            print("Comparing b20s (sim vs theory), A and B", "A", b20_A, "theory", qeutils.b2_fid(b10_A), "B", b20_B, "theory", qeutils.b2_fid(b10_B))
            print("Comparing bs2s (sim vs theory), A and B", "A", bs2_A, "theory", qeutils.bs2_coev(b10_A), "B", bs2_B, "theory", qeutils.bs2_coev(b10_B))

            bs2_fid_A = bs2_fid_A

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

            b2_fid_A = b2_fid_A
            bs2_fid_A = bs2_fid_A 

            bs2_fid_B = bs2_fid_B
            b2_fid_B = b2_fid_B

        #b2_fid, bs2_fid are used in the bispectrum for the shot-trispectrum calculations. Ideally, you would want a cross-trispectrum.
        #For computational ease, I just insert the trispectrum for a single tracer here.

        P_AA = qeutils.get_interpolated(kA, Ptot_A)
        P_BB = qeutils.get_interpolated(kB, Ptot_B)

        print("SAME TRACERS", same_tracers)

        P_AB = P_AA if same_tracers else qeutils.get_interpolated(kl, b10_A*b10_B*pnl) #here is non-linear
        P_linear = jax.jit(lambda k: interpolate_function_lin(k))

        P_AA_signal = qeutils.get_interpolated(kA, b10_A**2*pl) #needs to be linear
        P_A_signal = qeutils.get_interpolated(kA, b10_A*pl)
        P_BB_signal = qeutils.get_interpolated(kB, b10_B**2*pl)
        P_B_signal = qeutils.get_interpolated(kB, b10_B*pl)
        P_AB_signal = qeutils.get_interpolated(kl, b10_A*b10_B*pl) 

        # Keypairs and estimator keys
        keypairs = config['keypairs'] #this is where we loop over the keypairs
        
        # k-range parameters
        kmin = kr_config['kmin']
        kmax = kr_config['kmax']
        k_samples = kr_config['k_samples']
        k_samples_extra = kr_config.get('k_samples_extra', 200)
        k_min_analysis = kr_config['k_min_analysis']
        k_max_analysis = kr_config['k_max_analysis']
        
        #define modes on which we reconstruct
        #Ks = np.linspace(k_min_analysis, k_max_analysis, k_samples)
        kmin_max = 2*k_min_analysis
        Ks_ = np.linspace(k_min_analysis, kmin_max, 20)
        Ks__ = np.linspace(kmin_max, kmin, k_samples_extra)
        Ks = np.logspace(np.log10(kmin), np.log10(k_max_analysis), k_samples)
        Ks = np.concatenate([Ks_, Ks__, Ks])
        Ks = np.unique(Ks)
        
        # Sampling parameters for Monte Carlo integration
        Nsamples_base = sampling_config['Nsamples_base']        

        estimator_configs, M = get_kernels()


        # Separate estimators that use M from those that don't
        estimators_with_M = ['c11', 'c01', 'c02', 'phiphi', 'c11a', 'c01a', 'c02a', 'phiphia']
        estimators_without_M = [key for key in estimator_configs if key not in estimators_with_M]
        
        # Get the M function
        Mscipy = get_quick_M()
        M_jax = jax.jit(lambda x: Mscipy(x))
        extra_funcs = {M: M_jax}

        estimator_lam_jax = {}

        # Handle estimators without M (no extra_funcs needed)
        for key in estimators_without_M:
            estimator_lam_jax[key] = sympy2jax.SymbolicModule(estimator_configs[key]['F'])
        
        # Handle estimators with M (extra_funcs required)
        for key in estimators_with_M:
            estimator_lam_jax[key] = sympy2jax.SymbolicModule(
                estimator_configs[key]['F'], 
                extra_funcs=extra_funcs
            )

        
        #estimator_lam_jax = {key: sympy2jax.SymbolicModule(estimator_configs[key]['F']) for key in estimator_configs}
        
        f_jax = {key: qeutils.get_f(estimator_lam_jax[key], P_linear, estimator_configs[key]["ca"], estimator_configs[key]["cb"]) for key in estimator_lam_jax if key != "n"}
        #f_jax["n"] = qeutils.get_f(estimator_lam_jax["n"], P_linear, estimator_configs["n"]["ca"], estimator_configs["n"]["cb"]) 
        f_jax["n"] = qeutils.get_f_squeezed(estimator_lam_jax["n"], P_linear)
        
        Fkernels = [qeutils.Fg, qeutils.Fs, qeutils.Ft]
        print("b2_A, b2_B", b2_fid_A, b2_fid_B)
        Fbiases_A = [qeutils.bias_g(b10_A, b2_fid_A), qeutils.bias_s(b10_A), qeutils.bias_t(b10_A, bs2_fid_A)]
        Fbiases_B = [qeutils.bias_g(b10_B, b2_fid_B), qeutils.bias_s(b10_B), qeutils.bias_t(b10_B, bs2_fid_B)]

        # Prepare output directory
        output_dir = Path(output_config['directory'])/config['name']
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Run analysis
        out_normalization_AB = {} #this is the inverse of the normalization factor for the estimator
        out_variance_AB = {} #this is the variance of the estimator
        out_cross_shot_AB = {} #this is the cross-shot noise
        out_cross_shot_AB_withB = {} #this is the cross-shot noise, but with B as the second tracer
        out_shot_bispectrum = {} #this is the shot noise of the bispectrum, but assuming one single tracer
        out_shot_trispectrum = {} #this is the shot noise of the trispectrum, but assuming one single tracer
        out_weight_integral_AB = {}

        out_normalization_BA = {} #this is the inverse of the normalization factor for the estimator, swapping A and B
        out_variance_BA = {} #this is the variance of the estimator, swapping A and B
        out_cross_shot_BA = {} #this is the cross-shot noise, swapping A and B


        P_A_signal_jax = jax.jit(lambda k: P_A_signal(k))
        P_B_signal_jax = jax.jit(lambda k: P_B_signal(k))
        #P_BB_signal_jax = jax.jit(lambda k: P_BB_signal(k))
        #P_AB_signal_jax = jax.jit(lambda k: P_AB_signal(k))
        #(P_signal_X, P_signal_Y, P_signal_Z, Fkernels, Fbiases_X, Fbiases_Y, Fbiases_Z)

        print("Fbiases_A", Fbiases_A)
        print("Fbiases_B", Fbiases_B)

        bispectrum_cont_ABB = qeutils.get_bispectrum_XYZ(P_A_signal_jax, P_B_signal_jax, P_B_signal_jax, Fkernels, Fbiases_A, Fbiases_B, Fbiases_B)
        bispectrum_cont_BAA = qeutils.get_bispectrum_XYZ(P_B_signal_jax, P_A_signal_jax, P_A_signal_jax, Fkernels, Fbiases_B, Fbiases_A, Fbiases_A)
        
        # Add progress bar for keypair calculations
        print("Running analysis for each keypair...")
        pbar = tqdm(total=len(keypairs), desc="Processing keypairs")
        for keypair in keypairs:
            keypair, quantity = keypair
            key1, key2 = keypair
            # Update the progress bar to show current keypair
            pbar.set_description(f"Processing ({key1}, {key2})")

            full_case = ("allfull" in quantity) or ("Nfull" in quantity) or ("Vfull" in quantity) or ("Tfull" in quantity) or ("Bfull" in quantity) or ("Bcrossfull" in quantity)

            if full_case:
                w_A = qeutils.get_full_w(f_jax[key1], P_AA, P_BB, P_AB)
            else:
                w_A = qeutils.get_w(f_jax[key1], P_AA, P_BB)
            
            if "N" in quantity or "all" in quantity or full_case:

                if not full_case:#in principle I could just use N_per_mode_weighted
                    N_single_AB = qeutils.N_per_mode(f_jax[key1], f_jax[key2], P_AA, P_BB, kmin, kmax, Nsamples_base=Nsamples_base, gauss_filter=False)
                    result_N = qeutils.integrate(Ks, N_single_AB, batch_size=2)
                else:
                    print("Calculating full normalization.")
                    N_per_mode_weighted = qeutils.N_per_mode_weighted(w_A, f_jax[key2], kmin, kmax, Nsamples_base=Nsamples_base, gauss_filter = False)
                    result_N = qeutils.integrate(Ks, N_per_mode_weighted, batch_size=2)

                out_normalization_AB[tuple(keypair)] = result_N
                out_normalization_AB[(key2, key1)] = out_normalization_AB[tuple(keypair)]


            if key1 == key2:
                out_weight_integral_AB[key1] = qeutils.integrate(Ks, qeutils.weight_integral(w_A, kmin, kmax, Nsamples_base=Nsamples_base, gauss_filter = False), batch_size=2)
            # N_single_AB_weighted = qeutils.N_per_mode_weighted(w_A, f_jax[key2], kmin, kmax, Nsamples_base=Nsamples_base, gauss_filter = False)
            #print(out_normalization_AB[tuple(keypair)]/qeutils.integrate(Ks, N_single_AB_weighted, batch_size=2))

            if "Bcross" in quantity or "all" in quantity:
                cross_single_AB = qeutils.cross_shot_mixed_AAB(w_A, nbar_A, P_AB_signal, kmin=kmin, kmax=kmax, Nsamples_base=Nsamples_base)
                out_cross_shot_AB[tuple(keypair)] = qeutils.integrate(Ks, cross_single_AB, batch_size=2)
                out_cross_shot_AB[(key2, key1)] = out_cross_shot_AB[tuple(keypair)]

                cross_single_AB = qeutils.cross_shot_mixed_AAB(w_A, nbar_B, P_AB_signal, kmin=kmin, kmax=kmax, Nsamples_base=Nsamples_base, activate_k2 = False)
                out_cross_shot_AB_withB[tuple(keypair)] = qeutils.integrate(Ks, cross_single_AB, batch_size=2)
                out_cross_shot_AB_withB[(key2, key1)] = out_cross_shot_AB_withB[tuple(keypair)]

            if full_case:
                w_B = qeutils.get_full_w(f_jax[key2], P_AA, P_BB, P_AB)
            else:
                w_B = qeutils.get_w(f_jax[key2], P_AA, P_BB)


            if "T" in quantity or "all" in quantity:
                print("Calculating shot trispectrum")
                #torchquad = True
                #shot_trispectrum = qeutils.shot_trispectrum(w_A, w_B, P_AA_signal, bispectrum_cont, nbar_A, kmin, kmax, Nsamples_base=400//20, torchquad = torchquad)
                #shot_result = qeutils.integrate(Ks, shot_trispectrum, batch_size=2) if torchquad else qeutils.integrate_vegas(Ks, shot_trispectrum, kmin = kmin, kmax = kmax)
                #will focus on shot-noise for w_A=w_A
                shot_trispectrum = qeutils.shot_trispectrum_mixed(w_A, w_A, P_AB, bispectrum_cont_ABB, bispectrum_cont_BAA, nbar_A, nbar_B, kmin, kmax, Nsamples_base=400//20)
                shot_result = qeutils.integrate(Ks, shot_trispectrum, batch_size=2)

                out_shot_trispectrum[tuple(keypair)] = shot_result
                out_shot_trispectrum[(key2, key1)] = out_shot_trispectrum[tuple(keypair)]


            #if "B" in quantity or "all" in quantity or full_case:
            #    shot_bispectrum = qeutils.shot_bispectrum(w_A, nbar_A, P_AA_signal, kmin, kmax, Nsamples_base=Nsamples_base)
            #    shot_bispectrum_result = qeutils.integrate(Ks, shot_bispectrum, batch_size=2)
            #    out_shot_bispectrum[tuple(keypair)] = shot_bispectrum_result
            #    out_shot_bispectrum[(key2, key1)] = out_shot_bispectrum[tuple(keypair)]

            #AB-XY = AB-AB
            #
            #variance_single_AB = qeutils.variance_per_mode(w_A, w_B, P_linear, P_linear, P_linear, P_linear, kmin, kmax, Nsamples_base=Nsamples_base//20)
            if "V" in quantity or "all" in quantity or full_case:
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
        np.save(output_dir / f"{filename_prefix}_cross_shot_AB_withB.npy", out_cross_shot_AB_withB)
        np.save(output_dir / f"{filename_prefix}_weight_integral_AB.npy", out_weight_integral_AB)
        np.save(output_dir / f"{filename_prefix}_variance_AB.npy", out_variance_AB)
        np.save(output_dir / f"{filename_prefix}_shot_trispectrum_AB.npy", out_shot_trispectrum)
        #np.save(output_dir / f"{filename_prefix}_shot_bispectrum_AB.npy", out_shot_bispectrum)
        np.save(output_dir / f"{filename_prefix}_Ks.npy", Ks)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run noise and bias analysis using configuration from a YAML file.')
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file', default='config_abacus_thy.yaml')
    parser.add_argument('--config_hod', type=str, help='Path to the HOD configuration file', default = None) #default='config_hod_0.yaml')
    parser.add_argument('--config_dir', type=str, help='Path to the configuration directory', default='../configs/abacus/')
    parser.add_argument('--gpu', type=int, help='GPU to use', default=0)
    args = parser.parse_args()
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # Only GPU 3 will be visible to JAX
    
    run_analysis(args.config_dir + "/" + args.config, args.config_dir + "/" + args.config_hod if args.config_hod is not None else None)

