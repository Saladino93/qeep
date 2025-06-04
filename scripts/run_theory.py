from qeep import qeutils
import jax

import numpy as np
import sympy as sp
import sympy2jax
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm


def run_analysis(config_path):
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Extract parameters from config
    ps_config = config['power_spectrum']
    bias_config = config['bias']
    nd_config = config['number_density']
    kr_config = config['k_range']
    sampling_config = config['sampling']
    physics_config = config['physics']
    #other_config = config['other']
    output_config = config['output']

    gauss_filter = kr_config['gauss_filter']
    
    # Load power spectrum data
    knl, pnl = np.loadtxt(ps_config['main_directory']+config['name']+"/"+ps_config['nonlinear']).T
    kl, pl = np.loadtxt(ps_config['main_directory']+config['name']+"/"+ps_config['linear']).T
    
    interpolate_function = qeutils.get_interpolated(knl, pnl)
    interpolate_function_lin = qeutils.get_interpolated(kl, pl)
    
    q1, q2, mu = sp.symbols('q1 q2 mu')
    
    def get_total_P(b10, nbar):
        shot = 1/nbar
        return jax.jit(lambda k: b10**2*interpolate_function(k)+shot)
    
    # Bias parameters
    b10_A = bias_config['b10_A']
    b10_B = bias_config['b10_B']
    b20_A = bias_config['b20_A']
    b20_B = bias_config['b20_B']
    bs2_A = bias_config['bs2_A']
    bs2_B = bias_config['bs2_B']
    
    bthetaA = bias_config['bthetaA']
    bthetaB = bias_config['bthetaB']
    brA = bias_config['brA']
    brB = bias_config['brB']
    
    # Number density parameters
    nbar_A = float(nd_config['nbar_A'])
    nbar_B = float(nd_config['nbar_B'])
    
    P_AA = get_total_P(b10_A, nbar_A)
    P_BB = get_total_P(b10_B, nbar_B)
    P_AB = jax.jit(lambda k: b10_A*b10_B*interpolate_function(k))
    P_linear = jax.jit(lambda k: interpolate_function_lin(k))
    
    # Physics parameters
    deltac = physics_config['deltac']
    a1 = physics_config['a1']
    a2 = physics_config['a2']
    Fg_factor = 17/21  # Use literal fraction as requested
    
    # Other parameters
    #epsilon = other_config['epsilon']
    
    # Define Cg dictionary
    Cg = {}
    Cg["g"] = lambda b1A, b2A, b1B, bthetaA, bdeltathetaA, fX, bmrA, brA: (b1A+Fg_factor*1/2*b2A)*b1B
    Cg["s"] = lambda b1A, b2A, bs2A, b1B, b2B, bs2B: b1A*b1B
    Cg["t"] = lambda b1A, b2A, bs2A, b1B, b2B, bs2B: (b1A+7/2*bs2A)*b1B #(2/7*b1A+1/2*bs2A)*b1B
    Cg["n"] = lambda b1A, b2A, bs2A, b1B, b2B, bs2B: 1*(b1A==b1B)
    Cg["x"] = lambda b1A, epsilon, brB, bDB, bthetaB, H: b1A*epsilon*(17/6*brB-7/3*H*bthetaB-5/3*H*bDB)
    
    # Keypairs and estimator keys
    keypairs = config['keypairs']
    estimator_keys = config['estimator_keys']
    
    # k-range parameters
    kmin = kr_config['kmin']
    kmax = kr_config['kmax']
    k_samples = kr_config['k_samples']
    k_min_analysis = kr_config['k_min_analysis']
    k_max_analysis = kr_config['k_max_analysis']
    
    Ks = np.linspace(k_min_analysis, k_max_analysis, k_samples)
    
    # Sampling parameters
    Nsamples_base = sampling_config['Nsamples_base']
    
    # Define estimator configs
    estimator_configs = {
        'g': {
            'F': Fg_factor*q1/q1,
            'ca': 1., 
            'cb': 1.
        },
        'ga': { #antisymmetric part of g
            'F': Fg_factor*q1/q1,
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
        if False:
            variance_single_AB = qeutils.variance_per_mode(w_A, w_B, P_AA, P_BB, P_AB, P_AB, kmin, kmax, Nsamples_base=Nsamples_base//10, gauss_filter = False)
            out_variance_AB[tuple(keypair)] = qeutils.integrate(Ks, variance_single_AB, batch_size=2)
            out_variance_AB[(key2, key1)] = out_variance_AB[tuple(keypair)]
        # Update progress
        pbar.update(1)
    
    pbar.close()

    
    # Save results to files
    filename_prefix = output_config['filename_prefix']
        
    # Save as NPY files
    np.save(output_dir / f"{filename_prefix}_normalization_AB.npy", out_normalization_AB)
    np.save(output_dir / f"{filename_prefix}_cross_shot_AB.npy", out_cross_shot_AB)
    np.save(output_dir / f"{filename_prefix}_variance_AB.npy", out_variance_AB)

    np.save(output_dir / f"{filename_prefix}_Ks.npy", Ks)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run noise and bias analysis using configuration from a YAML file.')
    parser.add_argument('-config', type=str, help='Path to the YAML configuration file', default='config_abacus.yaml')
    args = parser.parse_args()
    
    run_analysis(args.config)

