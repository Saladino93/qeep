from qeep import qeutils
import jax
import jax.numpy as jnp
import numpy as np
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm

from qeep import fisher

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import LogLocator, LogFormatter, AutoMinorLocator
from matplotlib.gridspec import GridSpec
from matplotlib import patheffects
import matplotlib as mpl

# ---- CONFIGURATION ----
# Use golden ratio for figure dimensions
GOLDEN_RATIO = (5**0.5 - 1) / 2
FIG_WIDTH = 5  # inches
FIG_HEIGHT = FIG_WIDTH * GOLDEN_RATIO
DPI = 300

# Check for and configure LaTeX if available (optional but professional)
# Uncomment this if you have LaTeX installed
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
# })

# If not using LaTeX, use a clean serif font
# Try to use TeX fonts that are included with matplotlib
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm"  # Use Computer Modern math font
})

# Define a modern, colorblind-friendly palette with higher contrast
# Based on colorblindness-friendly scientific palettes like viridis
# and ones recommended by Nature and Science publications
COLORBLIND_PALETTE = [
    '#0072B2',  # Blue
    '#D55E00',  # Orange
    '#009E73',  # Green
    '#CC79A7',  # Pink
    '#56B4E9',  # Light blue
    '#E69F00',  # Yellow
    '#000000',  # Black
    '#F0E442'   # Light yellow
]


key_selected = "n"

# More descriptive and professionally formatted names
names = {
    "n": r"$\mathcal{D}$",          # Using calligraphic D for density
    "x": r"$\varepsilon$",           # Using proper epsilon symbol
    "s": r"$\mathcal{S}$",           # Calligraphic S
    "t": r"$\mathcal{T}$",           # Calligraphic T
    "g": r"$\mathcal{G}$",           # Calligraphic G
    "x2": r"$10\varepsilon$"         # 10 epsilon
}

estimators_all = ["s", "g", "t", "n", "x", "x2"]
estimators_base = ["s", "g", "t", "x"]



def symm(f, A, B, **kwargs):
    return (f(A, B, **kwargs)+f(B, A, **kwargs))/2
def asymm(f, A, B, **kwargs):
    return (f(A, B, **kwargs)-f(B, A, **kwargs))/2

def bs2_coev(b10):
  """
  Coevolution value of the bs2 parameter
  """
  return -2./7.*(b10-1)

def b2_fit(b10):
    return 2*(0.412-2.143*b10 + 0.929*b10**2 + 0.008*b10**3)

def br_formula(b1, z = 0.5):
    h = 0.6736
    H0 = h*100
    bias = 6.8*((1+z)*H0)**-1.*(b1-1) #formula 20 from Schmidt 2016
    return bias

def bGX(b10):
    factor = 1421/510
    return br_formula(b10)*factor

def bSX(b10):
    factor = 17/6
    return br_formula(b10)*factor

def bTX(b10):
    factor = 91/30
    return br_formula(b10)*factor


def cg_g(biases_A, biases_B, e):
    return (biases_A[0]+21/17*biases_A[1]+e*biases_A[2])*biases_B[0]

def cg_s(biases_A, biases_B, e):
    return (biases_A[0]+e*biases_A[1])*biases_B[0]

def cg_t(biases_A, biases_B, e):
    return (biases_A[0]+7/2*biases_A[1]+e*biases_A[2])*biases_B[0]


def get_Cg_biases(e, b1A, b1B, b2A, b2B, bGXA, bGXB, bSXA, bSXB, bs2A, bs2B, bTXA, bTXB, zero_shift = 1.):
    """
    Cg = jnp.array([Cg_g, Cg_ga, Cg_s, Cg_sa, Cg_t, Cg_ta])
    """

    biases_A_G = jnp.array([b1A, b2A, bGXA])
    biases_B_G = jnp.array([b1B, b2B, bGXB])

    biases_A_S = jnp.array([b1A, bSXA])
    biases_B_S = jnp.array([b1B, bSXB])

    biases_A_T = jnp.array([b1A, bs2A, bTXA])
    biases_B_T = jnp.array([b1B, bs2B, bTXB])

    Cg_g = symm(cg_g, biases_A_G, biases_B_G, e = e) #need to change definition of symm and asymm
    Cg_ga = asymm(cg_g, biases_A_G, biases_B_G, e = e)

    Cg_s = symm(cg_s, biases_A_S, biases_B_S, e = e)
    Cg_sa = asymm(cg_s, biases_A_S, biases_B_S, e = e)*zero_shift

    Cg_t = symm(cg_t, biases_A_T, biases_B_T, e = e)
    Cg_ta = asymm(cg_t, biases_A_T, biases_B_T, e = e)

    Cg = jnp.array([Cg_g, Cg_ga, Cg_s, Cg_sa, Cg_t, Cg_ta])

    return Cg


def get_bias_jax(v, zero_shift = 1, N = None, jax_out_normalization_AB = None):
    e, b1A, b1B, b2A, b2B, bs2A, bs2B, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB = v
    cg_values = get_Cg_biases(e, b1A, b1B, b2A, b2B, bGXA, bGXB, bSXA, bSXB, bs2A, bs2B, bTXA, bTXB, zero_shift)
    partials = N * jax_out_normalization_AB * cg_values[:, jnp.newaxis]
    # Sum across keys to get the total bias
    bias = jnp.sum(partials, axis=0)
    return jnp.nan_to_num(bias)


def get_bias_jax_A_eq_B(v, zero_shift = 1, N = None, jax_out_normalization_AB = None):
    """
    Here, I assume A and B biases are the same, except for the higher order ones!
    """

    e, b1A, b2A, bs2A, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB = v
    b1B, b2B, bs2B = b1A, b2A, bs2A

    w = jnp.array([e, b1A, b1B, b2A, b2B, bs2A, bs2B, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB])
   
    return get_bias_jax(w, zero_shift, N, jax_out_normalization_AB)


def E_bottaro():
    z_eq = 3400
    a_eq = 1/(1+z_eq)
    a = 1.
    fchi = 1.
    factor = jnp.log(a/a_eq)-181/90
    factor *= fchi
    factor *= 6/5
    return factor

def G(epsilon, one = 1):
    E = E_bottaro()
    return 1+epsilon*E*one

def get_total_cross(v, Ks, one = 0, zero_shift = 1, plinf = None, N = None, jax_out_normalization_AB = None):
    #v has e, b1A, b2A, bs2A, bGXA, bSXA, bTXA, bGXB, bSXB, bTXB
    b1C = v[1]*G(v[0], one)**3
    return b1C*get_bias_jax_A_eq_B(v, zero_shift, N, jax_out_normalization_AB)*plinf(Ks)


def get_total_auto(v, Ks, variance, one = 0, zero_shift = 1, plinf = None, N = None, jax_out_normalization_AB = None):
    #v has e, b1A, b2A, bs2A, bGXA, bSXA, bTXA, bGXB, bSXB, bTXB
    return G(v[0], one)**4*get_bias_jax_A_eq_B(v, zero_shift, N, jax_out_normalization_AB)**2*plinf(Ks)+variance


def get_galaxy_auto(v, Ks, one = 0, plinf = None):
    #v has e, b1A, b2A, bs2A, bGXA, bSXA, bTXA, bGXB, bSXB, bTXB
    return v[1]**2*plinf(Ks)*G(v[0], one)**2



def get_cov(Ks, variance = 0, one = 0, zero_eps_shift = 1, plinf = None, N = None, jax_out_normalization_AB = None):
    @jax.jit
    def covariance_full(K_array, v):
        """
        This covariance includes:
        galaxy power spectrum A
        reconstruction cross spectrum A-R
        reconstruction auto spectrum R
    
        We assume b1 biases are same for A and B, hence we just include 3 probes

        If rec_noise = 0, no rec noise in auto-reconstruction. rec_noise = 1, includes it.
        If one = 0, not growth term. one = 1, include growth term.
        If zero_eps_shift = 1, you include shift asymmetric terms. zero_eps_shift = 0, you null them.
        """
        
        n_probes = 3
        
        #P = plinf_jax(K_array)
    
        CAR = get_total_cross(v, Ks, one, zero_shift = zero_eps_shift, plinf = plinf, N = N, jax_out_normalization_AB = jax_out_normalization_AB) #gets cross-spectrum
        CRR = get_total_auto(v, Ks, variance, one, zero_shift = zero_eps_shift, plinf = plinf, N = N, jax_out_normalization_AB = jax_out_normalization_AB) #gets reconstruction auto-spectrum
        CAA = get_galaxy_auto(v, Ks, one, plinf = plinf) #gets galaxy auto-spectrum
    
        C = jnp.zeros((len(K_array), n_probes, n_probes))
        
        #R, A, B
        C = C.at[:, 0, 0].set(CRR)
        C = C.at[:, 0, 1].set(CAR)
        C = C.at[:, 1, 0].set(CAR)
        C = C.at[:, 1, 1].set(CAA)
        
        return C
    return covariance_full



def get_simple_error(K_array, F, fast = True, Kmin = 0.001, Kmax = 0.05, V = 1):
    
    err_ms, err_us = [], []
    
    Kmaxarr = min(0.1, Kmax)
    modes = jnp.linspace(Kmin, Kmaxarr, 20)

    for KK in modes:
        if not fast:
            err_m, err_u = fisher.get_error_bars_from_F(fisher.get_F_integrated(K_array, F, KK, Kmax, V = V))
        else:
            err_m, err_u = fisher.get_error_bars_from_F(fisher.get_F_integrated_fast(K_array, F, KK, Kmax, V = V))
        err_ms.append(err_m)
        err_us.append(err_u)

    err_ms = jnp.array(err_ms)
    err_us = jnp.array(err_us)

    return modes, err_ms, err_us



def run_forecast(config_path, config_path_hod, key = "n"):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(config_path_hod, 'r') as f:
        config_hod = yaml.safe_load(f)



    ps_main_directory = config['power_spectrum']['main_directory']
    name_config = config['name']
    gen_nl_power = np.loadtxt(ps_main_directory+name_config+"/"+config['power_spectrum']['nonlinear'])
    gen_power = np.loadtxt(ps_main_directory+name_config+"/"+config['power_spectrum']['linear'])

    output_config = config['output']
    filename_prefix = output_config['filename_prefix']
    filename_prefix = output_config['filename_prefix']
    output_dir = Path(output_config['directory'])/config['name']

    #ic = load_dens(ic_dir, sim_name, ngrid)
    import jax.numpy as jnp
    pnlinf = lambda kmag: jnp.interp(kmag, gen_nl_power[:,0], gen_nl_power[:,1])
    plinf = lambda kmag: jnp.interp(kmag, gen_power[:,0], gen_power[:,1])

    kmin, kmax = config['k_range']['kmin'], config['k_range']['kmax']

    output_config = config['output']
    filename_prefix = output_config['filename_prefix']
    filename_prefix = output_config['filename_prefix']
    output_dir = Path(output_config['directory'])/config['name']

    nome = "analysisAbacusSummit_base_c000_ph000_z0.500_LRG_ELG_normalization_AB.npy"
    nomev = "analysisAbacusSummit_base_c000_ph000_z0.500_LRG_ELG_variance_AB.npy"
    #nometri = "analysisAbacusSummit_base_c000_ph000_z0.500_LRG_ELG_shot_trispectrum_AB.npy"
    #out_normalization_AB = np.load(output_dir / f"{filename_prefix}_normalization_AB.npy", allow_pickle = True).item()

    out_normalization_AB = np.load(output_dir / nome, allow_pickle = True).item()
    #out_variance_AB = np.load(output_dir / f"{filename_prefix}_variance_AB.npy", allow_pickle = True).item()
    #analysis_cross_shot_AB = np.load(output_dir / f"{filename_prefix}_cross_shot_AB.npy", allow_pickle = True).item()
    out_variance_AB_other = np.load(output_dir / nomev, allow_pickle = True).item()
    #out_shot_trispectrum = np.load(output_dir / nometri, allow_pickle = True).item()

    kr_config = config['k_range']
    kmin = kr_config['kmin']
    kmax = kr_config['kmax']
    k_samples = kr_config['k_samples']
    k_min_analysis = kr_config['k_min_analysis']
    k_max_analysis = kr_config['k_max_analysis']

    kmin_max = 2*k_min_analysis
    Ks_ = np.linspace(k_min_analysis, kmin_max, 20)
    Ks = np.logspace(np.log10(kmin_max), np.log10(k_max_analysis), k_samples)
    Ks = np.concatenate([Ks_, Ks])
    Ks = np.unique(Ks)


    Ks = np.linspace(k_min_analysis, k_max_analysis, k_samples)




    keys = ["g", "ga", "s", "sa", "t", "ta"]
    labels = [r"$\mathrm{\mathcal{G}}_{\mathrm{symm}}$", r"$\mathrm{\mathcal{S}}_{\mathrm{symm}}$", r"$\mathrm{\mathcal{T}}_{\mathrm{symm}}$", r"$\mathrm{\mathcal{G}}_{\mathrm{asymm}}$", r"$\mathrm{\mathcal{S}}_{\mathrm{asymm}}$", r"$\mathrm{\mathcal{T}}_{\mathrm{asymm}}$"]

    num_keys = 6  # For g, s, t, ga, sa, ta

    N = np.array(out_normalization_AB[(key, key)]**-1.)
    N[N>1e10] = 0
    N = jnp.array(N)

    jax_out_normalization_AB = np.empty((len(keys), len(Ks)))
    for i, k in enumerate(keys):
        jax_out_normalization_AB[i, :] = out_normalization_AB[(key, k)]
    jax_out_normalization_AB = jnp.array(jax_out_normalization_AB)


    b1A = 1.6
    b1B = 2
    e = 1e-3

    b2A, b2B = b2_fit(b1A), b2_fit(b1B)
    bGXA, bGXB = bGX(b1A), bGX(b1B)
    bSXA, bSXB = bSX(b1A), bSX(b1B)
    bs2A, bs2B = bs2_coev(b1A), bs2_coev(b1B)
    bTXA, bTXB = bTX(b1A), bTX(b1B)


    variance = jnp.array(out_variance_AB_other[(key, key)])*N**2

    #ASSUMES ONLY A, NO B INFO

    K_array = Ks
    vA = jnp.array([e, b1A, b2A, bs2A, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB])

    from interpax import Interpolator1D
    plinf_jax = Interpolator1D(gen_power[:,0], gen_power[:,1], method="cubic")


    #here we make forecast for the cross-correlation only
    Ofunc = lambda K_arr, v: get_total_cross(v, K_arr, N = N, jax_out_normalization_AB = jax_out_normalization_AB, plinf = plinf_jax)
    Ofunc_no_shift = lambda K_arr, v: get_total_cross(v, K_arr, zero_shift = 0, N = N, jax_out_normalization_AB = jax_out_normalization_AB, plinf = plinf_jax)

    variance_func = lambda K_arr, v: (get_galaxy_auto(v, K_arr, plinf = plinf_jax))*(get_total_auto(v, K_arr, plinf = plinf_jax, variance = variance, N = N, jax_out_normalization_AB = jax_out_normalization_AB))
    F_single = fisher.fisher_per_mode_single(vA, K_array, Ofunc, variance_func) #Ofunc(K_array, v) returns (n_modes, nprobes, nprobes)
    F_single = jnp.nan_to_num(F_single)

    F_single_no_shift = fisher.fisher_per_mode_single(vA, K_array, Ofunc_no_shift, variance_func) #Ofunc(K_array, v) returns (n_modes, nprobes, nprobes)
    F_single_no_shift = jnp.nan_to_num(F_single_no_shift)


    F = fisher.fisher_per_mode(vA, K_array, get_cov(Ks, plinf = plinf_jax, N = N, jax_out_normalization_AB = jax_out_normalization_AB))
    F_with_variance = fisher.fisher_per_mode(vA, K_array, get_cov(Ks, variance = variance, plinf = plinf_jax, N = N, jax_out_normalization_AB = jax_out_normalization_AB))
    F_with_variance_no_shift = fisher.fisher_per_mode(vA, K_array, get_cov(Ks, variance = variance, plinf = plinf_jax, zero_eps_shift = jnp.array(0), N = N, jax_out_normalization_AB = jax_out_normalization_AB))
    F_with_variance_growth = fisher.fisher_per_mode(vA, K_array, get_cov(Ks, variance = variance, plinf = plinf_jax, one = jnp.array(1), N = N, jax_out_normalization_AB = jax_out_normalization_AB))

    #v = [e, b1A, b2A, bs2A, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB]
    #let's marginalize over bias parameters only, e, b1A, b2A, bs2A
    indices_set = {"basic_biases": [0, 1, 2, 3], "all_biases": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}

    cases = ["cross", "cross_no_shift", "full_no_noise", "full", "full_no_shift", "full_growth"]
    Fs = [F_single, F_single_no_shift, F, F_with_variance, F_with_variance_no_shift, F_with_variance_growth]

    #cross is BS
    #full is BS+TS

    errors = {}
    for c, F_ in zip(cases, Fs):
        temp_errors = {}
        for key in indices_set:
            F_sel = F[:, indices_set[key], :][:, :, indices_set[key]] #select the relevant biases over which we marginalize
            modes, err_ms, err_us = get_simple_error(K_array, F_sel, fast = True, Kmin = 0.001, Kmax = 0.05, V = 1)
            temp_errors[key] = {"modes": modes, "err_ms": err_ms, "err_us": err_us}
        errors[c] = temp_errors

    np.save(output_dir / f"{filename_prefix}_errors.npy", errors)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run forecast using configuration from a YAML file.')
    parser.add_argument('--config', type=str, help='Path to the YAML configuration file', default='config_abacus_recs.yaml')
    parser.add_argument('--config_hod', type=str, help='Path to the HOD configuration file', default='config_hod_0.yaml')
    parser.add_argument('--config_dir', type=str, help='Path to the configuration directory', default='../configs/abacus/')
    args = parser.parse_args()
    
    run_forecast(args.config_dir + "/" + args.config, args.config_dir + "/" + args.config_hod)