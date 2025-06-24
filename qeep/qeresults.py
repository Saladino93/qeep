import jax.numpy as jnp

import numpy as np

import pathlib


class QEResults:
    def __init__(self, config):

        name_config = config['name']
        output_config = config['output']
        output_dir = pathlib.Path(output_config['directory'])/name_config

        nome = "analysisAbacusSummit_base_c000_ph000_z0.500_LRG_ELG_normalization_AB.npy"
        nomev = "analysisAbacusSummit_base_c000_ph000_z0.500_LRG_ELG_variance_AB.npy"
        nometri = "analysisAbacusSummit_base_c000_ph000_z0.500_LRG_ELG_shot_trispectrum_AB.npy"
        nomebis_mixed = "analysisAbacusSummit_base_c000_ph000_z0.500_LRG_ELG_cross_shot_AB.npy"
        nomebis = "analysisAbacusSummit_base_c000_ph000_z0.500_LRG_ELG_shot_bispectrum_AB.npy"

        self.out_normalization_AB = np.load(output_dir / nome, allow_pickle = True).item() #inverse of normalization N

        self.analysis_cross_shot_AB = np.load(output_dir / nomebis_mixed, allow_pickle = True).item() #cross shot-noise
        self.out_variance_AB = np.load(output_dir / nomev, allow_pickle = True).item() #variance
        self.out_shot_trispectrum = np.load(output_dir / nometri, allow_pickle = True).item() #trispectrum shot noise, assuming all the same
        self.out_shot_bispectrum = np.load(output_dir / nomebis, allow_pickle = True).item() #bispectrum shot noise, assuming all the same


        ps_main_directory = config['power_spectrum']['main_directory']
        self.gen_nl_power = np.loadtxt(ps_main_directory+name_config+"/"+config['power_spectrum']['nonlinear'])
        self.gen_power = np.loadtxt(ps_main_directory+name_config+"/"+config['power_spectrum']['linear'])

        self.pnlinf = lambda kmag: jnp.interp(kmag, self.gen_nl_power[:,0], self.gen_nl_power[:,1])
        self.plinf = lambda kmag: jnp.interp(kmag, self.gen_power[:,0], self.gen_power[:,1])

        kr_config = config['k_range']
        kmin = kr_config['kmin']
        kmax = kr_config['kmax']
        k_samples = kr_config['k_samples']
        k_min_analysis = kr_config['k_min_analysis']
        k_max_analysis = kr_config['k_max_analysis']

        self.kmin = kmin
        self.kmax = kmax
        self.k_min_analysis = k_min_analysis
        self.k_max_analysis = k_max_analysis

        kmin_max = 2*k_min_analysis
        Ks_ = jnp.linspace(k_min_analysis, kmin_max, 20)
        Ks = jnp.logspace(jnp.log10(kmin_max), jnp.log10(k_max_analysis), k_samples)
        Ks = jnp.concatenate([Ks_, Ks])
        self.Ks = jnp.unique(Ks)

    def get_out_normalization_AB(self):
        result = {}
        for key, value in self.out_normalization_AB.items():
            value = np.array(value)
            value[np.abs(value)<1e-10] = 0
            result[key] = jnp.array(value)
        return result

    def get_get_norm(self):
        def get_norm(key):
            N = np.array(self.out_normalization_AB[(key, key)]**-1.)
            N[np.abs(N)>1e10] = 0
            N = jnp.array(N)
            return N
        return get_norm
    
    def get_get_variance(self):
        def get_variance(key):
            N = self.get_get_norm()(key)
            V = self.out_variance_AB[(key, key)]*N**2
            return V
        return get_variance
    
    def get_get_trispectrum(self):
        def get_trispectrum(key):
            N = self.get_get_norm()(key)
            T = self.out_shot_trispectrum[(key, key)]*N**2
            return T
        return get_trispectrum

    def get_get_bispectrum_mixed(self):
        def get_bispectrum_mixed(key):
            N = self.get_get_norm()(key)
            B = self.analysis_cross_shot_AB[(key, key)]*N
            return B
        return get_bispectrum_mixed

    def get_get_bispectrum(self):
        def get_bispectrum(key):
            N = self.get_get_norm()(key)
            B = self.out_shot_bispectrum[(key, key)]*N
            return B
        return get_bispectrum


    def get_get_tot_noises(self):
        def get_tot_noises(key):
            variance = self.get_get_variance()(key)
            bispectrum = self.get_get_bispectrum()(key)
            trispectrum = self.get_get_trispectrum()(key)
            bispectrum_mixed = self.get_get_bispectrum_mixed()(key)
            return variance, bispectrum, trispectrum, bispectrum_mixed

    @staticmethod
    def E_bottaro():
        z_eq = 3400
        a_eq = 1/(1+z_eq)
        a = 1.
        fchi = 1.
        factor = jnp.log(a/a_eq)-181/90
        factor *= fchi
        factor *= 6/5
        return factor

    def get_G(self):
        def G(epsilon, one = 1):
            E = self.E_bottaro()
            return 1+epsilon*E*one
        return G
    


def get_normalization_array(key, keys, Ks, out_normalization_AB):
    jax_out_normalization_AB = np.empty((len(keys), len(Ks)))
    for i, k in enumerate(keys):
        jax_out_normalization_AB[i, :] = out_normalization_AB[(key, k)]
    jax_out_normalization_AB = jnp.array(jax_out_normalization_AB)
    return jax_out_normalization_AB