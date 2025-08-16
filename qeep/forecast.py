"""
Code with forecasting specifics. Just a big dump of functions.

I am assuming no dependence on K.

*TODO* Sympy function to automatically generate covariances and stuff.
"""

from qeep import biases, qeresults as qres, fisher
from jax import numpy as jnp
import jax
import yaml
import numpy as np



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

def get_CG(v, asymm_shift = 1., only_asymm_shift = 0.):  
    """
    Effective biases.

    asymm_shift = 1., means we include the C^{S}_{[AB]} signature from the anti-symmetric term. If zero, this is not included.
    only_asymm_shift = 0., means that we are including G and T anti-symmetric. If 1., we exclude them. This is useful to check origin of constraints on epsilon.
    """
    e, b1A, b1B, b2A, b2B, bs2A, bs2B, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB = v
    Cg = biases.get_Cg_biases(e, b1A, b1B, b2A, b2B, bs2A, bs2B, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB, asymm_shift = asymm_shift, only_asymm_shift = only_asymm_shift)
    return Cg


def get_tot_bias_Cg(normalization, Cg, responses):
    """
    This is the overall bias coming from all the present terms, given Cg.
    """
    partials = normalization * responses * Cg[:, jnp.newaxis]
    bias = jnp.sum(partials, axis=0)
    return jnp.nan_to_num(bias)

def get_tot_bias(v, normalization, responses, asymm_shift = 1, only_asymm_shift = 0.):
    """
    This is the overall bias given your parameter vector v, the normalization of the estimator, and the responses.
    """
    Cg = get_CG(v, asymm_shift = asymm_shift, only_asymm_shift = only_asymm_shift)
    return get_tot_bias_Cg(normalization, Cg, responses)

def galaxy_power_spectrum(v, plinear, nshot, growth_eps = 1.):
    """
    Standard power spectrum with linear bias and shot noise.
    We also include an additional growth factor depending on epsilon.
    """
    eps, b1X = v
    return  b1X**2*plinear*G(eps, growth_eps)**2 + nshot

def cross_galaxy_power_spectrum(vA, vB, plinear, cross_shot = 0., growth_eps = 1.):
    """
    Standard power spectrum with linear bias and shot noise.
    We also include an additional growth factor depending on epsilon.
    """
    eps, b1X = vA
    eps, b1Y = vB
    return  b1X*b1Y*plinear*G(eps, growth_eps)**2+cross_shot
    
def reconstruction_cross_spectrum(v, b1X, plinear, normalization, responses, bisp_noise, asymm_shift = 1, only_asymm_shift = 0., growth_eps = 1.):
    """
    bisp_noise, bispectrum shot noise
    growth_eps, include growth term epsilon or not
    """
    bias = get_tot_bias(v, normalization, responses, asymm_shift = asymm_shift, only_asymm_shift = only_asymm_shift)
    signal = b1X*bias*plinear*G(v[0], growth_eps)**3
    return signal + bisp_noise

def reconstruction_auto_spectrum(v, plinear, normalization, responses, var_noise, tris_noise, signal = 1., asymm_shift = 1, only_asymm_shift = 0, growth_eps = 1.,
                                normalization_2 = None, responses_2 = None):
    bias = get_tot_bias(v, normalization, responses, asymm_shift = asymm_shift, only_asymm_shift = only_asymm_shift)
    if normalization_2 is not None:
        bias2 = get_tot_bias(v, normalization_2, responses_2, asymm_shift = asymm_shift, only_asymm_shift = only_asymm_shift)
    else:
        bias2 = bias
    signal = bias*bias2*plinear*signal*G(v[0], growth_eps)**4
    return signal + var_noise + tris_noise


class Forecaster:
    def __init__(self, configuration, factor_boosting_A = 1., factor_boosting_B = 1., asymm_shift = 1., only_asymm_shift = 0., growth_eps = 0., shot_noise_A = 1., shot_noise_B = 1., shot_noise_AB = 0., variance_factor = 1., bispectrum_factor = 1., trispectrum_factor = 1., v_function = lambda v: v):
        """
        """

        with open(configuration, 'r') as f:
            config = yaml.safe_load(f)

        self.configuration = configuration
        self.config = config

        ps_main_directory = config['power_spectrum']['main_directory']
        name_config = config['name']
        gen_nl_power = np.loadtxt("../"+ps_main_directory+name_config+"/"+config['power_spectrum']['nonlinear'])
        gen_power = np.loadtxt("../"+ps_main_directory+name_config+"/"+config['power_spectrum']['linear'])

        kr_config = config['k_range']
        kmin = kr_config['kmin']
        kmax = kr_config['kmax']
        k_samples = kr_config['k_samples']
        k_min_analysis = kr_config['k_min_analysis']
        k_max_analysis = kr_config['k_max_analysis']
        k_samples_extra = kr_config.get('k_samples_extra', 200)

        kmin_max = 2*k_min_analysis
        Ks_ = np.linspace(k_min_analysis, kmin_max, 20)
        Ks__ = np.linspace(kmin_max, kmin, k_samples_extra)
        Ks = np.logspace(np.log10(kmin), np.log10(k_max_analysis), k_samples)
        Ks = np.concatenate([Ks_, Ks__, Ks])
        Ks = np.unique(Ks)
        pnlinf = lambda kmag: jnp.interp(kmag, gen_nl_power[:,0], gen_nl_power[:,1])
        plinf = lambda kmag: jnp.interp(kmag, gen_power[:,0], gen_power[:,1])


        relative_path = "../."
        self.QR = qres.QEResults(self.config, relative_path = relative_path)

        nbar_A = config['number_density']['nbar_A']
        nbar_B = config['number_density']['nbar_B']

        self.PL_calculated = plinf(Ks)
        self.Ks = Ks
        self.kmin = kmin
        self.kmax = kmax
        self.k_min_analysis = k_min_analysis
        self.k_max_analysis = k_max_analysis

        self.factor_boosting_A = factor_boosting_A
        self.factor_boosting_B = factor_boosting_B
        self.asymm_shift = asymm_shift
        self.only_asymm_shift = only_asymm_shift
        self.growth_eps = growth_eps
        self.shot_noise_A = shot_noise_A*1/nbar_A
        self.shot_noise_B = shot_noise_B*1/nbar_B
        self.shot_noise_AB = shot_noise_AB
        self.variance_factor = variance_factor
        self.bispectrum_factor = bispectrum_factor
        self.trispectrum_factor = trispectrum_factor
        self.v_function = v_function


    def get_functions(self, QR, key, key2 = None, keys = ["g", "s", "t", "ga", "sa", "ta"], wrapped = True):
        """
        Returns: CAA, CBB, CRR, CAB, CAR, CBR
        """
        result = self._get_functions(QR, key, key2, self.Ks, self.PL_calculated, self.shot_noise_A, self.shot_noise_B, self.shot_noise_AB, self.variance_factor, self.bispectrum_factor, self.trispectrum_factor, self.asymm_shift, self.only_asymm_shift, self.growth_eps, keys = keys)
        if wrapped:
            return get_functions_vA_vB(*result, v_function = self.v_function) #v_function allows to have custom parameters, but they need to map to the same space as the input vector used across the functions here
        else:
            return result
         

    def _get_functions(self, QR, key, key2 = None, Ks = None, plinear = None, shot_noise_A = 1., shot_noise_B = 1., shot_noise_AB = 0., variance_factor = 1., bispectrum_factor = 1., trispectrum_factor = 1., asymm_shift = 1., only_asymm_shift = 0., growth_eps = 0., keys = ["g", "s", "t", "ga", "sa", "ta"]):
        """
        This will give me basic functions that I can use to combine spectra.

        key is some QE key.

        Could be made more efficient and less redundant....

        Returns: CAA, CBB, CRR, CAB, CAR, CBR
        """   

        out_normalization_AB = QR.get_out_normalization_AB()

        if key2 is None:
            variance_noise = QR.get_get_variance()(key)*variance_factor    
            bispectrum_noise_with_B = QR.get_get_bispectrum_mixed_withB()(key)*bispectrum_factor if bispectrum_factor != 0 else 0
            bispectrum_noise = QR.get_get_bispectrum_mixed()(key)*bispectrum_factor if bispectrum_factor != 0 else 0
            trispectrum_noise = (QR.get_get_trispectrum()(key))*trispectrum_factor if trispectrum_factor != 0 else 0
            normalization = QR.get_get_norm()(key)
            
            responses = qres.get_normalization_array(key, keys, Ks, out_normalization_AB)
                
            CRR = lambda v: reconstruction_auto_spectrum(v, plinear, normalization, responses, variance_noise, trispectrum_noise, asymm_shift = asymm_shift, growth_eps = growth_eps, only_asymm_shift = only_asymm_shift) #gets reconstruction auto-spectrum

            CAA = lambda v: galaxy_power_spectrum(v[:2], plinear, shot_noise_A, growth_eps = growth_eps) #gets galaxy auto-spectrum
            CAR = lambda v, b1X: reconstruction_cross_spectrum(v, b1X, plinear, normalization, responses, bispectrum_noise, asymm_shift = asymm_shift, growth_eps = growth_eps, only_asymm_shift = only_asymm_shift) #gets cross-spectrum
            
            CBB = lambda v: galaxy_power_spectrum(v[:2], plinear, shot_noise_B, growth_eps = growth_eps)
            CBR = lambda v, b1X: reconstruction_cross_spectrum(v, b1X, plinear, normalization, responses, bispectrum_noise_with_B, asymm_shift = asymm_shift, growth_eps = growth_eps, only_asymm_shift = only_asymm_shift) #gets cross-spectrum

            CAB = lambda vA, vB: cross_galaxy_power_spectrum(vA[:2], vB[:2], plinear, cross_shot = shot_noise_AB, growth_eps = growth_eps)

            return CAA, CBB, CRR, CAB, CAR, CBR

        else:
            normalization_2 = QR.get_get_norm()(key2)
            normalization = QR.get_get_norm()(key)

            responses = qres.get_normalization_array(key, keys, Ks, out_normalization_AB)
            responses_2 = qres.get_normalization_array(key2, keys, Ks, out_normalization_AB)
            
            variance_noise_cross = QR.get_get_variance()(key, key2)*variance_factor    
            trispectrum_noise_cross = QR.get_get_trispectrum()(key, key2)*trispectrum_factor if trispectrum_factor != 0 else 0
            CRR_key_key2 = lambda v: reconstruction_auto_spectrum(v, plinear, normalization, responses, variance_noise_cross, trispectrum_noise_cross, 
                                                                asymm_shift = asymm_shift, growth_eps = growth_eps, only_asymm_shift = only_asymm_shift,
                                                                normalization_2 = normalization_2, responses_2 = responses_2)
            return CRR_key_key2
        

    def get_all_fisher(self, v_fiducial, Ks, key, key_alt = None, relative_path = "../."):

        QR = self.QR #qres.QEResults(self.config, relative_path = relative_path)

        key2 = None

        CAA, CBB, CRR, CAB, CAR, CBR = self.get_functions(QR, key, key2 = key2)
        ps_joint_full = [CAA, CBB, CRR, CAB, CAR, CBR]

        out_results = {}

    
        #AR only
        F_AA_cross_base = fisher.fisher_per_mode_single_with_covariance(v_fiducial, Ks, *get_Ofuncs_A_cross_base(CAA, CRR, CAR))
        F_AA_cross_base = jnp.nan_to_num(F_AA_cross_base)
        out_results["F_AA_cross_base"] = F_AA_cross_base

        #AR, BR
        F_cross_base = fisher.fisher_per_mode_single_with_covariance(v_fiducial, Ks, *get_Ofuncs_AB_cross_base(CAA, CBB, CRR, CAB, CAR, CBR))
        F_cross_base = jnp.nan_to_num(F_cross_base)
        out_results["F_cross_base"] = F_cross_base

        #AR, BR, RR
        F_joint_base = fisher.fisher_per_mode_single_with_covariance(v_fiducial, Ks, *get_Ofuncs_AB_joint_base(CAA, CBB, CRR, CAB, CAR, CBR))
        F_joint_base = jnp.nan_to_num(F_joint_base)
        out_results["F_joint_base"] = F_joint_base

        #AB, AR, BR
        F_cross = fisher.fisher_per_mode_single_with_covariance(v_fiducial, Ks, *get_Ofuncs_AB_cross(CAA, CBB, CRR, CAB, CAR, CBR))
        F_cross = jnp.nan_to_num(F_cross)
        out_results["F_cross"] = F_cross

        #AA, BB, AB, AR, BR
        F_cross_full = fisher.fisher_per_mode_single_with_covariance(v_fiducial, Ks, *get_Ofuncs_AB_cross_full(CAA, CBB, CRR, CAB, CAR, CBR))
        F_cross_full = jnp.nan_to_num(F_cross_full)
        out_results["F_cross_full"] = F_cross_full

        #AB, AR, BR, RR
        F_joint = fisher.fisher_per_mode_single_with_covariance(v_fiducial, Ks, *get_Ofuncs_AB_joint(CAA, CBB, CRR, CAB, CAR, CBR))
        F_joint = jnp.nan_to_num(F_joint)
        out_results["F_joint"] = F_joint

        #AA, BB, AB, AR, BR, RR
        F_joint_full = fisher.fisher_per_mode_single_with_covariance(v_fiducial, Ks, *get_Ofuncs_AB_joint_full(CAA, CBB, CRR, CAB, CAR, CBR))
        F_joint_full = jnp.nan_to_num(F_joint_full)
        out_results["F_joint_full"] = F_joint_full

        #AA, BB, AB, AR, BR, RR, but with equivalent trace calculation
        F_joint_full_trace = fisher.fisher_per_mode(v_fiducial, Ks, get_cov_with_AB(*ps_joint_full))
        F_joint_full_trace = jnp.nan_to_num(F_joint_full_trace)
        out_results["F_joint_full_trace"] = F_joint_full_trace


        ps_joint_A = [CAA, CRR, CAR]
        F_joint_full_trace_A_only = fisher.fisher_per_mode(v_fiducial, Ks, get_cov(*ps_joint_A))
        F_joint_full_trace_A_only = jnp.nan_to_num(F_joint_full_trace_A_only)
        out_results["F_joint_full_trace_A_only"] = F_joint_full_trace_A_only

        if key_alt is not None:
            CAAg, CBBg, CRRg, CABg, CARg, CBRg = self.get_functions(QR, key_alt, key2 = key2)
            _, _, CR2R2, _, CAR2, CBR2 = self.get_functions(QR, key_alt, key2 = key2)
            CRR2 = self.get_functions(QR, key, key2 = key_alt, wrapped = False)

            ps_joint_full_growth_only = [CAAg, CBBg, CRRg, CABg, CARg, CBRg]

            F_joint_full_trace_growth_only = fisher.fisher_per_mode(v_fiducial, Ks, get_cov_with_AB(*ps_joint_full_growth_only))
            F_joint_full_trace_growth_only = jnp.nan_to_num(F_joint_full_trace_growth_only)
            out_results["F_joint_full_trace_growth_only"] = F_joint_full_trace_growth_only

            
            F_growth_joint_tr = fisher.fisher_per_mode(v_fiducial, Ks, get_cov_with_AB_R_R2(CAA, CBB, CRR, CR2R2, CAB, CAR, CBR, CAR2, CBR2, CRR2))
            F_growth_joint_tr = jnp.nan_to_num(F_growth_joint_tr)
            out_results["F_growth_joint_tr"] = F_growth_joint_tr

            ps_joint_A_growth_only = [CAAg, CRRg, CARg]
            F_joint_full_trace_growth_only_A_only = fisher.fisher_per_mode(v_fiducial, Ks, get_cov(*ps_joint_A_growth_only))
            F_joint_full_trace_growth_only_A_only = jnp.nan_to_num(F_joint_full_trace_growth_only_A_only)
            out_results["F_joint_full_trace_growth_only_A_only"] = F_joint_full_trace_growth_only_A_only

            #AB, AR, BR, AG, BG, RR, GG
            F_joint = fisher.fisher_per_mode_single_with_covariance(v_fiducial, Ks, *get_Ofuncs_AB_R_R2_joint(CAA, CBB, CRR, CR2R2, CAB, CAR, CBR, CAR2, CBR2, CRR2))
            F_joint = jnp.nan_to_num(F_joint)
            out_results["F_joint_with_growth"] = F_joint

            #AB, AG, BG
            F_cross = fisher.fisher_per_mode_single_with_covariance(v_fiducial, Ks, *get_Ofuncs_AB_cross(CAA, CBB, CR2R2, CAB, CAR2, CBR2))
            F_cross = jnp.nan_to_num(F_cross)
            out_results["F_cross_with_growth"] = F_cross

            #AR2, BR2
            F_cross_base = fisher.fisher_per_mode_single_with_covariance(v_fiducial, Ks, *get_Ofuncs_AB_cross_base(CAA, CBB, CR2R2, CAB, CAR2, CBR2))
            F_cross_base = jnp.nan_to_num(F_cross_base)
            out_results["F_cross_base_growth"] = F_cross_base

            #AR2, BR2, R2R2
            F_joint_base = fisher.fisher_per_mode_single_with_covariance(v_fiducial, Ks, *get_Ofuncs_AB_joint_base(CAA, CBB, CR2R2, CAB, CAR2, CBR2))
            F_joint_base = jnp.nan_to_num(F_joint_base)
            out_results["F_joint_base_growth"] = F_joint_base

            #AR2 only
            F_AA_cross_base = fisher.fisher_per_mode_single_with_covariance(v_fiducial, Ks, *get_Ofuncs_A_cross_base(CAA, CR2R2, CAR2))
            F_AA_cross_base = jnp.nan_to_num(F_AA_cross_base)
            out_results["F_AA_cross_base_growth"] = F_AA_cross_base

            #RR2 only
            F_RR_cross_base = fisher.fisher_per_mode_single_with_covariance(v_fiducial, Ks, *get_Ofuncs_A_cross_base(CRR, CR2R2, CRR2))
            F_RR_cross_base = jnp.nan_to_num(F_RR_cross_base)
            out_results["F_RR_cross_base_growth"] = F_RR_cross_base

        return out_results


def get_v(vA):
    e, b1A, b2A, bs2A, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB = vA
    b1B, b2B, bs2B = b1A, b2A, bs2A
    v = jnp.array([e, b1A, b1B, b2A, b2B, bs2A, bs2B, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB])
    return v

def get_vA_vB(v):
    #v = jnp.array([e, b1A, b1B, b2A, b2B, bs2A, bs2B, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB])
    
    #what we do is the following, to write less code
    #we create a vector for A
    #then, we create for B, just swapping some places
    
    e, b1A, b1B, b2A, b2B, bs2A, bs2B, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB = v
    vA = v #ok, this is my default ordering
    vB = jnp.array([e, b1B, b1A, b2B, b2A, bs2B, bs2A, bGXB, bGXA, bSXB, bSXA, bTXB, bTXA])
    return vA, vB


def get_vB_from_vA(v):
    #v = jnp.array([e, b1A, b1B, b2A, b2B, bs2A, bs2B, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB])
    e, b1A, b1B, b2A, b2B, bs2A, bs2B, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB = v
    vB = jnp.array([e, b1B, b1A, b2B, b2A, bs2B, bs2A, bGXB, bGXA, bSXB, bSXA, bTXB, bTXA])
    return vB

def get_functions_vA_only(CAA, CRR, CAR):
    """
    This assumes I only have vA as my input when studying stuff
    vA = e, b1A, b2A, bs2A, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB
    """
        
    def CAA_(vA):
        return CAA(get_v(vA))

    def CAR_(vA):
            return CAR(get_v(vA))

    def CRR_(vA):
            return CRR(get_v(vA))

    return CAA_, CRR_, CAR_


def get_functions_A(CAA, CRR, CAR):
    """
    This assumes full
    v = jnp.array([e, b1A, b1B, b2A, b2B, bs2A, bs2B, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB])
    """
        
    def CAA_(v):
        return CAA(v)

    def CAR_(v):
            return CAR(v, v[1])

    def CRR_(v):
            return CRR(v)

    return CAA_, CRR_, CAR_

def get_functions_vA_vB(CAA, CBB, CRR, CAB, CAR, CBR, v_function = lambda v: v):
    """
    This assumes I have the full v for A and B as my input when studying stuff.
    v = jnp.array([e, b1A, b1B, b2A, b2B, bs2A, bs2B, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB])

    I assume by convention that A follows that ordering.
    """
    
    def CAA_(v):
        return CAA(v_function(v))

    def CAR_(v):
            return CAR(v_function(v), v_function(v)[1])

    def CRR_(v):
            return CRR(v_function(v))

    def CBB_(v):
        return CBB(get_vB_from_vA(v_function(v)))

    def CBR_(v):
            return CBR(v_function(v), get_vB_from_vA(v_function(v))[1])

    def CAB_(v):
            return CAB(v_function(v), get_vB_from_vA(v_function(v)))

    return CAA_, CBB_, CRR_, CAB_, CAR_, CBR_ 


def get_cov(CAAf, CRRf, CARf):
    @jax.jit
    def covariance_full(K_array, vA):

        n_probes = 2

        C = jnp.zeros((len(K_array), n_probes, n_probes))
        
        CAR = CARf(vA)
        
        C = C.at[:, 0, 0].set(CRRf(vA))
        C = C.at[:, 0, 1].set(CAR)
        C = C.at[:, 1, 0].set(CAR)
        C = C.at[:, 1, 1].set(CAAf(vA))
        
        return C
    
    return covariance_full


def get_cov_with_AB(CAAf, CBBf, CRRf, CABf, CARf, CBRf):
    @jax.jit
    def covariance_full(K_array, v):

        n_probes = 3

        C = jnp.zeros((len(K_array), n_probes, n_probes))
        
        CAR = CARf(v)
        CBR = CBRf(v)
        CAB = CABf(v)
        
        C = C.at[:, 0, 0].set(CRRf(v))
        C = C.at[:, 0, 1].set(CAR)
        C = C.at[:, 1, 0].set(CAR)
        C = C.at[:, 0, 2].set(CBR)
        C = C.at[:, 2, 0].set(CBR)
        C = C.at[:, 1, 1].set(CAAf(v))
        C = C.at[:, 1, 2].set(CAB)
        C = C.at[:, 2, 1].set(CAB)
        C = C.at[:, 2, 2].set(CBBf(v))
        
        return C
    
    return covariance_full

def get_cov_with_AB_R_R2(CAAf, CBBf, CRRf, CR2R2f, CABf, CARf, CBRf, 
                         CAR2f, CBR2f, CRR2f):
    @jax.jit
    def covariance_full(K_array, v):

        n_probes = 4

        C = jnp.zeros((len(K_array), n_probes, n_probes))
        
        CAR = CARf(v)
        CBR = CBRf(v)
        CAB = CABf(v)
        
        CAR2 = CAR2f(v)
        CBR2 = CBR2f(v)
        CRR2 = CRR2f(v)
        CR2R2 = CR2R2f(v)
        
        C = C.at[:, 0, 0].set(CRRf(v))
        C = C.at[:, 0, 1].set(CAR)
        C = C.at[:, 1, 0].set(CAR)
        C = C.at[:, 0, 2].set(CBR)
        C = C.at[:, 2, 0].set(CBR)
        C = C.at[:, 1, 1].set(CAAf(v))
        C = C.at[:, 1, 2].set(CAB)
        C = C.at[:, 2, 1].set(CAB)
        C = C.at[:, 2, 2].set(CBBf(v))

        C = C.at[:, 1, 3].set(CAR2)
        C = C.at[:, 3, 1].set(CAR2)
        C = C.at[:, 2, 3].set(CBR2)
        C = C.at[:, 3, 2].set(CBR2)
        C = C.at[:, 3, 0].set(CRR2)
        C = C.at[:, 0, 3].set(CRR2)
        C = C.at[:, 3, 3].set(CR2R2)
        
        return C
    
    return covariance_full


def get_cov_with_AB_ps_only(CAAf, CBBf, CABf):
    @jax.jit
    def covariance_full(K_array, v):

        n_probes = 2

        C = jnp.zeros((len(K_array), n_probes, n_probes))

        CAB = CABf(v)
        
        C = C.at[:, 0, 0].set(CAAf(v))
        C = C.at[:, 0, 1].set(CAB)
        C = C.at[:, 1, 0].set(CAB)
        C = C.at[:, 1, 1].set(CBBf(v))
        
        return C
    
    return covariance_full




def get_Ofuncs_A_cross_base(CAAf, CRRf, CARf):
    def Ofunc(K_arr, v):
        return jnp.array([CARf(v)]).T
    def variance_func_cross_cov(K_arr, v):
        C = jnp.zeros((CAAf(v).shape[0], 1, 1))
        C = C.at[:, 0, 0].set(CAAf(v)*CRRf(v)+CARf(v)**2) 
        return C
    
    return Ofunc, variance_func_cross_cov

def get_Ofuncs_AB_cross_base(CAAf, CBBf, CRRf, CABf, CARf, CBRf):
    def Ofunc(K_arr, v):
        return jnp.array([CARf(v), CBRf(v)]).T
    def variance_func_cross_cov(K_arr, v):
        C = jnp.zeros((CAAf(v).shape[0], 2, 2))
        C = C.at[:, 0, 0].set(CAAf(v)*CRRf(v)+CARf(v)**2) 
        C = C.at[:, 1, 1].set(CBBf(v)*CRRf(v)+CBRf(v)**2)

        diag = CABf(v)*CRRf(v)+CARf(v)*CBRf(v) #AR-BR
        C = C.at[:, 0, 1].set(diag) 
        C = C.at[:, 1, 0].set(diag) 

        return C
    
    return Ofunc, variance_func_cross_cov


def get_Ofuncs_AB_joint_base(CAAf, CBBf, CRRf, CABf, CARf, CBRf):

    def Ofunc(K_arr, v):
        return jnp.array([CARf(v), CBRf(v), CRRf(v)]).T
    
    def variance_func_joint_cov(K_arr, v):
        C = jnp.zeros((CAAf(v).shape[0], 3, 3))
        C = C.at[:, 0, 0].set(CAAf(v)*CRRf(v)+CARf(v)**2) 
        C = C.at[:, 1, 1].set(CBBf(v)*CRRf(v)+CBRf(v)**2)
        C = C.at[:, 2, 2].set(2*CRRf(v)**2)

        diag = CABf(v)*CRRf(v)+CARf(v)*CBRf(v) #AR-BR
        C = C.at[:, 0, 1].set(diag) 
        C = C.at[:, 1, 0].set(diag) 

        diag = 2*CARf(v)*CRRf(v) #RR-AR
        C = C.at[:, 0, 2].set(diag) 
        C = C.at[:, 2, 0].set(diag) 

        diag = 2*CBRf(v)*CRRf(v) #RR-BR
        C = C.at[:, 1, 2].set(diag) 
        C = C.at[:, 2, 1].set(diag) 

        return C

    return Ofunc, variance_func_joint_cov

def get_Ofuncs_AB_cross(CAAf, CBBf, CRRf, CABf, CARf, CBRf):
    def Ofunc(K_arr, v):
        return jnp.array([CABf(v), CARf(v), CBRf(v)]).T
    def variance_func_cross_cov(K_arr, v):
        C = jnp.zeros((CAAf(v).shape[0], 3, 3))
        C = C.at[:, 0, 0].set(CAAf(v)*CBBf(v)+CABf(v)**2)
        C = C.at[:, 1, 1].set(CAAf(v)*CRRf(v)+CARf(v)**2) 
        C = C.at[:, 2, 2].set(CBBf(v)*CRRf(v)+CBRf(v)**2)

        diag = CAAf(v)*CBRf(v)+CARf(v)*CABf(v) #AB-AR
        C = C.at[:, 0, 1].set(diag) 
        C = C.at[:, 1, 0].set(diag) 

        diag = CBBf(v)*CARf(v)+CBRf(v)*CABf(v) #AB-BR
        C = C.at[:, 0, 2].set(diag) 
        C = C.at[:, 2, 0].set(diag) 

        diag = CABf(v)*CRRf(v)+CARf(v)*CBRf(v) #AR-BR
        C = C.at[:, 1, 2].set(diag) 
        C = C.at[:, 2, 1].set(diag) 
        return C
    
    return Ofunc, variance_func_cross_cov


def get_Ofuncs_AB_joint(CAAf, CBBf, CRRf, CABf, CARf, CBRf):
    def Ofunc(K_arr, v):
        return jnp.array([CRRf(v), CABf(v), CARf(v), CBRf(v)]).T
    def variance_func_cross_cov(K_arr, v):
        C = jnp.zeros((CAAf(v).shape[0], 4, 4))

        C = C.at[:, 0, 0].set(2*CRRf(v)**2)
        C = C.at[:, 1, 1].set(CAAf(v)*CBBf(v)+CABf(v)**2)
        C = C.at[:, 2, 2].set(CAAf(v)*CRRf(v)+CARf(v)**2) 
        C = C.at[:, 3, 3].set(CBBf(v)*CRRf(v)+CBRf(v)**2)

        diag = 2*CARf(v)*CBRf(v) #RR-AB
        C = C.at[:, 0, 1].set(diag) 
        C = C.at[:, 1, 0].set(diag) 

        diag = 2*CARf(v)*CRRf(v) #RR-AR
        C = C.at[:, 0, 2].set(diag) 
        C = C.at[:, 2, 0].set(diag) 

        diag = 2*CBRf(v)*CRRf(v) #RR-BR
        C = C.at[:, 0, 3].set(diag) 
        C = C.at[:, 3, 0].set(diag) 

        diag = CAAf(v)*CBRf(v)+CARf(v)*CABf(v) #AB-AR
        C = C.at[:, 1, 2].set(diag) 
        C = C.at[:, 2, 1].set(diag) 

        diag = CBBf(v)*CARf(v)+CBRf(v)*CABf(v) #AB-BR
        C = C.at[:, 1, 3].set(diag) 
        C = C.at[:, 3, 1].set(diag) 

        diag = CABf(v)*CRRf(v)+CARf(v)*CBRf(v) #AR-BR
        C = C.at[:, 2, 3].set(diag) 
        C = C.at[:, 3, 2].set(diag) 
        
        return C
    
    return Ofunc, variance_func_cross_cov


def get_Ofuncs_AB_R_R2_joint(CAAf, CBBf, CRRf, CR2R2f, CABf, CARf, CBRf, CAR2f, CBR2f, CRR2f):
    @jax.jit
    def Ofunc(K_arr, v):
        return jnp.array([CRRf(v), CR2R2f(v), CABf(v), CARf(v), CBRf(v), CAR2f(v), CBR2f(v)]).T
    
    @jax.jit
    def variance_func_cross_cov(K_arr, v):
        C = jnp.zeros((CAAf(v).shape[0], 7, 7))
        
        # Diagonal terms
        C = C.at[:, 0, 0].set(2*CRRf(v)**2)  # RR-RR
        C = C.at[:, 1, 1].set(2*CR2R2f(v)**2)  # R2R2-R2R2
        C = C.at[:, 2, 2].set(CAAf(v)*CBBf(v) + CABf(v)**2)  # AB-AB
        C = C.at[:, 3, 3].set(CAAf(v)*CRRf(v) + CARf(v)**2)  # AR-AR
        C = C.at[:, 4, 4].set(CBBf(v)*CRRf(v) + CBRf(v)**2)  # BR-BR
        C = C.at[:, 5, 5].set(CAAf(v)*CR2R2f(v) + CAR2f(v)**2)  # AR2-AR2
        C = C.at[:, 6, 6].set(CBBf(v)*CR2R2f(v) + CBR2f(v)**2)  # BR2-BR2

        # Cross terms with RR
        C = C.at[:, 0, 1].set(2*CRR2f(v)**2)  # RR-R2R2
        C = C.at[:, 0, 2].set(CARf(v)*CBRf(v))  # RR-AB
        C = C.at[:, 0, 3].set(2*CARf(v)*CRRf(v))  # RR-AR
        C = C.at[:, 0, 4].set(2*CBRf(v)*CRRf(v))  # RR-BR
        C = C.at[:, 0, 5].set(CARf(v)*CRR2f(v) + CAR2f(v)*CRRf(v))  # RR-AR2
        C = C.at[:, 0, 6].set(CBRf(v)*CRR2f(v) + CBR2f(v)*CRRf(v))  # RR-BR2

        # Cross terms with R2R2
        C = C.at[:, 1, 2].set(CAR2f(v)*CBR2f(v))  # R2R2-AB
        C = C.at[:, 1, 3].set(CAR2f(v)*CRR2f(v))  # R2R2-AR
        C = C.at[:, 1, 4].set(CBR2f(v)*CRR2f(v))  # R2R2-BR
        C = C.at[:, 1, 5].set(2*CAR2f(v)*CR2R2f(v))  # R2R2-AR2
        C = C.at[:, 1, 6].set(2*CBR2f(v)*CR2R2f(v))  # R2R2-BR2

        # Cross terms with AB
        C = C.at[:, 2, 3].set(CAAf(v)*CBRf(v) + CARf(v)*CABf(v))  # AB-AR
        C = C.at[:, 2, 4].set(CBBf(v)*CARf(v) + CBRf(v)*CABf(v))  # AB-BR
        C = C.at[:, 2, 5].set(CAAf(v)*CBR2f(v) + CAR2f(v)*CABf(v))  # AB-AR2
        C = C.at[:, 2, 6].set(CBBf(v)*CAR2f(v) + CBR2f(v)*CABf(v))  # AB-BR2

        # Cross terms with AR
        C = C.at[:, 3, 4].set(CABf(v)*CRRf(v) + CARf(v)*CBRf(v))  # AR-BR
        C = C.at[:, 3, 5].set(CAAf(v)*CRR2f(v) + CARf(v)*CAR2f(v))  # AR-AR2
        C = C.at[:, 3, 6].set(CABf(v)*CRR2f(v) + CARf(v)*CBR2f(v))  # AR-BR2

        # Cross terms with BR
        C = C.at[:, 4, 5].set(CABf(v)*CRR2f(v) + CBRf(v)*CAR2f(v))  # BR-AR2
        C = C.at[:, 4, 6].set(CBBf(v)*CRR2f(v) + CBRf(v)*CBR2f(v))  # BR-BR2

        # Cross terms with AR2
        C = C.at[:, 5, 6].set(CABf(v)*CR2R2f(v) + CAR2f(v)*CBR2f(v))  # AR2-BR2

        # Fill lower triangle
        for i in range(7):
            for j in range(i):
                C = C.at[:, i, j].set(C[:, j, i])

        return C
    
    return Ofunc, variance_func_cross_cov


def get_Ofuncs_AB_cross_full(CAAf, CBBf, CRRf, CABf, CARf, CBRf):
    def Ofunc(K_arr, v):
        return jnp.array([CAAf(v), CBBf(v), CABf(v), CARf(v), CBRf(v)]).T 
    def variance_func_cross_cov(K_arr, v):
        C = jnp.zeros((CAAf(v).shape[0], 5, 5))
        C = C.at[:, 0, 0].set(2*CAAf(v)**2)
        C = C.at[:, 1, 1].set(2*CBBf(v)**2) 
        C = C.at[:, 2, 2].set(CAAf(v)*CBBf(v)+CABf(v)**2) 
        C = C.at[:, 3, 3].set(CRRf(v)*CAAf(v)+CARf(v)**2) 
        C = C.at[:, 4, 4].set(CRRf(v)*CBBf(v)+CBRf(v)**2) 

        #cross cov with AA
        C = C.at[:, 0, 1].set(2*CABf(v)**2) #AA-BB
        C = C.at[:, 1, 0].set(C[:, 0, 1])

        C = C.at[:, 0, 2].set(2*CAAf(v)*CABf(v)) #AA-AB
        C = C.at[:, 2, 0].set(C[:, 0, 2])

        C = C.at[:, 0, 3].set(2*CAAf(v)*CARf(v)) #AA-AR
        C = C.at[:, 3, 0].set(C[:, 0, 3])

        C = C.at[:, 0, 4].set(2*CABf(v)*CARf(v)) #AA-BR
        C = C.at[:, 4, 0].set(C[:, 0, 4])


        #cross cov with BB

        C = C.at[:, 1, 2].set(2*CBBf(v)*CABf(v)) #BB-AB
        C = C.at[:, 2, 1].set(C[:, 1, 2])

        C = C.at[:, 1, 3].set(2*CABf(v)*CBRf(v)) #BB-AR
        C = C.at[:, 3, 1].set(C[:, 1, 3])

        C = C.at[:, 1, 4].set(2*CBBf(v)*CBRf(v)) #BB-BR
        C = C.at[:, 4, 1].set(C[:, 1, 4])

        #cross cov with AB

        C = C.at[:, 2, 3].set(CAAf(v)*CBRf(v)+CARf(v)*CABf(v)) #AB-AR
        C = C.at[:, 3, 2].set(C[:, 2, 3])

        C = C.at[:, 2, 4].set(CABf(v)*CBRf(v)+CARf(v)*CBBf(v)) #AB-BR
        C = C.at[:, 4, 2].set(C[:, 2, 4])

        #cross cov with AR

        C = C.at[:, 3, 4].set(CABf(v)*CRRf(v)+CARf(v)*CBRf(v)) #AR-BR
        C = C.at[:, 4, 3].set(C[:, 3, 4])
        
        return C
    
    return Ofunc, variance_func_cross_cov


def get_Ofuncs_AB_joint_full(CAAf, CBBf, CRRf, CABf, CARf, CBRf):
    @jax.jit
    def Ofunc(K_arr, v):
        return jnp.array([CAAf(v), CBBf(v), CRRf(v), CABf(v), CARf(v), CBRf(v)]).T
    @jax.jit
    def variance_func_cross_cov(K_arr, v):
        C = jnp.zeros((CAAf(v).shape[0], 6, 6))

        C = C.at[:, 0, 0].set(2*CAAf(v)**2) 
        C = C.at[:, 1, 1].set(2*CBBf(v)**2) 
        C = C.at[:, 3, 3].set(CABf(v)**2+CAAf(v)*CBBf(v))  #cov of AB
        C = C.at[:, 4, 4].set(CAAf(v)*CRRf(v)+CARf(v)**2)  #cov of AR
        C = C.at[:, 5, 5].set(CBBf(v)*CRRf(v)+CBRf(v)**2)  #cov of BR
        C = C.at[:, 2, 2].set(2*CRRf(v)**2)  #cov of RR

        diag = 2*CABf(v)**2 #cov AA - BB
        C = C.at[:, 0, 1].set(diag) 
        C = C.at[:, 1, 0].set(diag) 

        diag = 2*CABf(v)*CAAf(v) #cov AA - AB
        C = C.at[:, 0, 3].set(diag) 
        C = C.at[:, 3, 0].set(diag) 

        diag = 2*CARf(v)*CAAf(v) #cov AA - AR
        C = C.at[:, 0, 4].set(diag) 
        C = C.at[:, 4, 0].set(diag) 

        diag = 2*CARf(v)*CABf(v) #cov AA - BR
        C = C.at[:, 0, 5].set(diag) 
        C = C.at[:, 5, 0].set(diag) 

        diag = 2*CARf(v)**2 #cov AA - RR
        C = C.at[:, 0, 2].set(diag) 
        C = C.at[:, 2, 0].set(diag) 

        diag = 2*CABf(v)*CBBf(v) #cov BB - AB
        C = C.at[:, 1, 3].set(diag) 
        C = C.at[:, 3, 1].set(diag) 

        diag = 2*CBRf(v)*CABf(v) #cov BB - AR
        C = C.at[:, 1, 4].set(diag) 
        C = C.at[:, 4, 1].set(diag) 

        diag = 2*CBRf(v)*CBBf(v) #cov BB - BR
        C = C.at[:, 1, 5].set(diag) 
        C = C.at[:, 5, 1].set(diag) 

        diag = 2*CBRf(v)**2 #cov BB - RR
        C = C.at[:, 1, 2].set(diag) 
        C = C.at[:, 2, 1].set(diag) 

        diag = CAAf(v)*CBRf(v)+CARf(v)*CABf(v) #cov AB - AR
        C = C.at[:, 3, 4].set(diag) 
        C = C.at[:, 4, 3].set(diag) 

        diag = CABf(v)*CBRf(v)+CARf(v)*CBBf(v) #cov AB - BR
        C = C.at[:, 3, 5].set(diag) 
        C = C.at[:, 5, 3].set(diag) 

        diag = 2*CARf(v)*CBRf(v) #cov AB - RR
        C = C.at[:, 3, 2].set(diag) 
        C = C.at[:, 2, 3].set(diag) 

        diag = CARf(v)*CBRf(v)+CRRf(v)*CABf(v) #cov AR - BR
        C = C.at[:, 4, 5].set(diag) 
        C = C.at[:, 5, 4].set(diag) 

        diag = 2*CARf(v)*CRRf(v) #cov AR - RR
        C = C.at[:, 4, 2].set(diag) 
        C = C.at[:, 2, 4].set(diag) 

        diag = 2*CBRf(v)*CRRf(v) #cov BR - RR
        C = C.at[:, 5, 2].set(diag) 
        C = C.at[:, 2, 5].set(diag) 

        return C
    
    return Ofunc, variance_func_cross_cov



def get_Ofuncs(CAAf, CRRf, CARf):
    Ofunc = lambda K_arr, v: CARf(v)
    variance_func = lambda K_arr, v: CAAf(v)*CRRf(v)+CARf(v)**2
    return Ofunc, variance_func

def get_Ofuncs_A_cov(CAAf, CRRf, CARf):
    def Ofunc(K_arr, v):
        return jnp.array([CARf(v)]).T
    def variance_func_cross_cov(K_arr, v):
        C = jnp.zeros((CAAf(v).shape[0], 1, 1))
        C = C.at[:, 0, 0].set(CAAf(v)*CRRf(v)+CARf(v)**2) 
        return C  
    return Ofunc, variance_func_cross_cov


def get_Ofuncs_AB_cross_no_AB(CAAf, CBBf, CRRf, CABf, CARf, CBRf):
    def Ofunc(K_arr, v):
        return jnp.array([CARf(v), CBRf(v)]).T
    def variance_func_cross_cov(K_arr, v):
        C = jnp.zeros((CAAf(v).shape[0], 2, 2))
        C = C.at[:, 0, 0].set(CAAf(v)*CRRf(v)+CARf(v)**2) 
        C = C.at[:, 1, 1].set(CBBf(v)*CRRf(v)+CBRf(v)**2)
        diag = CABf(v)*CRRf(v)+CARf(v)*CBRf(v)
        C = C.at[:, 0, 1].set(diag) 
        C = C.at[:, 1, 0].set(diag) 
        return C
    
    return Ofunc, variance_func_cross_cov