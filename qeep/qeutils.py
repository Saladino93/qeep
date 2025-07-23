"""
TO DO:
- in principle for each key I can re-use the same grid for all the modes, so I can save time!!
Check torchquad documentation, https://torchquad.readthedocs.io/en/latest/tutorial.html#reusing-sample-points
-Save points
-Re-use saved points
"""
import numpy as np
import jax
import jax.numpy as jnp
from interpax import interp1d
from torchquad import MonteCarlo, Boole
from torchquad import set_up_backend  # Necessary to enable GPU support
set_up_backend("jax", data_type="float64")

import vegas


@jax.jit
def Fs(k1, k2, k1_mag, k2_mag):
  return 0.5*(1/k1_mag**2+1/k2_mag**2)*dot(k1, k2)

@jax.jit
def Ft(k1, k2, k1_mag, k2_mag):
  return 2/7*(mu(k1, k2, k1_mag, k2_mag)**2-1/3)

@jax.jit
def Fg(k1, k2, k1_mag, k2_mag):
  return 17/21

def mu(k1, k2, k1_mag, k2_mag):
  return dot(k1, k2)/(k1_mag*k2_mag)

def symmetric_part(fAB, fBA):
  return 0.5*(fAB+fBA)

def antisymmetric_part(fAB, fBA):
  return 0.5*(fAB-fBA)

def get_interpolated_standard(k, p):

  x_data_jax = jnp.array(k)
  y_data_jax = jnp.array(p)

  @jax.jit
  def interpolate_function(x_new):
      # For 1D linear interpolation
      return jnp.interp(x_new, x_data_jax, y_data_jax)

  return interpolate_function

def get_interpolated(k, p):

  x_data_jax = jnp.log(jnp.array(k))
  y_data_jax = jnp.log(jnp.array(p))

  @jax.jit
  def interpolate_function(x_new):
      # For 1D linear interpolation
      return jnp.exp(jnp.interp(jnp.log(x_new), x_data_jax, y_data_jax))

  return interpolate_function


def integrate(Ks, function, batch_size = 5):
  """
  Integrate function for each mode K in Ks.
  """
  #batch_size = 5 # if Ks.size > 20 else 10
  results = []
  for i in range(0, len(Ks), batch_size):
      batch_Ks = jnp.array(Ks[i:i+batch_size])
      batch_results = jax.vmap(function)(batch_Ks)
      results.append(batch_results)
  results = jnp.concatenate(results)
  return results


def integrate_vegas(Ks, function, batch_size = 3, kmin = 0.015, kmax = 0.15):
  """
  Integrate function for each mode K in Ks.
  """
  #return jnp.array([function(K) for K in Ks])
  results = []
  for K in Ks:    
    integrand_ = function(K)
    integ = vegas.Integrator([[0, 2*jnp.pi], [-1, 1], [kmin, kmax], [0, 2*jnp.pi], [-1, 1], [kmin, kmax]], gpu_pad = True)
    integ(integrand_, nitn=20, neval=1e3)
    result = integ(integrand_, nitn=20, neval=1e5, adapt=True)
    print(result.mean, result.sdev)
    results.append(result.mean)
  results = jnp.array(results)
  return results


def set_up_fixed_grid(N, dim, integration_domain):
  integration_domain = jnp.array(integration_domain)
  integrator = MonteCarlo()
  grid_points, hs, n_per_dim = integrator.calculate_grid(N, integration_domain)


def dot(k1, k2):
   return jnp.sum(k1*k2, axis=-1)

def cosine_angle(k1, k2, k1_mag, k2_mag):
    dot_product = dot(k1, k2)
    cos_angle = dot_product / (k1_mag * k2_mag)
    return cos_angle


def spherical_to_cartesian(phi, mu, k_mag):
    """
    Convert from spherical coordinates to Cartesian coordinates.

    Parameters:
    phi: azimuthal angle (in the x-y plane)
    mu: cosine of the polar angle (mu = cos(theta))
    k_mag: magnitude of the vector

    Returns:
    k_vec: Cartesian coordinates (x, y, z)
    """
    # Calculate sine of the polar angle from mu
    sin_theta = jnp.sqrt(1 - mu**2)

    # Calculate the Cartesian components
    x = k_mag * sin_theta * jnp.cos(phi)
    y = k_mag * sin_theta * jnp.sin(phi)
    z = k_mag * mu

    # Stack the components into a single vector
    k_vec = jnp.c_[x, y, z]

    return k_vec



def maskf(k, kmin, kmax):
  return (k>=kmin) & (k<=kmax)

def get_k1_k2_mask(K, k, phi, mu, kmin, kmax, gauss_filter = False, K_sign = 1):

  K_minus_k = jnp.sqrt(K**2+k**2-2*K*k*mu*K_sign)

  K = spherical_to_cartesian(phi*0, mu*0+1, K) #[0, 0, K], aligned with z-axis
  k1 = spherical_to_cartesian(phi, mu, k)
  k2 = K_sign*K-k1

  if not gauss_filter:
    mask = (k>=kmin) & (k<=kmax)
    mask_2 = (K_minus_k>=kmin) & (K_minus_k<=kmax)
    mask *= mask_2
  else:
    get_gauss_filter = lambda R: (lambda K: jnp.exp(-K**2*R**2/2))
    R = 1/kmax
    gauss_filter = get_gauss_filter(R)
    mask = gauss_filter(k)**2
    mask_2 = gauss_filter(K_minus_k)**2
    mask *= mask_2

  #could reuse magnitude of k1 and k2
  return k1, k2, mask


def bs2_coev(b10):
  """
  Coevolution value of the bs2 parameter
  """
  return -2./7.*(b10-1)

def b2_fid(b10):
  return 2*(0.412 - 2.143*b10 + 0.929*b10**2 + 0.008*b10**3)

def bias_t(b10, bs2):
   return b10+7/2*bs2

def bias_g(b10, b2, Fg = 17/21):
   return b10+1/Fg*b2

def bias_s(b10):
   return b10

def get_f(F, P_linear, a, b):
  """
  Returns function for f = 2[a*F(k1+k2, -k1)P(|k1|)+b*F(k1+k2, -k2)P(k2)]
  """
  @jax.jit
  def f(k1, k2):
    K = k1+k2
    K_mag = jnp.linalg.norm(K, axis=-1)
    k1_mag = jnp.linalg.norm(k1, axis=-1)
    k2_mag = jnp.linalg.norm(k2, axis=-1)

    mu_1 = -cosine_angle(K, k1, K_mag, k1_mag)
    mu_2 = -cosine_angle(K, k2, K_mag, k2_mag)

    result = 2*(a*F(q1 = K_mag, q2 = k1_mag, mu = mu_1)*P_linear(k1_mag)+b*F(q1 = K_mag, q2 = k2_mag, mu = mu_2)*P_linear(k2_mag))
    return result
  return f

def get_f_squeezed(f, P_linear):
  """
  Returns function for f(k, K, mu)*P_linear(k). Useful for the squeezed limit of the full f.
  """
  @jax.jit
  def f_squeezed(k1, k2):
    K = k1+k2
    K_mag = jnp.linalg.norm(K, axis=-1)
    k1_mag = jnp.linalg.norm(k1, axis=-1)
    mu_1 = cosine_angle(K, k1, K_mag, k1_mag)
    result = f(q1 = k1_mag, K = K_mag, mu = mu_1)*P_linear(k1_mag) #note, here we explictly use q1, K, mu for the squeezed limit
    return result
  return f_squeezed


def N_per_mode(f_jax_A: callable, f_jax_B: callable, P_AA: callable, P_BB: callable,
               kmin: float = 0.015, kmax: float = 0.15, Ndim = 2, Nsamples_base = 10000, gauss_filter = False):
    """
    Returns calculation of noise normalization per mode
    """

    @jax.jit
    def single_calculation(K):
        def integrand(x):
            # Calculate N(K) = \int_{\vec{k}} [(P(k)+P(K-k))]^2/(PAA(k)*PBB(K-k))
            mu, k = x[:, 0], x[:, 1]

            phi_vol = 2*jnp.pi
            volume = phi_vol*k**2/(2*jnp.pi)**3

            k1, k2, mask = get_k1_k2_mask(K, k, mu*0, mu, kmin, kmax, gauss_filter)
            
            k1_mag, k2_mag = jnp.linalg.norm(k1, axis=-1), jnp.linalg.norm(k2, axis=-1)

            fresultA = f_jax_A(k1, k2)

            fresultB = f_jax_B(k1, k2)

            return fresultA*fresultB/(2*P_AA(k1_mag)*P_BB(k2_mag))*volume*mask

        N = Nsamples_base**Ndim+1

        mc = MonteCarlo()
        result = mc.integrate(integrand, dim = Ndim, N = N,
                              integration_domain = [[-1, 1], [kmin, kmax]],
                              backend = "jax")
        return result

    return single_calculation


def N_per_mode_weighted(w_A: callable, f_jax_B: callable,
               kmin: float = 0.015, kmax: float = 0.15, Ndim = 2, Nsamples_base = 10000, gauss_filter = False):
    """
    Returns calculation of noise normalization per mode
    """

    @jax.jit
    def single_calculation(K):
        def integrand(x):
            mu, k = x[:, 0], x[:, 1]

            phi_vol = 2*jnp.pi
            volume = phi_vol*k**2/(2*jnp.pi)**3

            k1, k2, mask = get_k1_k2_mask(K, k, mu*0, mu, kmin, kmax, gauss_filter)
  
            wresultA = w_A(k1, k2)
            fresultB = f_jax_B(k1, k2)
            return wresultA*fresultB*volume*mask

        N = Nsamples_base**Ndim+1

        mc = MonteCarlo()
        result = mc.integrate(integrand, dim = Ndim, N = N,
                              integration_domain = [[-1, 1], [kmin, kmax]],
                              backend = "jax")
        return result

    return single_calculation


def weight_integral(w_A: callable,
               kmin: float = 0.015, kmax: float = 0.15, Ndim = 2, Nsamples_base = 10000, gauss_filter = False):
    """
    Returns calculation of noise normalization per mode
    """

    @jax.jit
    def single_calculation(K):
        def integrand(x):
            mu, k = x[:, 0], x[:, 1]

            phi_vol = 2*jnp.pi
            volume = phi_vol*k**2/(2*jnp.pi)**3

            k1, k2, mask = get_k1_k2_mask(K, k, mu*0, mu, kmin, kmax, gauss_filter)
  
            wresultA = w_A(k1, k2)
            return wresultA*volume*mask

        N = Nsamples_base**Ndim+1

        mc = MonteCarlo()
        result = mc.integrate(integrand, dim = Ndim, N = N,
                              integration_domain = [[-1, 1], [kmin, kmax]],
                              backend = "jax")
        return result

    return single_calculation

def get_w(f, P_AA, P_BB):
      """
      Unnormalized Weight function w that depends on f and the total power spectra of tracers.

      The weight is given by: f/(2*PAA_tot(k1)*PBB_tot(k2))

      This implementation is optimized for batch processing with Vegas integration.
      It assumes k1 and k2 are 3D vectors, potentially with batch dimensions.
      """

      @jax.jit
      def w(k1, k2):
          # Calculate magnitudes
          k1_mag = jnp.linalg.norm(k1, axis=-1)
          k2_mag = jnp.linalg.norm(k2, axis=-1)

          # Get the f function value using vectors
          f_value = f(k1, k2)

          # Get power spectrum values for the tracers
          P_AA_value = P_AA(k1_mag)
          P_BB_value = P_BB(k2_mag)

          denominator = 2.0 * P_AA_value * P_BB_value
          
          # w_α(k1,k2) = f_α(k1,k2) / (2*PAA_tot(k1)*PBB_tot(k2))
          result = f_value / denominator
          return result

      return w


def get_full_w(f, P_AA, P_BB, P_AB, equal_tracers = False):
      """
      Unnormalized Weight function w that depends on f and the total power spectra of tracers.

      The weight is the optimal QE one for Gaussian noise.

      This implementation is optimized for batch processing with Vegas integration.
      It assumes k1 and k2 are 3D vectors, potentially with batch dimensions.
      """

      @jax.jit
      def w(k1, k2):
          # Calculate magnitudes
          k1_mag = jnp.linalg.norm(k1, axis=-1)
          k2_mag = jnp.linalg.norm(k2, axis=-1)

          # Get the f function value using vectors
          f_value_k1_k2 = f(k1, k2)
          f_value_k2_k1 = f(k2, k1)

          # Get power spectrum values for the tracers
          P_AA_value_k1 = P_AA(k1_mag)
          P_BB_value_k2 = P_BB(k2_mag)
          P_AA_value_k2 = P_AA(k2_mag)
          P_BB_value_k1 = P_BB(k1_mag)
          P_AB_value_k1 = P_AB(k1_mag)
          P_AB_value_k2 = P_AB(k2_mag)

          
          denominator = (P_AA_value_k1 * P_BB_value_k1 * P_AA_value_k2 * P_BB_value_k2 - P_AB_value_k1**2 * P_AB_value_k2**2) #2.0, I remove factor of 2 to be consistent with my usual norm.
          numerator = f_value_k1_k2 * P_AA_value_k2 * P_BB_value_k1 - f_value_k2_k1 * P_AB_value_k1 * P_AB_value_k2

          #denominator = (P_AA_value_k1 * P_BB_value_k2 + P_AB_value_k1 * P_AB_value_k2) #2.0, I remove factor of 2 to be consistent with my usual norm.
          #numerator = f_value_k1_k2

          result = numerator / denominator
          return result

      return w



def variance_per_mode(weight_AB_alpha: callable, weight_XY_beta: callable,
                      P_AX: callable, P_BY: callable, P_AY: callable, P_BX: callable, kmin: float = 0.051, kmax: float = 0.15,
                      Nsamples_base = 1000, Ndim = 3, gauss_filter = False):
    """
    This calculates the noise between two estimator h^{AB}_alpha, h^{XY}_beta built with A, B, X, Y tracers, for alpha, beta estimation.


    """

    def single_calculation(K):
        @jax.jit
        def integrand(x):
            phi, mu, k = x[:, 0], x[:, 1], x[:, 2]

            volume = k**2/(2*jnp.pi)**3

            k1, k2, mask = get_k1_k2_mask(K, k, phi, mu, kmin, kmax, gauss_filter = gauss_filter)
            k1_mag, k2_mag = jnp.linalg.norm(k1, axis=-1), jnp.linalg.norm(k2, axis=-1)

            w_result_AB = weight_AB_alpha(k1, k2)
            w_result_XY_1 = weight_XY_beta(k1, k2)
            w_result_XY_2 = weight_XY_beta(k2, k1)
            P_AX_value = P_AX(k1_mag)
            P_BY_value = P_BY(k2_mag)
            P_AY_value = P_AY(k1_mag)
            P_BX_value = P_BX(k2_mag)

            result = w_result_XY_1*P_AX_value*P_BY_value+w_result_XY_2*P_AY_value*P_BX_value
            result *= w_result_AB

            return result*volume*mask

        N = Nsamples_base**Ndim+1


        integration_domain = [[0, 2*jnp.pi], [-1, 1], [kmin, kmax]]

        mc = MonteCarlo()
        result = mc.integrate(integrand, dim = Ndim, N = N,
                              integration_domain = integration_domain,
                              backend = "jax")
        return result

    return single_calculation


def variance_per_mode_fast(weight_AB_alpha: callable, weight_XY_beta: callable,
                      P_AX: callable, P_BY: callable, P_AY: callable, P_BX: callable, kmin: float = 0.051, kmax: float = 0.15,
                      Nsamples_base = 1000, Ndim = 2, gauss_filter = False):
    """
    This calculates the noise between two estimator h^{AB}_alpha, h^{XY}_beta built with A, B, X, Y tracers, for alpha, beta estimation.
    Assume azimuthal symmetry.

    """

    def single_calculation(K):
        @jax.jit
        def integrand(x):
            mu, k = x[:, 0], x[:, 1]

            volume = 2*jnp.pi*k**2/(2*jnp.pi)**3 #2 is for integration over phi

            k1, k2, mask = get_k1_k2_mask(K, k, mu*0., mu, kmin, kmax, gauss_filter = gauss_filter)
            k1_mag, k2_mag = jnp.linalg.norm(k1, axis=-1), jnp.linalg.norm(k2, axis=-1)

            w_result_AB = weight_AB_alpha(k1, k2)
            w_result_XY_1 = weight_XY_beta(k1, k2)
            w_result_XY_2 = weight_XY_beta(k2, k1)
  
            P_AX_value = P_AX(k1_mag)
            P_BY_value = P_BY(k2_mag)
            P_AY_value = P_AY(k1_mag)
            P_BX_value = P_BX(k2_mag)

            result = w_result_XY_1*P_AX_value*P_BY_value+w_result_XY_2*P_AY_value*P_BX_value
            result *= w_result_AB

            return result*volume*mask

        N = Nsamples_base**Ndim+1


        integration_domain = [[-1, 1], [kmin, kmax]]

        mc = MonteCarlo()
        result = mc.integrate(integrand, dim = Ndim, N = N,
                              integration_domain = integration_domain,
                              backend = "jax")
        return result

    return single_calculation



def variance_per_mode_integrand(K: float, weight_AB_alpha: callable, weight_XY_beta: callable,
                      P_AX: callable, P_BY: callable, P_AY: callable, P_BX: callable, kmin: float = 0.051, kmax: float = 0.15,
                      Nsamples_base = 1000, Ndim = 3):
    """
    This calculates the noise between two estimator h^{AB}_alpha, h^{XY}_beta built with A, B, X, Y tracers, for alpha, beta estimation.


    """


    @jax.jit
    def integrand(x):
        phi, mu, k = x[:, 0], x[:, 1], x[:, 2]

        volume = k**2/(2*jnp.pi)**3

        k1, k2, mask = get_k1_k2_mask(K, k, phi, mu, kmin, kmax) #k1, K-k1
        k1_mag, k2_mag = jnp.linalg.norm(k1, axis=-1), jnp.linalg.norm(k2, axis=-1)

        w_result_AB = weight_AB_alpha(k1, k2)
        w_result_XY_1 = weight_XY_beta(k1, k2)
        w_result_XY_2 = weight_XY_beta(k2, k1)
        P_AX_value = P_AX(k1_mag)
        P_BY_value = P_BY(k2_mag)
        P_AY_value = P_BY(k1_mag)
        P_BX_value = P_BX(k2_mag)

        result = w_result_XY_1*P_AX_value*P_BY_value+w_result_XY_2*P_AY_value*P_BX_value
        result *= w_result_AB

        return result*volume*mask

    return integrand


"""def set_up_fixed_grid(N, dim, integration_domain):
  integration_domain = jnp.array(integration_domain)
  integrator = MonteCarlo()
  grid_points = integrator.calculate_sample_points(N, integration_domain)
  return integrator, grid_points

N = int(1e5)
dim = 3
kmin, kmax = 0.05, 0.15
integration_domain = jnp.array([[0, 2*jnp.pi], [-1, 1], [kmin, kmax]])

integrator, grid_points = set_up_fixed_grid(N, dim, integration_domain)

for K in Ks:
  integrand = variance_per_mode_integrand(K, w_A, w_B, P_AX = P_AA, P_BY = P_BB, P_AY = P_AB, P_BX = P_AB, kmin = kmin, kmax = kmax, Nsamples_base = 200)
  function_values, _ = integrator.evaluate_integrand(integrand, grid_points)
  integral1 = integrator.calculate_result(function_values, integration_domain)
"""



def cross_shot_mixed_AAB(weight_AB_alpha: callable, nbar_A: float, P_AB: callable,
               kmin: float = 0.015, kmax: float = 0.15, Ndim = 2, Nsamples_base = 10000, activate_k2 = True):
    """
    Returns calculation of mixed shot noise
    """

    @jax.jit
    def single_calculation(K):
        def integrand(x):
            mu, k = x[:, 0], x[:, 1]

            phi_vol = 2*jnp.pi
            volume = phi_vol*k**2/(2*jnp.pi)**3

            k1, k2, mask = get_k1_k2_mask(K, k, mu*0, mu, kmin, kmax)
            k2_mag = jnp.linalg.norm(k2, axis=-1)
            k1_mag = k
            #K_mag = K*jnp.ones_like(k)

            w_result_AB = weight_AB_alpha(k1, k2)
            shotA = 1/nbar_A
            PS = P_AB(k2_mag)*shotA if activate_k2 else P_AB(k1_mag)*shotA

            return w_result_AB*PS*volume*mask

        N = Nsamples_base**Ndim+1

        mc = MonteCarlo()
        result = mc.integrate(integrand, dim = Ndim, N = N,
                              integration_domain = [[-1, 1], [kmin, kmax]],
                              backend = "jax")
        return result

    return single_calculation


def shot_bispectrum(weight_AB_alpha: callable, nbar_A: float, P_AB: callable,
               kmin: float = 0.015, kmax: float = 0.15, Ndim = 2, Nsamples_base = 10000):
    """
    Returns calculation of mixed shot noise. This is for a single field. Simplifies calculations.
    """

    @jax.jit
    def single_calculation(K):
        def integrand(x):
            mu, k = x[:, 0], x[:, 1]

            phi_vol = 2*jnp.pi
            volume = phi_vol*k**2/(2*jnp.pi)**3

            k1, k2, mask = get_k1_k2_mask(K, k, mu*0, mu, kmin, kmax)
            k2_mag = jnp.linalg.norm(k2, axis=-1)

            w_result_AB = weight_AB_alpha(k1, k2)

            shotA = 1/nbar_A

            PS = P_AB(k2_mag)+P_AB(k)+P_AB(K)+shotA
            PS *= shotA

            return w_result_AB*PS*volume*mask

        N = Nsamples_base**Ndim+1

        mc = MonteCarlo()
        result = mc.integrate(integrand, dim = Ndim, N = N,
                              integration_domain = [[-1, 1], [kmin, kmax]],
                              backend = "jax")
        return result

    return single_calculation


def shot_bispectrum_alternative(weight_AB_alpha: callable, nbar_A: float, P_AB: callable,
               kmin: float = 0.015, kmax: float = 0.15, Ndim = 2, Nsamples_base = 10000, Norm_K: callable = None):
    """
    Returns calculation of mixed shot noise. This is for a single field. Simplifies calculations.
    """

    @jax.jit
    def single_calculation(K):
        def integrand(x):
            mu, k = x[:, 0], x[:, 1]

            phi_vol = 2*jnp.pi
            volume = phi_vol*k**2/(2*jnp.pi)**3

            k1, k2, mask = get_k1_k2_mask(K, k, mu*0, mu, kmin, kmax)

            w_result_AB = weight_AB_alpha(k1, k2)

            shotA = 1/nbar_A

            PS = 2*P_AB(k)
            PS *= shotA

            return w_result_AB*PS*volume*mask

        N = Nsamples_base**Ndim+1

        mc = MonteCarlo()
        result = mc.integrate(integrand, dim = Ndim, N = N,
                              integration_domain = [[-1, 1], [kmin, kmax]],
                              backend = "jax")
        
        result *= Norm_K(K) #need to normalize, as w here is not normalized
        result += (1/nbar_A*(1/nbar_A+P_AB(K)))
        result /= Norm_K(K)

        return result

    return single_calculation





def variance_per_mode_integrand(K: float, weight_AB_alpha: callable, weight_XY_beta: callable,
                      P_AX: callable, P_BY: callable, P_AY: callable, P_BX: callable, kmin: float = 0.051, kmax: float = 0.15,
                      Nsamples_base = 1000, Ndim = 3):
    """
    This calculates the noise between two estimator h^{AB}_alpha, h^{XY}_beta built with A, B, X, Y tracers, for alpha, beta estimation.
    """


    @jax.jit
    def integrand(x):
        phi, mu, k = x[:, 0], x[:, 1], x[:, 2]

        volume = k**2/(2*jnp.pi)**3

        k1, k2, mask = get_k1_k2_mask(K, k, phi, mu, kmin, kmax)
        k1_mag, k2_mag = jnp.linalg.norm(k1, axis=-1), jnp.linalg.norm(k2, axis=-1)

        w_result_AB = weight_AB_alpha(k1, k2)
        w_result_XY_1 = weight_XY_beta(k1, k2)
        w_result_XY_2 = weight_XY_beta(k2, k1)
        P_AX_value = P_AX(k1_mag)
        P_BY_value = P_BY(k2_mag)
        P_AY_value = P_BY(k1_mag)
        P_BX_value = P_BX(k2_mag)

        result = w_result_XY_1*P_AX_value*P_BY_value+w_result_XY_2*P_AY_value*P_BX_value
        result *= w_result_AB

        return result*volume*mask

    return integrand


def get_bispectrum_XYZ(P_signal_X, P_signal_Y, P_signal_Z, Fkernels, Fbiases_X, Fbiases_Y, Fbiases_Z):
  # Pre-convert Fbiases to JAX array outside the jitted function
  Fbiases_array_X = jnp.array(Fbiases_X)
  Fbiases_array_Y = jnp.array(Fbiases_Y)
  Fbiases_array_Z = jnp.array(Fbiases_Z)
  
  #@jax.jit
  def bispectrum_XYZ(k1, k2, k3, k1_mag, k2_mag, k3_mag):
      P_X_1 = P_signal_X(k1_mag)
      P_Y_2 = P_signal_Y(k2_mag)
      P_Z_3 = P_signal_Z(k3_mag)
          
      #XY = 2*P_X_1*P_Y_2*jnp.sum([F(k1, k2, k1_mag, k2_mag)*c for F, c in zip(Fkernels, Fbiases)], axis = 0)
      #YZ = 2*P_Y_2*P_Z_3*jnp.sum([F(k2, k3, k2_mag, k3_mag)*c for F, c in zip(Fkernels, Fbiases)], axis = 0)
      #XZ = 2*P_X_1*P_Z_3*jnp.sum([F(k3, k1, k3_mag, k1_mag)*c for F, c in zip(Fkernels, Fbiases)], axis = 0)
      
      # XY term: k1, k2
      XY_sum_Z = (Fkernels[0](k1, k2, k1_mag, k2_mag) * Fbiases_array_Z[0] +
                Fkernels[1](k1, k2, k1_mag, k2_mag) * Fbiases_array_Z[1] +
                Fkernels[2](k1, k2, k1_mag, k2_mag) * Fbiases_array_Z[2])
      XY = 2*P_X_1*P_Y_2*XY_sum_Z
      
      # YZ term: k2, k3  
      YZ_sum_X = (Fkernels[0](k2, k3, k2_mag, k3_mag) * Fbiases_array_X[0] +
                Fkernels[1](k2, k3, k2_mag, k3_mag) * Fbiases_array_X[1] +
                Fkernels[2](k2, k3, k2_mag, k3_mag) * Fbiases_array_X[2])
      YZ = 2*P_Y_2*P_Z_3*YZ_sum_X
      
      # XZ term: k3, k1
      XZ_sum_Y = (Fkernels[0](k3, k1, k3_mag, k1_mag) * Fbiases_array_Y[0] +
                Fkernels[1](k3, k1, k3_mag, k1_mag) * Fbiases_array_Y[1] +
                Fkernels[2](k3, k1, k3_mag, k1_mag) * Fbiases_array_Y[2])
      XZ = 2*P_X_1*P_Z_3*XZ_sum_Y

      somma = XY+YZ+XZ
      return somma
  return bispectrum_XYZ


def shot_trispectrum(weight_AB_alpha: callable, weight_XY_beta: callable,
                      P_cont: callable, bispectrum_cont: callable, nbar: float, kmin: float = 0.051, kmax: float = 0.15,
                      Nsamples_base = 1000, Ndim = 6, torchquad = True):
    """
    Trispectrum shot-noise. ASSUMES A=B, SO THAT YOU CAN GET A MORE PESSIMISTIC VIEW. THIS AVOIDS CALCULATING TOO MANY TERMS IN GENERAL CASE.
    """

    def single_calculation(K):
        @jax.jit
        def funzione(k, k_p, phi, mu, phi_p, mu_p):

            volume = k**2/(2*jnp.pi)**3
            volume_p = k_p**2/(2*jnp.pi)**3

            k1, k2, mask = get_k1_k2_mask(K, k, phi, mu, kmin, kmax)
            Kvec = k1+k2
            
            k1_mag, k2_mag = jnp.linalg.norm(k1, axis=-1), jnp.linalg.norm(k2, axis=-1)
            ones = jnp.ones_like(k1_mag)
            K_mag = ones*K

            #get_k1_k2_mask gives you q, and K-q, vectorial form
            #if you want to get -K-q, you need to flip the sign of the direction, so K_sign = -1
            k1_p, k2_p, mask_p = get_k1_k2_mask(K, k_p, phi_p, mu_p, kmin, kmax, K_sign = -1)
            k1_p_mag, k2_p_mag = jnp.linalg.norm(k1_p, axis=-1), jnp.linalg.norm(k2_p, axis=-1)

            w_result_AB = weight_AB_alpha(k1, k2)
            w_result_XY_p = weight_XY_beta(k1_p, k2_p)
            
            shot = 1/nbar
            shot3 = shot**3
            shot2 = shot**2

            k1_plus_k2_p = k1+k2_p
            k1_plus_k1_p = k1+k1_p
            minus_k1_minus_k1_p = -k1-k1_p

            k1_plus_k1_p_mag = jnp.linalg.norm(k1_plus_k1_p, axis = -1)
            k1_plus_k2_p_mag = jnp.linalg.norm(k1_plus_k2_p, axis = -1)
            minus_k1_minus_k1_p_mag = jnp.linalg.norm(minus_k1_minus_k1_p, axis = -1)

            mask_all = 1.
            #mask_all *= maskf(k1_plus_k1_p_mag, kmin, kmax)
            #mask_all *= maskf(k1_plus_k2_p_mag, kmin, kmax)
            #mask_all *= maskf(minus_k1_minus_k1_p_mag, kmin, kmax)
            
            pgcont_1 = P_cont(k1_mag)
            pgcont_2 = P_cont(k2_mag)
            pgcont_p_1 = P_cont(k1_p_mag)
            pgcont_p_2 = P_cont(k2_p_mag)

            somma = 0.

            somma += shot3
            
            temp = pgcont_1+pgcont_2+pgcont_p_1+pgcont_p_2
            temp *= shot2
            somma += temp
            
            pgcont_long = P_cont(K)*ones
            pgcont_q_minus_K_minus_qp = P_cont(k1_plus_k2_p_mag)
            pgcont_q_plus_qp = P_cont(k1_plus_k1_p_mag)

            temp = pgcont_q_minus_K_minus_qp+pgcont_q_plus_qp+pgcont_long
            temp *= shot2
            somma += temp

            temp = bispectrum_cont(Kvec, k1_p, k2_p, K_mag, k1_p_mag, k2_p_mag)
            temp += 4*bispectrum_cont(minus_k1_minus_k1_p, k1, k1_p, minus_k1_minus_k1_p_mag, k1_mag, k1_p_mag)
            temp += bispectrum_cont(-Kvec, k1, k2, K_mag, k1_mag, k2_mag)
            temp *= shot
            somma += temp

            somma *= w_result_AB*w_result_XY_p

            return somma*volume*volume_p*mask*mask_p*mask_all
        

        @jax.jit
        def integrand_(x):
            phi, mu, k = x[:, 0], x[:, 1], x[:, 2]
            phi_p, mu_p, k_p = x[:, 3], x[:, 4], x[:, 5]

            somma_1 = funzione(k, k_p, phi, mu, phi_p, mu_p)
            #somma_2 = funzione(k_p, k, phi, -mu, phi_p, mu_p)
            #media = (somma_1+somma_2)/2
            media = somma_1
            return media


        if torchquad:
          N = Nsamples_base**Ndim+1

          integration_domain = [[0, 2*jnp.pi], [-1, 1], [kmin, kmax], [0, 2*jnp.pi], [-1, 1], [kmin, kmax]]

          mc = MonteCarlo()
          result = mc.integrate(integrand_, dim = Ndim, N = N,
                                integration_domain = integration_domain,
                                backend = "jax")
        else:
           
          @vegas.lbatchintegrand
          def integrand(x):
             return integrand_(jnp.array(x))
          
          #integ = vegas.Integrator([[0, 2*jnp.pi], [-1, 1], [kmin, kmax], [0, 2*jnp.pi], [-1, 1], [kmin, kmax]], gpu_pad = True)
          #integ(integrand, nitn=20, neval=1e3)
          #result = integ(integrand, nitn=30, neval=1e5, adapt=True).mean
          result = integrand

        return result

    return single_calculation




def shot_trispectrum_mixed(K: float, weight_AB_alpha: callable, P_AB: callable, bispectrum_ABB: callable, bispectrum_BAA: callable, 
                           nbar_A: float, nbar_B: float, kmin: float = 0.051, kmax: float = 0.15,
                           Nsamples_base = 1000, Ndim = 6):
    """
    Trispectrum shot-noise. For mixed QE auto-spectrum AB-AB
    """

    def single_calculation(K):
        @jax.jit
        def funzione(k, k_p, phi, mu, phi_p, mu_p):

            volume = k**2/(2*jnp.pi)**3
            volume_p = k_p**2/(2*jnp.pi)**3

            k1, k2, mask = get_k1_k2_mask(K, k, phi, mu, kmin, kmax)
            Kvec = k1+k2
            
            k1_mag, k2_mag = jnp.linalg.norm(k1, axis=-1), jnp.linalg.norm(k2, axis=-1)
            ones = jnp.ones_like(k1_mag)
            K_mag = ones*K

            #get_k1_k2_mask gives you q, and K-q, vectorial form
            #if you want to get -K-q, you need to flip the sign of the direction, so K_sign = -1
            #k3, k4, mask_p = get_k1_k2_mask(K, k_p, phi_p, mu_p, kmin, kmax, K_sign = -1)
            #k3_mag, k4_mag = jnp.linalg.norm(k3, axis=-1), jnp.linalg.norm(k4, axis=-1)

            k3 = spherical_to_cartesian(phi_p, mu_p, k_p)
            k4 = -(k1+k2+k3)
            k3_mag = jnp.linalg.norm(k3, axis = -1)
            k4_mag = jnp.linalg.norm(k4, axis = -1)
            mask_p = maskf(k4_mag, kmin, kmax)*maskf(k3_mag, kmin, kmax)

            k1_plus_k3 = k1+k3
            k1_plus_k3_mag = jnp.linalg.norm(k1_plus_k3, axis = -1)
            k2_plus_k4 = k2+k4
            k2_plus_k4_mag = jnp.linalg.norm(k2_plus_k4, axis = -1)

            power_AB_value = P_AB(k1_plus_k3_mag) #cross-correlation between A and B
            power_AB_value *= 1/nbar_A*1/nbar_B
            
            somma = 0.
            somma += power_AB_value

            bispectrum_ABB_value = bispectrum_ABB(k1_plus_k3, k2, k4, k1_plus_k3_mag, k2_mag, k4_mag)
            bispectrum_BAA_value = bispectrum_BAA(k2_plus_k4, k1, k3, k2_plus_k4_mag, k1_mag, k3_mag)

            somma += (bispectrum_ABB_value*1/nbar_A)
            somma += (bispectrum_BAA_value*1/nbar_B)

            w_result_AB_12 = weight_AB_alpha(k1, k2)
            w_result_AB_34 = weight_AB_alpha(k3, k4)

            somma *= w_result_AB_12*w_result_AB_34

            somma *= volume*volume_p*mask*mask_p
            return somma

        @jax.jit
        def integrand_(x):
            phi, mu, k = x[:, 0], x[:, 1], x[:, 2]
            phi_p, mu_p, k_p = x[:, 3], x[:, 4], x[:, 5]
            return funzione(k, k_p, phi, mu, phi_p, mu_p)
        
        N = Nsamples_base**Ndim+1

        integration_domain = [[0, 2*jnp.pi], [-1, 1], [kmin, kmax], [0, 2*jnp.pi], [-1, 1], [kmin, kmax]]

        mc = MonteCarlo()
        result = mc.integrate(integrand_, dim = Ndim, N = N,
                              integration_domain = integration_domain,
                              backend = "jax")
        return result
            
    return single_calculation



def shot_trispectrum_general(K: float, weight_AB_alpha: callable, weight_XY_beta: callable,
                      P_AX: callable, P_BY: callable, P_AY: callable, P_BX: callable, nbar_A: float, nbar_B: float, kmin: float = 0.051, kmax: float = 0.15,
                      Nsamples_base = 1000, Ndim = 3):
    """
    Trispectrum shot-noise. ASSUMES A=B, SO THAT YOU CAN GET A MORE PESSIMISTIC VIEW. THIS AVOIDS CALCULATING TOO MANY TERMS IN GENERAL CASE.
    """

    def single_calculation(K):
        @jax.jit
        def funzione(k, k_p, phi, mu, phi_p, mu_p):

            volume = k**2/(2*jnp.pi)**3
            volume_p = k_p**2/(2*jnp.pi)**3

            k1, k2, mask = get_k1_k2_mask(K, k, phi, mu, kmin, kmax)
            Kvec = k1+k2
            
            k1_mag, k2_mag = jnp.linalg.norm(k1, axis=-1), jnp.linalg.norm(k2, axis=-1)
            ones = jnp.ones_like(k1_mag)
            K_mag = ones*K

            #get_k1_k2_mask gives you q, and K-q, vectorial form
            #if you want to get -K-q, you need to flip the sign of the direction, so K_sign = -1
            k3, k4, mask_p = get_k1_k2_mask(K, k_p, phi_p, mu_p, kmin, kmax, K_sign = -1)
            k3_mag, k4_mag = jnp.linalg.norm(k3, axis=-1), jnp.linalg.norm(k4, axis=-1)

            w_result_AB_12 = weight_AB_alpha(k1, k2)
            w_result_XY_34 = weight_XY_beta(k3, k4)
            
            shot = 1/nbar_A
            shot3 = shot**3
            shot2 = shot**2

            k1_plus_k2_mag = K_mag
            k1_plus_k3_mag = jnp.linalg.norm(k1+k3, axis = -1)
            k2_plus_k3_mag = jnp.linalg.norm(k2+k3, axis = -1) #this is same as (-k1_plus_k4)_mag

            power_spectra_sum = P_AX(k1_mag)+P_AX(k2_mag)+P_AX(k3_mag)+P_AX(k4_mag)+P_AX(k1_plus_k2_mag)+P_AX(k1_plus_k3_mag)+P_AX(k2_plus_k3_mag)

            somma = shot3
            somma += power_spectra_sum*shot2
            
            return somma*volume*volume_p*mask*mask_p
        

        @jax.jit
        def integrand_(x):
            phi, mu, k = x[:, 0], x[:, 1], x[:, 2]
            phi_p, mu_p, k_p = x[:, 3], x[:, 4], x[:, 5]

            somma_1 = funzione(k, k_p, phi, mu, phi_p, mu_p)
            somma_2 = funzione(k_p, k, phi, -mu, phi_p, mu_p)
            
            media = (somma_1+somma_2)/2
            #media = somma
            return media

    return single_calculation