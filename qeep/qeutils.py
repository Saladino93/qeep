import jax
import jax.numpy as jnp
from torchquad import MonteCarlo, Boole, VEGAS
from torchquad import set_up_backend  # Necessary to enable GPU support
set_up_backend("jax", data_type="float32")


def Fs(k1, k2):
  return 0.5*(1/jnp.linalg.norm(k1, axis = 0)**2+1/jnp.linalg.norm(k2, axis = 0)**2)*dot(k1, k2)

def Ft(k1, k2):
  return 2/7*(dot(k1, k2)**2/jnp.linalg.norm(k1, axis = 0)/jnp.linalg.norm(k2, axis = 0)-1/3)

def Fg(k1, k2):
  return 17/21

def symmetric_part(fAB, fBA):
  return 0.5*(fAB+fBA)

def antisymmetric_part(fAB, fBA):
  return 0.5*(fAB-fBA)

def get_interpolated(k, p):

  x_data_jax = jnp.array(k)
  y_data_jax = jnp.array(p)

  @jax.jit
  def interpolate_function(x_new):
      # For 1D linear interpolation
      return jnp.interp(x_new, x_data_jax, y_data_jax)

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


def get_k1_k2_mask(K, k, phi, mu, kmin, kmax, gauss_filter = False):

  K_minus_k = jnp.sqrt(K**2+k**2-2*K*k*mu)

  K = spherical_to_cartesian(phi*0, mu*0+1, K) #[0, 0, K], aligned with z-axis
  k1 = spherical_to_cartesian(phi, mu, k)
  k2 = K-k1

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

  return k1, k2, mask


def bs2_coev(b10):
  """
  Coevolution value of the bs2 parameter
  """
  return -2./7.*(b10-1)

def get_f(F, P_linear, a, b):
  """
  Returns function for f = 2[F(k1+k2, -k1)P(|k1|)+F(k1+k2, -k2)P(k2)]
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
            # Calculate N(K) = \int_{\vec{k}} [(P(k)+P(K-k))]^2/(PAA(k)*PBB(K-k))
            mu, k = x[:, 0], x[:, 1]

            phi_vol = 2*jnp.pi
            volume = phi_vol*k**2/(2*jnp.pi)**3

            k1, k2, mask = get_k1_k2_mask(K, k, mu*0, mu, kmin, kmax, gauss_filter)
            
            #k1_mag, k2_mag = jnp.linalg.norm(k1, axis=-1), jnp.linalg.norm(k2, axis=-1)

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
            P_AY_value = P_BY(k1_mag)
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
               kmin: float = 0.015, kmax: float = 0.15, Ndim = 2, Nsamples_base = 10000):
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

            w_result_AB = weight_AB_alpha(k1, k2)
            shotA = 1/nbar_A
            PS = P_AB(k2_mag)*shotA

            return w_result_AB*PS*volume*mask

        N = Nsamples_base**Ndim+1

        mc = MonteCarlo()
        result = mc.integrate(integrand, dim = Ndim, N = N,
                              integration_domain = [[-1, 1], [kmin, kmax]],
                              backend = "jax")
        return result

    return single_calculation


def cross_shot(weight_AB_alpha: callable, nbar_A: float, P_AB: callable,
               kmin: float = 0.015, kmax: float = 0.15, Ndim = 2, Nsamples_base = 10000):
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

            w_result_AB = weight_AB_alpha(k1, k2)
            shotA = 1/nbar_A
            PS = P_AB(k2_mag)*shotA

            return w_result_AB*PS*volume*mask

        N = Nsamples_base**Ndim+1

        mc = MonteCarlo()
        result = mc.integrate(integrand, dim = Ndim, N = N,
                              integration_domain = [[-1, 1], [kmin, kmax]],
                              backend = "jax")
        return result

    return single_calculation



def bispectrum_XYZ(k1, k2, k3, P_signal_X, P_signal_Y, P_signal_Z, Fkernels, Fbiases):
  P_X_1 = P_signal_X(k1)
  P_Y_2 = P_signal_Y(k2)
  P_Z_3 = P_signal_Z(k3)

  XY = 2*P_X_1*P_Y_2*jnp.sum([F(k1, k2)*c for F, c in zip(Fkernels, Fbiases)], axis = 0)
  YZ = 2*P_Y_2*P_Z_3*jnp.sum([F(k2, k3)*c for F, c in zip(Fkernels, Fbiases)], axis = 0)
  XZ = 2*P_X_1*P_Z_3*jnp.sum([F(k1, k3)*c for F, c in zip(Fkernels, Fbiases)], axis = 0)

  somma = XY+YZ+XZ
  return somma




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