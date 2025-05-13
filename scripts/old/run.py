from torchquad import Simpson, set_up_backend
import cupy as cp
from torchquad import MonteCarlo
from scipy import integrate
import numpy as np

import jax

print(jax.devices())  # Should show GPU devices if available


import sympy as sp
import sympy2jax
import pandas as pd

import time


nonlinear_power = np.loadtxt("nonlinear_power_quijote.txt").T

from scipy import interpolate
knl, pnl = nonlinear_power
pnlf = interpolate.interp1d(knl, pnl, kind='linear', fill_value=1e10, bounds_error=False)

kl, pl = np.loadtxt("linear_power_quijote.txt").T
plf = interpolate.interp1d(kl, pl, kind='linear', fill_value=1e10, bounds_error=False)

ks = np.linspace(0.01, 1.0, 10000)


import jax
import jax.numpy as jnp
from jax.scipy import ndimage

# Convert your data to JAX arrays
x_data_jax = jnp.array(knl)
y_data_jax = jnp.array(pnl)

@jax.jit
def interpolate_function(x_new):
    # For 1D linear interpolation
    return jnp.interp(x_new, x_data_jax, y_data_jax)


x_data_lin_jax = jnp.array(kl)
y_data_lin_jax = jnp.array(pl)

@jax.jit
def interpolate_function_lin(x_new):
    # For 1D linear interpolation
    return jnp.interp(x_new, x_data_lin_jax, y_data_lin_jax)


q1, q2, mu = sp.symbols('q1 q2 mu')


def get_total_P(b10, nbar):
  shot = 1/nbar
  return lambda k: b10**2*interpolate_function(k)+shot

b10_A = 1 #1.6
nbar_A = 3e5 #3.3e-4
b10_B = 1 #1.6
nbar_B = 3e5 #3.3e-4

P_AA = get_total_P(b10_A, nbar_A)
P_BB = get_total_P(b10_B, nbar_B)
P_AB = lambda k: b10_A*b10_B*interpolate_function(k)
P_linear = lambda k: interpolate_function_lin(k)

results = []


deltac = 1.42
# a1 = 1 in spherical collapse
a1 = 1.
# a2 = -17/21 in spherical collapse
a2 = -0.8095238095

Fg_factor = 17/21
Ft_one_third = False
Fg_factor = 5/7

cgs = {}
cgs["g"] = lambda b1, b2, bs2: b1+Fg_factor**-1*b2
cgs["s"] = lambda b1, b2, bs2: b1
cgs["t"] = lambda b1, b2, bs2: b1+7/2*bs2
cgs["x"] = lambda b1, b2, bs2: b1
cgs["n"] = lambda b1, b2, bs2: b1
cgs["phiphi"] = lambda b1, fnl: b1*fnl
cgs["c01"] = lambda b1, fnl: fnl*2*deltac*(b1-1.)
cgs["c11"] = lambda b1, b2, fnl: fnl*(2./a1)*(deltac*(b2-2*(a1+a2)*(b1-1.))-a1**2.*(b1-1.))+2.*fnl*deltac*(b1-1.)
cgs["c02"] = lambda b1, b2, fnl: fnl**2*4*deltac*((deltac/a1**2.)*(b2-2.*(a1+a2)*(b1-1.))-2.*(b1-1.))


Cg = {}
Cg["g"] = lambda b1A, b2A, b1B, bthetaA, bdeltathetaA, fX, bmrA, brA: (Fg_factor*b1A+1/2*b2A)*b1B
Cg["s"] = lambda b1A, b2A, bs2A, b1B, b2B, bs2B: b1A*b1B
Cg["t"] = lambda b1A, b2A, bs2A, b1B, b2B, bs2B: (2/7*b1A+1/2*bs2A)*b1B
Cg["n"] = lambda b1A, b2A, bs2A, b1B, b2B, bs2B: 1*(b1A==b1B)
Cg["x"] = lambda b1A, epsilon, brB, bDB, bthetaB, H: b1A*epsilon*(17/6*brB-7/3*H*bthetaB-5/3*H*bDB)
Cg["phiphi"] = lambda b1A, b1B, fnl: b1A*fnl


b20_A = 0 #-0.3
b20_B = 0 #-0.3

bthetaB = 0.5
bthetaA = 0.3

brA = 1
brB = 2

epsilon = 1e-3
epsilon_2 = 10*epsilon

bs2f = lambda b10: -2./7.*(b10-1)

bs2_A = 1 #bs2f(b10_A)
bs2_B = 1 #bs2f(b10_B)

fX = 1
bmrA, bdeltathetaA = 0, 0
bmrB, bdeltathetaB = 0, 0

estimator_configs = {
        'g': {
            'F': Fg_factor*q1/q1,
            'ca': Cg["g"](b1A = b10_A, b2A = b20_A, b1B = b10_B, bthetaA = 0, bdeltathetaA = 0, fX = 0, bmrA = 0, brA = 0), #AB
            'cb': Cg["g"](b1A = b10_B, b2A = b20_B, b1B = b10_A, bthetaA = 0, bdeltathetaA = 0, fX = 0, bmrA = 0, brA = 0) #BA
        },
        's': {
            'F': 0.5*(q2/q1+q1/q2)*mu, #NOTE: MU IS ALWAYS THE ANGLE BETWEEN Q1 AND Q2, BUT OFTEN IN THE CODE Q1 IS JUST K!
            'ca': Cg["s"](b1A = b10_A, b2A = 0, bs2A = 0, b1B = b10_B, b2B = 0, bs2B = 0),#b10,
            'cb': Cg["s"](b1A = b10_B, b2A = 0, bs2A = 0, b1B = b10_A, b2B = 0, bs2B = 0)#b10
        },
        't': {
            'F': (2./7.)*mu**2.-1./3.*(Ft_one_third),
            'ca': Cg["s"](b1A = b10_A, b2A = 0, bs2A = bs2_A, b1B = b10_B, b2B = 0, bs2B = 0),
            'cb': Cg["s"](b1A = b10_B, b2A = 0, bs2A = bs2_B, b1B = b10_A, b2B = 0, bs2B = 0)
        },
        'x': {
            #'F': 1/(q1+q2)*mu,
            'F': 0.5*(q2/q1+q1/q2)*mu,
            'ca': Cg["x"](b1A = b10_A, epsilon = epsilon, bthetaB = 0, brB = brB, bDB = 0, H = 0),
            'cb': Cg["x"](b1A = b10_B, epsilon = epsilon, bthetaB = 0, brB = brA, bDB = 0, H = 0)
        },
        'x2': {
            #'F': 1/(q1+q2)*mu,
            'F': 0.5*(q2/q1+q1/q2)*mu,
            'ca': Cg["x"](b1A = b10_A, epsilon = epsilon_2, bthetaB = 0, brB = brB, bDB = 0, H = 0),
            'cb': Cg["x"](b1A = b10_B, epsilon = epsilon_2, bthetaB = 0, brB = brA, bDB = 0, H = 0)
        },
        'n': {
            #'F': 1/(q1+q2)*mu,
            'F': 0.5*mu*q2/q1,
            'ca': Cg["n"](1, 0, 0, 1, 0, 0),
            'cb': Cg["n"](1, 0, 0, 0, 0, 0)
        }
    }

estimator_lam = {key: sp.lambdify([q1, q2, mu], estimator_configs[key]['F'], 'numpy') for key in estimator_configs}
estimator_lam_jax = {key: sympy2jax.SymbolicModule(estimator_configs[key]['F']) for key in estimator_configs}


def get_f(F, a, b):
  @jax.jit
  def f(k, K, mu):
    K_minus_k = jnp.sqrt(K**2+k**2-2*K*k*mu)
    mu1 = -mu
    mu2 = -(K**2 - k*K*mu) / (K*K_minus_k)
    result = 2*(a*F(q1 = K, q2 = k, mu = mu1)*P_linear(k)+b*F(q1 = K, q2 = K_minus_k, mu = mu2)*P_linear(K_minus_k))
    return result
  return f


f_jax = {key: get_f(estimator_lam_jax[key], estimator_configs[key]["ca"], estimator_configs[key]["cb"]) for key in estimator_lam_jax}




df = pd.read_pickle("quadratic_results.pkl")
Ks = df["K"]

Ks = np.linspace(Ks.min(), Ks.max(), 300)

Ks = np.linspace(0.001, 0.3, 400)



keypairs = [("n", "n"), ("s", "n"), ("g", "n"), ("x", "n")]#, ("g", "g"), ("s", "s")]


estimator_keys = ["n", "g", "x", "s", "t", "x2"]
estimator_pairs = []
for i, alpha in enumerate(estimator_keys):
    for beta in estimator_keys[i:]:  # Only compute upper triangle (including diagonal)
        estimator_pairs.append((alpha, beta))

print(f"Calculating {len(estimator_pairs)} estimator pairs: {estimator_pairs}")

keypairs = estimator_pairs

start = time.time()
out = {}

for keypair in keypairs:
  keyA, keyB = keypair
  @jax.jit
  def single_calculation(K):
      kmin, kmax = 0.051, 0.15

      def integrand(x):
          # Calculate N(K) = \int_{\vec{k}} [(P(k)+P(K-k))]^2/(PAA(k)*PBB(K-k))
          mu, k = x[:, 0], x[:, 1]
          phi_vol = 2*np.pi
          volume = phi_vol*k**2/(2*np.pi)**3
          K_minus_k = jnp.sqrt(K**2+k**2-2*K*k*mu)
          mask = (k>=kmin) & (k<=kmax)
          mask_2 = (K_minus_k>=kmin) & (K_minus_k<=kmax)
          mask *= mask_2
          fresultA = f_jax[keyA](k, K, mu)
          fresultB = f_jax[keyB](k, K, mu)
          return fresultA*fresultB/(2*P_AA(k)*P_BB(K_minus_k))*volume*mask

      Ndim = 2
      N = 10000**Ndim+1

      mc = MonteCarlo()
      result = mc.integrate(integrand, dim=Ndim, N=N,
                            integration_domain=[[-1, 1], [kmin, kmax]],
                            backend="jax")
      return result

  batch_size = 20 if Ks.size > 20 else 10
  results = []

  for i in range(0, len(Ks), batch_size):
      batch_Ks = jnp.array(Ks[i:i+batch_size])
      batch_results = jax.vmap(single_calculation)(batch_Ks)
      results.append(batch_results)
  results = np.concatenate(results)
  out[keypair] = results


  """# Vectorize the calculation over K values
  calculate_all = jax.vmap(single_calculation)
  results = calculate_all(jnp.array(Ks))
  results = np.array(results)

  out[keypair] = results"""

out["Ks"] = Ks
end = time.time()
print(f"It took {end-start}")


np.save("out", out)


Ks = df["K"]


#keypairs = [("n", "n"), ("s", "n"), ("g", "n"), ("x", "n")]#, ("g", "g"), ("s", "s")]


estimator_keys = ["n", "g", "x", "s", "t", "x2"]
estimator_pairs = []
for i, alpha in enumerate(estimator_keys):
    beta = alpha
    estimator_pairs.append((alpha, beta))

print(f"Cross shot noise: Calculating {len(estimator_pairs)} estimator pairs: {estimator_pairs}")

start = time.time()

keypairs = estimator_pairs

out_cross_shot = {}

for keypair in keypairs:
  keyA, keyB = keypair
  @jax.jit
  def single_calculation(K):
      kmin, kmax = 0.051, 0.15

      def integrand(x):
          # Calculate N(K) = \int_{\vec{k}} [(P(k)+P(K-k))]^2/(PAA(k)*PBB(K-k))
          mu, k = x[:, 0], x[:, 1]
          phi_vol = 2*np.pi
          volume = phi_vol*k**2/(2*np.pi)**3
          K_minus_k = jnp.sqrt(K**2+k**2-2*K*k*mu)
          mask = (k>=kmin) & (k<=kmax)
          mask_2 = (K_minus_k>=kmin) & (K_minus_k<=kmax)
          mask *= mask_2
          fresultA = f_jax[keyA](k, K, mu)
          weightA = fresultA/(2*P_AA(k)*P_BB(K_minus_k))
          shotA = 1/nbar_A
          PS = P_AB(k)*shotA
          return weightA*PS*volume*mask

      Ndim = 2
      N = 10000**Ndim+1

      mc = MonteCarlo()
      result = mc.integrate(integrand, dim=Ndim, N=N,
                            integration_domain=[[-1, 1], [kmin, kmax]],
                            backend="jax")
      return result
  results = []

  for i in range(0, len(Ks), batch_size):
      batch_Ks = jnp.array(Ks[i:i+batch_size])
      batch_results = jax.vmap(single_calculation)(batch_Ks)
      results.append(batch_results)
  results = np.concatenate(results)
  out_cross_shot[keypair] = results

  """# Vectorize the calculation over K values
  calculate_all = jax.vmap(single_calculation)
  results = calculate_all(jnp.array(Ks))
  results = np.array(results)

  out[keypair] = results"""

out_cross_shot["Ks"] = Ks
np.save("out_cross_shot", out_cross_shot)

end = time.time()
print(f"It took {end-start}")