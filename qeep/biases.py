import jax.numpy as jnp


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



def get_Cg_biases(e, b1A, b1B, b2A, b2B, bs2A, bs2B, bGXA, bGXB, bSXA, bSXB, bTXA, bTXB, asymm_shift = 1., only_asymm_shift = 0.):
    """
    Cg = jnp.array([Cg_g, Cg_ga, Cg_s, Cg_sa, Cg_t, Cg_ta])


    asymm_shift, if zero, asymmetric shift term is zero.
    only_asymm_shift, if 1, only shift asymmetric term is included for the EP.
    """

    biases_A_G = jnp.array([b1A, b2A, bGXA])
    biases_B_G = jnp.array([b1B, b2B, bGXB])

    biases_A_S = jnp.array([b1A, bSXA])
    biases_B_S = jnp.array([b1B, bSXB])

    biases_A_T = jnp.array([b1A, bs2A, bTXA])
    biases_B_T = jnp.array([b1B, bs2B, bTXB])

    Cg_g = symm(cg_g, biases_A_G, biases_B_G, e = e) #need to change definition of symm and asymm
    Cg_ga = asymm(cg_g, biases_A_G, biases_B_G, e = e)*(1-only_asymm_shift)

    Cg_s = symm(cg_s, biases_A_S, biases_B_S, e = e)
    Cg_sa = asymm(cg_s, biases_A_S, biases_B_S, e = e)*asymm_shift

    Cg_t = symm(cg_t, biases_A_T, biases_B_T, e = e)
    Cg_ta = asymm(cg_t, biases_A_T, biases_B_T, e = e)*(1-only_asymm_shift)

    Cg = jnp.array([Cg_g, Cg_s, Cg_t, Cg_ga, Cg_sa, Cg_ta])

    return Cg 