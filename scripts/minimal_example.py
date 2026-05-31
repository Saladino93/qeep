"""
Minimal self-contained example: the theory normalization of a quadratic estimator.

This is the smallest thing you can run to check that the install works and to see
the main pieces of the code in action. It computes the per-mode normalization N(K)
of the *growth* estimator ("g") for a single tracer, using the linear/non-linear
power spectra shipped in ../data/.

The growth kernel is just a constant (17/21), so this example needs only the core
dependencies -- no sympy / sympy2jax / M(k), unlike the full scripts/thy.py.

Run it from anywhere:

    python scripts/minimal_example.py

It prints N(K) on a few long-wavelength modes K. The estimator normalization
(the response that multiplies the raw quadratic estimator) is 1/N(K).
"""

import os

import jax
# Use double precision everywhere -- important for the power-spectrum dynamic range.
jax.config.update("jax_enable_x64", True)

import numpy as np

from qeep import qeutils


# --- locate the shipped power spectra relative to this file -------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def growth_kernel(q1, q2, mu):
    """Mode-coupling kernel F for the growth estimator: the constant 17/21.

    qeep calls the kernel as F(q1=K, q2=k, mu=cosine angle); multiplying by
    (q1 / q1) keeps the constant broadcast to the right array shape (this matches
    the 'g' kernel in scripts/thy.py).
    """
    return (17.0 / 21.0) * (q1 / q1)


def main():
    # 1. Load linear and non-linear matter power spectra P(k) [columns: k, P(k)].
    kl, pl = np.loadtxt(os.path.join(DATA, "linear_power_abacus.txt")).T
    knl, pnl = np.loadtxt(os.path.join(DATA, "nonlinear_power_abacus.txt")).T

    P_linear = qeutils.get_interpolated(kl, pl)        # log-log interpolated callable

    # 2. Total (observed) power for a single tracer:
    #    P_tot(k) = b1^2 P_nl(k) + 1/nbar   (linear bias + Poisson shot noise).
    b1 = 2.0                 # linear bias of the tracer
    nbar = 1e-3              # number density [ (h/Mpc)^3 ]
    Ptot = b1**2 * pnl + 1.0 / nbar
    P_AA = qeutils.get_interpolated(knl, Ptot)
    P_BB = P_AA             # single tracer: A and B are the same

    # 3. Estimator weight f for the growth kernel.
    #    f = 2[a F(K,-k1) P_lin(k1) + b F(K,-k2) P_lin(k2)], here a = b = 1.
    f_g = qeutils.get_f(growth_kernel, P_linear, a=1.0, b=1.0)

    # 4. Small-scale modes the estimator integrates over.
    kmin, kmax = 0.02, 0.15

    # 5. Per-mode normalization N(K). Nsamples_base is kept small so it runs fast;
    #    increase it (e.g. 1000) for converged results.
    N_of_K = qeutils.N_per_mode(
        f_g, f_g, P_AA, P_BB, kmin=kmin, kmax=kmax, Nsamples_base=300
    )

    # 6. Evaluate on a handful of long-wavelength reconstruction modes K.
    Ks = np.linspace(0.005, 0.02, 5)
    N = qeutils.integrate(Ks, N_of_K, batch_size=2)

    print("Growth estimator, single tracer (b1=%.1f, nbar=%.0e)" % (b1, nbar))
    print(f"{'K [h/Mpc]':>12} {'N(K)':>16} {'norm = 1/N(K)':>18}")
    for K, n in zip(Ks, np.asarray(N)):
        print(f"{K:12.4f} {n:16.6e} {1.0 / n:18.6e}")


if __name__ == "__main__":
    main()
