"""
Get Fisher matrix.
"""

import jax
import jax.numpy as jnp

from interpax import Interpolator1D

from torchquad import Simpson, set_up_backend
set_up_backend("jax", data_type="float64")



def safe_inv(C, eps = 1e-8):
    """Safely invert a matrix or batch of matrices with regularization.
    
    Args:
        C: Input matrix or batch of matrices to invert
        eps: Small regularization constant for numerical stability
        
    Returns:
        The inverted matrix or batch of matrices
    """
    # Get the shape of the input
    shape = C.shape
    
    # For a single matrix (2D)
    if len(shape) == 2:
        n = shape[0]
        # Ensure the matrix is square
        if shape[0] != shape[1]:
            raise ValueError(f"Matrix must be square but got shape {shape}")
        reg_C = C + eps * jnp.eye(n)
        return jnp.linalg.inv(reg_C)
    
    # For batched matrices (3D or more)
    elif len(shape) >= 3:
        # Get the last two dimensions which should be equal for square matrices
        n = shape[-1]
        if shape[-2] != shape[-1]:
            raise ValueError(f"Matrices must be square but got shape {shape}")
        # Create an identity matrix of appropriate size and broadcast it
        identity = jnp.eye(n)
        # Broadcast the identity to match the batch dimensions
        for _ in range(len(shape) - 2):
            identity = identity[None, ...]
        # Add regularization
        reg_C = C + eps * identity
        return jnp.linalg.inv(reg_C)
    
    else:
        raise ValueError(f"Input must be at least 2D but got shape {shape}")


def fisher_per_mode(v, K_array, Cfunc, eps=1e-8):
    """
    Compute the Fisher matrix per mode for a general covariance matrix C,
    where Cfunc(K_array, v) returns (n_modes, nprobes, nprobes).
    """

    n_params = len(v)
    n_modes = len(K_array)

    # C: (n_modes, nprobes, nprobes)
    C = Cfunc(K_array, v)
    Cinv = safe_inv(C, eps)  # (n_modes, nprobes, nprobes)

    # Compute derivatives: dC_dv has shape (n_modes, nprobes, nprobes, n_params)
    dC_dv = jax.jacfwd(Cfunc, argnums=1)(K_array, v)  # (n_modes, nprobes, nprobes, n_params)

    # Fisher matrix per mode
    F = jnp.zeros((n_modes, n_params, n_params))
    for a in range(n_params):
        for b in range(n_params):
            # term_a: (n_modes, nprobes, nprobes)
            term_a = jnp.einsum('mij,mjk->mik', dC_dv[..., a], Cinv)
            term_b = jnp.einsum('mij,mjk->mik', dC_dv[..., b], Cinv)
            # product: (n_modes, nprobes, nprobes)
            product = jnp.einsum('mij,mjk->mik', term_a, term_b)
            # trace: (n_modes,)
            trace = jnp.einsum('mii->m', product)
            F = F.at[:, a, b].set(0.5 * trace)
    return F



def get_F_interp(K_array, Fmatrix):
    """
    Given a Fisher matrix per mode, interpolates to give a function.
    """
    @jax.jit
    def F_interp(K_eval):
        # K_eval: scalar or array
        # Returns: (..., n_params, n_params)
        n_params = Fmatrix.shape[1]
        F_interpolated = [[Interpolator1D(K_array, Fmatrix[:, a, b], method='cubic') for b in range(n_params)] for a in range(n_params)]
        return jnp.stack([
            jnp.stack([F_interpolated[a][b](K_eval) for b in range(n_params)], axis=-1)
            for a in range(n_params)
        ], axis=-2)
    return F_interp



def get_F_integrated(K_array, F, k_min_analysis = 0.01, k_max_analysis = 0.05, V = 1):
    """
    Given a Fisher matrix per mode, integrates to give a function.

    V is in Gpc^3 h^{-3}
    """

    V *= 1e9 #to Mpc^3 h^{-3}

    n_params = F.shape[1]

    Finterp = get_F_interp(K_array, F)

    integrator = Simpson()

    K_min = k_min_analysis
    K_max = k_max_analysis

    F_integrated = jnp.zeros((n_params, n_params))

    for a in range(n_params):
        for b in range(a, n_params):
            def scalar_integrand(K):
                # K: (N, 1)
                K = K.squeeze(-1)
                # Finterp should return (..., n_params, n_params)
                return K**2 * Finterp(K)[..., a, b]
            result = integrator.integrate(
                scalar_integrand,
                dim=1,
                N=199,
                integration_domain=[[K_min, K_max]]
            )
            F_integrated = F_integrated.at[a, b].set(result)
            if a != b:
                F_integrated = F_integrated.at[b, a].set(result)

    # Multiply by prefactor (factor of 2 from mu integration)
    F_integrated = F_integrated * (2 * V / (2 * jnp.pi)**2)

    return F_integrated


"""def get_error_bars(F_integrated):
    #Get error bars for the parameters.
    error_bars_marg = jnp.sqrt(jnp.diag(jnp.linalg.inv(F_integrated)))
    error_bars_unmarg = jnp.pow(jnp.diag((F_integrated)), -0.5)
    return error_bars_marg, error_bars_unmarg"""


def get_covariance_matrix_from_F(F_integrated):
    """
    Get covariance matrix for the parameters.
    """
    return safe_inv(F_integrated)

def get_error_bars_from_F(F_integrated):
    """
    Get error bars for the parameters.
    """
    error_bars_marg = jnp.sqrt(jnp.diag(get_covariance_matrix_from_F(F_integrated)))
    error_bars_unmarg = jnp.pow(jnp.diag(F_integrated), -0.5)
    return error_bars_marg, error_bars_unmarg


def get_fisher_matrix(v, K_array, Cfunc, k_min_analysis = 0.01, k_max_analysis = 0.05, V = 1):
    """
    Get Fisher matrix for the parameters.
    """
    F = fisher_per_mode(v, K_array, Cfunc)
    F_integrated = get_F_integrated(K_array, F, k_min_analysis, k_max_analysis, V)
    return F_integrated