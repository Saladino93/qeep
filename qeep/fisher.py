"""
Get Fisher matrix.
"""

import jax
import jax.numpy as jnp

from interpax import Interpolator1D

from torchquad import Simpson, set_up_backend
set_up_backend("jax", data_type="float64")



def safe_inv(C, eps = 1e-40):
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




def fisher_per_mode_single_with_covariance(v, K_array, Ofunc, covariance_func):
    """
    Compute the Fisher matrix per mode for a general observable O,
    where Ofunc(K_array, v) returns (n_modes, nprobes, nprobes).
    Assumes diagonal covariance matrix as described in the variance_func.
    """

    n_params = len(v)
    n_modes = len(K_array)

    #O = Ofunc(K_array, v)
    C = covariance_func(K_array, v)

    Cinv = safe_inv(C)

    # Compute derivatives: dO_dv has shape (n_modes, nprobes, n_params)
    dO_dv = jax.jacfwd(Ofunc, argnums=1)(K_array, v)  # (n_modes, n_probes, n_params)

    # Fisher matrix per mode
    F = jnp.zeros((n_modes, n_params, n_params))

    #k is mode, i is probe, a is parameter, j is probe
    F = jnp.einsum('kia, kij, kjb -> kab', dO_dv, Cinv, dO_dv)
    return F

def fisher_per_mode_single(v, K_array, Ofunc, variance_func):
    """
    Compute the Fisher matrix per mode for a general observable O,
    where Ofunc(K_array, v) returns (n_modes, nprobes, nprobes).
    Assumes diagonal covariance matrix as described in the variance_func.
    """

    n_params = len(v)
    n_modes = len(K_array)

    #O = Ofunc(K_array, v)
    V = variance_func(K_array, v)

    # Compute derivatives: dO_dv has shape (n_modes, nprobes, nprobes, n_params)
    dO_dv = jax.jacfwd(Ofunc, argnums=1)(K_array, v)  # (n_modes, n_params)

    # Fisher matrix per mode
    F = jnp.zeros((n_modes, n_params, n_params))
    for a in range(n_params):
        for b in range(a, n_params):
            # term_a: (n_modes, nprobes, nprobes)
            term_a = dO_dv[:, a]
            term_b = dO_dv[:, b]
            # product: (n_modes, nprobes, nprobes)
            product = term_a * term_b / V
            F = F.at[:, a, b].set(product)
            if a != b:
                F = F.at[:, b, a].set(product)
    return F

def fisher_per_mode(v, K_array, Cfunc, eps=1e-30):
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

    FF = jnp.einsum('kij,kjla,klm, kmib ->kab', Cinv, dC_dv, Cinv, dC_dv)*0.5

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
    For now, no mu dependence assumed in F.

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
                dim=1, #integration over K, potentially over mu would be dim=2
                N=199,
                integration_domain=[[K_min, K_max]]
            )
            F_integrated = F_integrated.at[a, b].set(result)
            if a != b:
                F_integrated = F_integrated.at[b, a].set(result)

    # Multiply by prefactor (factor of 2 from mu integration)
    F_integrated = F_integrated * (2 * V / (2 * jnp.pi)**2)

    return F_integrated


def get_F_integrated_fast(K_array, F, k_min_analysis=0.01, k_max_analysis=0.05, V=1, N=199):
    """
    Fast vectorized version of get_F_integrated.
    Given a Fisher matrix per mode, integrates to give a function.
    Optimized version that vectorizes the integration over all parameter pairs.
    
    Args:
        K_array: Array of k values for interpolation
        F: Fisher matrix per mode (n_modes, n_params, n_params)
        k_min_analysis: Minimum k for integration
        k_max_analysis: Maximum k for integration
        V: Volume in Gpc^3 h^{-3}
    
    Returns:
        F_integrated: Integrated Fisher matrix (n_params, n_params)
    """
    V *= 1e9  # Convert to Mpc^3 h^{-3}
    n_params = F.shape[1]

    # Get interpolation function
    Finterp = get_F_interp_symmetric_memory_efficient(K_array, F)
    integrator = Simpson()

    # Vectorized integrand that computes all matrix elements at once
    def vectorized_integrand(K):
        """
        Compute K^2 * F(K) for all parameter pairs simultaneously.
        
        Args:
            K: Integration points (N, 1)
            
        Returns:
            K^2 * F(K) with shape (N, n_params, n_params)
        """
        K_squeezed = K.squeeze(-1)  # (N,)
        F_vals = Finterp(K_squeezed)  # (N, n_params, n_params)
        K2_expanded = K_squeezed**2  # (N,)
        # Broadcast K^2 to match F dimensions: (N,) -> (N, 1, 1)
        K2_broadcast = K2_expanded[..., None, None]
        return K2_broadcast * F_vals  # (N, n_params, n_params)

    # Integrate all matrix elements simultaneously
    F_integrated_raw = integrator.integrate(
        vectorized_integrand,
        dim=1,  # Integration over K
        N=N,
        integration_domain=[[k_min_analysis, k_max_analysis]]
    )

    # Ensure symmetry (should already be symmetric, but enforce numerically)
    F_integrated = 0.5 * (F_integrated_raw + F_integrated_raw.T)

    # Apply prefactor (factor of 2 from mu integration)
    prefactor = 2 * V / (2 * jnp.pi)**2
    F_integrated = F_integrated * prefactor

    return F_integrated



def get_F_interp_symmetric_memory_efficient(K_array, Fmatrix):
    """
    Most memory-efficient version - minimal intermediate arrays.
    """
    n_k, n_params, _ = Fmatrix.shape
    
    # Pre-compute indices
    i_upper, j_upper = jnp.triu_indices(n_params)
    n_unique = len(i_upper)
    
    # Extract and interpolate only unique elements
    Fmatrix_unique = Fmatrix[:, i_upper, j_upper]
    F_interpolator = Interpolator1D(K_array, Fmatrix_unique, method='cubic')
    
    @jax.jit
    def F_interp(K_eval):
        unique_vals = F_interpolator(K_eval)
        
        # Direct reconstruction without intermediate full arrays
        if unique_vals.ndim == 1:
            # Scalar K_eval case
            F = jnp.zeros((n_params, n_params))
            F = F.at[i_upper, j_upper].set(unique_vals)
            F = F.at[j_upper, i_upper].set(unique_vals)
            return F
        else:
            # Array K_eval case
            batch_size = unique_vals.shape[0]
            F = jnp.zeros((batch_size, n_params, n_params))
            F = F.at[:, i_upper, j_upper].set(unique_vals)
            F = F.at[:, j_upper, i_upper].set(unique_vals)
            return F
    
    return F_interp




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
    Get Fisher matrix for the parameters. Cfunc needs to include all auto and cross spectra.
    """
    F = fisher_per_mode(v, K_array, Cfunc)
    F_integrated = get_F_integrated(K_array, F, k_min_analysis, k_max_analysis, V)
    return F_integrated



def get_F_integrated_fast_new(K_array, F, k_min_analysis=0.01, k_max_analysis=0.05, V=1):
    """
    Fast vectorized version of get_F_integrated.
    Given a Fisher matrix per mode, integrates to give a function.
    Optimized version that vectorizes the integration over all parameter pairs.
    
    Args:
        K_array: Array of k values for interpolation
        F: Fisher matrix per mode (n_modes, n_params, n_params)
        k_min_analysis: Minimum k for integration
        k_max_analysis: Maximum k for integration
        V: Volume in Gpc^3 h^{-3}
    
    Returns:
        F_integrated: Integrated Fisher matrix (n_params, n_params)
    """
    V *= 1e9  # Convert to Mpc^3 h^{-3}

    mask = (K_array > k_min_analysis) & (K_array < k_max_analysis)
    F_sel = F[mask]
    K_sel = K_array[mask][:, None, None]

    F_integrated = jnp.trapezoid(F_sel*K_sel**2, K_sel, axis=0)

    # Apply prefactor (factor of 2 from mu integration)
    prefactor = 2 * V / (2 * jnp.pi)**2
    F_integrated = F_integrated * prefactor

    return F_integrated