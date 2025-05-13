import numpy as np
import numba

from scipy.fft import fftfreq, rfftfreq




def get_kgrid(box, N):
    return 2 * np.pi * np.stack(
        np.meshgrid(
            fftfreq(N, d=box/N), fftfreq(N, d=box/N), rfftfreq(N, d=box/N),
            indexing='ij',
            )
        )

def get_kgrid_kmag(box, N):
    kgrid = get_kgrid(box, N)
    kmag = (kgrid**2).sum(axis=0)**0.5
    return kgrid, kmag


k_bin_edges_default = np.linspace(1e-4, 1, 100)

@numba.njit(parallel=True)
def calc_power_mu0_x_axis(delta_k, BoxSize, k_bin_edges = k_bin_edges_default, delta_k2 = None):
    """
    Compute the μ=0 power spectrum (perpendicular to x-axis) from a 3D Fourier field
    
    Parameters
    ----------
    delta_k : array_like
        Complex 3D Fourier field of shape (n1d, n1d, n1d//2+1)
    BoxSize : float
        Physical size of the simulation box
    k_bin_edges : array_like
        Edges of the k bins for binning the power spectrum
        
    Returns
    -------
    k_values : ndarray
        Mean k values in each bin
    power : ndarray
        Power spectrum for modes perpendicular to x-axis (μ=0)
    nmodes : ndarray
        Number of modes in each bin
    """
    if delta_k2 is None:
        delta_k2 = delta_k
        
    # Get dimensions
    n1d = delta_k.shape[0]
    middle = n1d // 2
    kzlen = n1d // 2 + 1
    
    # Fundamental mode
    kF = 2.0 * np.pi / BoxSize
    
    # Set up arrays for results
    Nk = len(k_bin_edges) - 1
    k_values = np.zeros(Nk, dtype=np.float64)
    power = np.zeros(Nk, dtype=np.float64)
    nmodes = np.zeros(Nk, dtype=np.int64)
    
    # Convert bin edges to grid units
    kedges = np.zeros_like(k_bin_edges)
    for i in range(len(k_bin_edges)):
        kedges[i] = k_bin_edges[i] / kF
    
    # Loop over all independent modes in the Fourier grid
    for kxx in numba.prange(n1d):
        # Convert array index to k-space value
        kx = kxx - n1d if kxx > middle else kxx
        
        # For μ=0 with x-axis as line of sight, we need kx=0
        if kx != 0:
            continue
            
        for kyy in range(n1d):
            ky = kyy - n1d if kyy > middle else kyy
            
            for kzz in range(kzlen):
                kz = kzz  # kz is always positive due to rfft
                
                # Handle symmetry planes
                if kz == 0 or (kz == middle and n1d % 2 == 0):
                    if ky < 0:
                        continue
                    elif ky == 0 or (ky == middle and n1d % 2 == 0):
                        if kz < 0:
                            continue
                
                # Compute |k| of the mode for binning
                k_perp = np.sqrt(ky*ky + kz*kz)
                
                # Find the bin for this k value
                bin_idx = -1
                for b in range(Nk):
                    if k_perp >= kedges[b] and k_perp < kedges[b+1]:
                        bin_idx = b
                        break
                
                if bin_idx >= 0:
                    # Compute the power
                    #mode_power = np.abs(delta_k[kxx, kyy, kzz])**2
                    mode_power = np.real(delta_k[kxx, kyy, kzz] * np.conj(delta_k2[kxx, kyy, kzz]))
                    
                    # Account for complex conjugate modes
                    mode_count = 1 if (kz == 0 or (kz == middle and n1d % 2 == 0)) else 2
                    
                    # Add to the bin
                    nmodes[bin_idx] += mode_count
                    power[bin_idx] += mode_power * mode_count
                    k_values[bin_idx] += k_perp * kF * mode_count
    
    # Normalize the results
    for i in range(Nk):
        if nmodes[i] > 0:
            power[i] = power[i] / nmodes[i] * BoxSize**3  # Proper units
            k_values[i] = k_values[i] / nmodes[i]
    
    return k_values, power#, nmodes