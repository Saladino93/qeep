# Quadratic Estimator for Cosmological Field Reconstruction

**Note, this was generated with Claude Sonnet 4.**

## Overview

This document describes the mathematical formulation of a quadratic estimator used for cosmological field reconstruction. The estimator combines inverse variance filtering with Wiener filtering to optimally extract growth information from cosmological density fields while suppressing noise.

## Mathematical Framework

### Field Preprocessing

The input real-space field is Fourier transformed:

$\tilde{\delta}(\mathbf{k}) = \text{FFT}[\delta(\mathbf{x})]$

### Inverse Variance Filtered (IVF) Field

The IVF field is constructed by weighting the Fourier modes by the inverse of the total power spectrum:

$\tilde{\delta}_{\text{IVF}}(\mathbf{k}) = \frac{\tilde{\delta}(\mathbf{k})}{P_{\text{tot}}(k)} \cdot W(k)$

**Parameters:**
- $P_{\text{tot}}(k)$ is the total power spectrum (signal + noise)
- $W(k)$ is the selection function: $W(k) = 1$ if $k_{\min} \leq k \leq k_{\max}$, and $0$ otherwise

**Purpose:** The IVF weighting suppresses noise-dominated modes while preserving signal-dominated modes.

### Wiener Filtered (WF) Field

The WF field multiplies the IVF field by the linear power spectrum:

$\tilde{\delta}_{\text{WF}}(\mathbf{k}) = \tilde{\delta}_{\text{IVF}}(\mathbf{k}) \cdot P_{\text{lin}}(k)$

**Parameters:**
- $P_{\text{lin}}(k)$ is the linear matter power spectrum

**Purpose:** The Wiener filter provides the optimal linear reconstruction of the underlying density field.

### Quadratic Estimator Construction

For two fields A and B, the quadratic products are formed through the following steps:

#### Real-space transformations:

$\delta_{\text{IVF}}^A(\mathbf{x}) = \text{IFFT}[\tilde{\delta}_{\text{IVF}}^A(\mathbf{k})]$

$\delta_{\text{WF}}^B(\mathbf{x}) = \text{IFFT}[\tilde{\delta}_{\text{WF}}^B(\mathbf{k})]$

#### Quadratic products:

$P_{AB}(\mathbf{x}) = \delta_{\text{IVF}}^A(\mathbf{x}) \cdot \delta_{\text{WF}}^B(\mathbf{x})$

$P_{BA}(\mathbf{x}) = \delta_{\text{IVF}}^B(\mathbf{x}) \cdot \delta_{\text{WF}}^A(\mathbf{x})$

### Final Estimated Field

The final growth-reconstructed field is:

$\tilde{F}_g(\mathbf{k}) = \text{FFT}\left[ 2F_g \frac{N_{\text{modes}}}{2} (P_{AB}(\mathbf{x}) + P_{BA}(\mathbf{x})) \right]$

**Parameters:**
- $F_g = \frac{17}{21}$ is the growth factor coefficient
- $N_{\text{modes}}$ is the total number of Fourier modes
- The factor of 2 in the denominator accounts for weight normalization

### Special Case: Single Field

When only one field is provided (`real_field_2` is None):

$P_{BA}(\mathbf{x}) = P_{AB}(\mathbf{x}) = \delta_{\text{IVF}}(\mathbf{x}) \cdot \delta_{\text{WF}}(\mathbf{x})$

## Complete Mathematical Expression

The final estimated field in Fourier space is:


$\boxed{
\tilde{F}_g(\mathbf{k}) = \text{FFT}\left[ \frac{17N_{\text{modes}}}{21} \left( \delta_{\text{IVF}}^A(\mathbf{x}) \delta_{\text{WF}}^B(\mathbf{x}) + \delta_{\text{IVF}}^B(\mathbf{x}) \delta_{\text{WF}}^A(\mathbf{x}) \right) \right]
}$

where each filtered field is constructed as:

$\delta_{\text{IVF}}^{A,B}(\mathbf{x}) = \text{IFFT}\left[ \frac{\tilde{\delta}^{A,B}(\mathbf{k})}{P_{\text{tot}}^{A,B}(k)} W(k) \right]$

$\delta_{\text{WF}}^{A,B}(\mathbf{x}) = \text{IFFT}\left[ \frac{\tilde{\delta}^{A,B}(\mathbf{k}) P_{\text{lin}}(k)}{P_{\text{tot}}^{A,B}(k)} W(k) \right]$

## Implementation Notes

### Function: `get_ivf_wf_selected`

**Input Parameters:**
- `real_field`: Input density field in real space
- `Ptot_interp`: Interpolated total power spectrum function
- `Plin_interp`: Interpolated linear power spectrum function
- `kmin`, `kmax`: k-space selection bounds
- `box`: Box size for k-grid calculation
- `nthread`: Number of threads for FFT operations
- `real_field_2`: Optional second field for cross-correlation
- `Ptot_interp_2`: Optional second total power spectrum

**Output:**
- `delta_ivf`: IVF field in Fourier space
- `delta_WF`: WF field in Fourier space
- `delta_ivf_2`: Second IVF field (if provided)
- `delta_WF_2`: Second WF field (if provided)

### Function: `get_growth_rec`

**Input Parameters:**
- Same as `get_ivf_wf_selected`

**Output:**
- `product_fft`: Final reconstructed growth field in Fourier space