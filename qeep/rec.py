"""
Module with reconstruction utils.
"""

import numpy as np

from scipy.fft import rfftn, irfftn

import rec_utils as utils


def get_rec(key, real_field, box = 2000, kmin = 0, kmax = 0, Ptot_interp = None, Plin_interp = None, nthread = 128, real_field_2 = None):
    if key == "g":
        return get_growth_rec(real_field, box, kmin, kmax, Ptot_interp, Plin_interp, nthread)
    elif key == "s":
        return get_shift_rec(real_field, box, kmin, kmax, Ptot_interp, Plin_interp, nthread)
    elif key == "t":
        return get_tidal_rec(real_field, box, kmin, kmax, Ptot_interp, Plin_interp, nthread)
    elif key == "n":
        return get_shift_n_rec(real_field, box, kmin, kmax, Ptot_interp, Plin_interp, nthread)
    else:
        raise ValueError(f"Key {key} does not exist")

def get_growth_rec(real_field, box = 2000, kmin = 0, kmax = 0, Ptot_interp = None, Plin_interp = None, nthread = 128):
    
    fft_field = rfftn(real_field, overwrite_x = False, workers = nthread)
    N = real_field.shape[0]
    kgrid, kmag = utils.get_kgrid_kmag(box, N)

    selection = (kmag>=kmin) & (kmag<=kmax) #hard cut, but can be a Gaussian filter for example too
    
    delta_A = fft_field*1/Ptot_interp*selection #IVF
    
    delta_B = fft_field*1/Ptot_interp*Plin_interp*selection #WF
    
    delta_A_real = irfftn(delta_A, overwrite_x=True, workers=nthread)
    delta_B_real = irfftn(delta_B, overwrite_x=True, workers=nthread)
    
    product = 4*delta_A_real*delta_B_real*17/21*delta_B.size #assumes symmetry
    
    product_fft = rfftn(product, overwrite_x=False, workers=nthread)

    return product_fft


def get_tidal_rec(real_field, box = 2000, kmin = 0, kmax = 0, Ptot_interp = None, Plin_interp = None, nthread = 128):
    
    fft_field = rfftn(real_field, overwrite_x = False, workers = nthread)
    N = real_field.shape[0]
    kgrid, kmag = utils.get_kgrid_kmag(box, N)

    selection = (kmag>=kmin) & (kmag<=kmax)
    #selection = gauss_filter(kmag)    
    delta_A = fft_field*1/Ptot_interp*selection #IVF
    delta_A_real = irfftn(delta_A, overwrite_x=True, workers=nthread)

    delta_WF = fft_field*1/Ptot_interp*selection*Plin_interp #WF
    delta_WF_real = irfftn(delta_WF, overwrite_x=True, workers=nthread)

    term_delta2 = 1/3*rfftn(delta_A_real*delta_WF_real, overwrite_x=True, workers=nthread)
    del delta_WF_real

    total = 0.
    
    for i in range(3):
        for j in range(i, 3):
            delta_B = fft_field*1/Ptot_interp*Plin_interp*selection*(kgrid[i]/kmag)*(kgrid[j]/kmag)
            delta_B[0, 0, 0] = 0.
            delta_B[kmag == 0] = 0.
            delta_B_real = irfftn(delta_B, overwrite_x=True, workers=nthread)
            product = delta_A_real*delta_B_real
            term = rfftn(product, overwrite_x=False, workers=nthread)
            term *= (kgrid[i]/kmag)*(kgrid[j]/kmag)
            term[kmag == 0.] = 0.
            factor = 1 if (i == j) else 2
            term *= factor
            total += term

    product_fft = 2*2/7*(total-term_delta2)*delta_A_real.size #assumes symmetry
    
    return product_fft


def get_shift_rec(real_field, box = 2000, kmin = 0, kmax = 0, Ptot_interp = None, Plin_interp = None, nthread = 128, ca = None, cb = None):

    fft_field = rfftn(real_field, overwrite_x = False, workers = nthread)
    N = real_field.shape[0]
    kgrid, kmag = utils.get_kgrid_kmag(box, N)

    selection = (kmag>=kmin) & (kmag<=kmax)
    #selection = gauss_filter(kmag)

    delta_A = fft_field*1/Ptot_interp*selection #IVF FIELD
    delta_A_real = irfftn(delta_A, overwrite_x=True, workers=nthread)

    delta_B = fft_field*1/Ptot_interp*Plin_interp*selection #WF FIELD

    inv_kmag_2 = 1/kmag**2
    inv_kmag_2[kmag == 0] = 0


    j_factor = 1j
    
    term = 0.

    for i in range(3):

        term_1 = delta_B*j_factor*(-kgrid[i])*inv_kmag_2
        term_2 = delta_B*j_factor*(-kgrid[i])

        term_1_real = irfftn(term_1, overwrite_x=True, workers=nthread)
        term_2_real = irfftn(term_2, overwrite_x=True, workers=nthread)
        
        product_1 = term_1_real*delta_A_real
        term_1 = rfftn(product_1, overwrite_x=False, workers=nthread)
        term_1 *= j_factor*kgrid[i]

        product_2 = term_2_real*delta_A_real
        term_2 = rfftn(product_2, overwrite_x=False, workers=nthread)
        term_2 *= j_factor*kgrid[i]*inv_kmag_2

        term += (term_1+term_2)*delta_A_real.size
    
    return term


def shift_single(WF, IVF_real, j_factor, ki, inv_kmag_2, nthread):
    term_WA_B = WF*j_factor*(-ki)
    term_WA_B_real = irfftn(term_WA_B, overwrite_x=True, workers=nthread)
    product_WA_B = term_WA_B_real*IVF_real
    product_WA_B_fft = rfftn(product_WA_B, overwrite_x=False, workers=nthread)
    product_WA_B_fft *= j_factor*ki*inv_kmag_2
    return product_WA_B_fft

def get_shift_n_rec(real_field, box = 2000, kmin = 0, kmax = 0, Ptot_interp = None, Plin_interp = None, nthread = 128, real_field_2 = None, Ptot_interp_2 = None):

    fft_field = rfftn(real_field, overwrite_x = False, workers = nthread)

    fft_field_2 = rfftn(real_field_2, overwrite_x = False, workers = nthread) if real_field_2 is not None else fft_field
    Ptot_interp_2 = Ptot_interp_2 if real_field_2 is not None else Ptot_interp

    N = real_field.shape[0]
    kgrid, kmag = utils.get_kgrid_kmag(box, N)

    selection = (kmag>=kmin) & (kmag<=kmax)

    delta_A_ivf = fft_field*1/Ptot_interp*selection #IVF FIELD A
    delta_A_WF = delta_A_ivf*Plin_interp

    delta_B_ivf = fft_field_2*1/Ptot_interp_2*selection #IVF FIELD B
    delta_B_ivf_real = irfftn(delta_B_ivf, overwrite_x=True, workers=nthread)
    #delta_B_WF = delta_B_ivf*Plin_interp

    inv_kmag_2 = 1/kmag**2
    inv_kmag_2[kmag == 0] = 0

    j_factor = 1j
    
    term = 0.
    for i in range(3):

        product_WA_B_fft = shift_single(delta_A_WF, delta_B_ivf_real, j_factor, kgrid[i], inv_kmag_2, nthread)

        #if real_field_2 is None:
        #    product_WB_A_fft = product_WA_B_fft
        #else:
        #    product_WB_A_fft = shift_single(delta_B_WF, delta_A_ivf_real, j_factor, kgrid[i], inv_kmag_2, nthread)

        total = (product_WA_B_fft)
        term += (total)*delta_B_ivf_real.size

    return term*0.5