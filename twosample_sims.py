import numpy as np
from simhelper import *

def noeff_twosample(Nsub, T, ndim, K, effect_size=None):
    y, z, Z, N = make_labels(Nsub, T, ndim, K)
    
    X = gaussian_data(N, Nsub, K, y, Z, ndim, effect_size=0)
    X2 = gaussian_data(N, Nsub, K, y, Z, ndim, effect_size=0)
    return X, X2, y, z

def gaussian_twosample(Nsub, T, ndim, K, effect_size=1):
    y, z, Z, N = make_labels(Nsub, T, ndim, K)
    
    X = gaussian_data(N, Nsub, K, y, Z, ndim, effect_size=effect_size)
    X2 = gaussian_data(N, Nsub, K, y, Z, ndim, effect_size=1.5*effect_size)
    
    return X, X2, y, z
                                     
def ballcirc_twosample(Nsub, T, ndim, K, effect_size=1):
    y, z, Z, N = make_labels(Nsub, T, ndim, K)
    z_persub = z[np.unique(y, return_index=True)[1]]
    
    X = ball_disc_data(z_persub, y, ndim, N, Nsub, effect_size=-.25*effect_size)
    X2 = ball_disc_data(z_persub, y, ndim, N, Nsub, effect_size=.75*effect_size)
    
    return X, X2, y, z