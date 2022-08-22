import numpy as np
from simhelper import *

def noeff_onesample(Nsub, T, ndim, K, effect_size=None):
    y, z, Z, N = make_labels(Nsub, T, ndim, K)
    
    X = gaussian_data(N, Nsub, K, y, Z, ndim, effect_size=0)
    return X, y, z

def gaussian_onesample(Nsub, T, ndim, K, effect_size=1):
    y, z, Z, N = make_labels(Nsub, T, ndim, K)
    
    X = gaussian_data(N, Nsub, K, y, Z, ndim, effect_size=effect_size)
    
    return X, y, z
                                     
def ballcirc_onesample(Nsub, T, ndim, K, effect_size=1):
    y, z, Z, N = make_labels(Nsub, T, ndim, K)
    z_persub = z[np.unique(y, return_index=True)[1]]
    X = ball_disc_data(z_persub, y, ndim, N, Nsub, effect_size=effect_size)
    return X, y, z