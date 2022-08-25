from typing import NamedTuple

import numpy as np
from numba import jit
from scipy._lib._util import MapWrapper

from hyppo.discrim._utils import _CheckInputs
from sklearn.utils import check_random_state
from onesample import *

def two_sample_test(x1, x2, y, z, reps=1000, is_distance=False, remove_isolates=True):    
    try:
        ddif_obs = statistic(x2, y, z)[2] - statistic(x1, y, z)[2]
    except:
        ddif_obs = np.nan
    try:
        cordif_obs = dcorr_statistic(x2, z) - dcorr_statistic(x1, z)
    except:
        cordif_obs = np.nan
    try:
        manovadif_obs = manova_statistic(x2, y, z) - manova_statistic(x1, y, z)
    except:
        manovadif_obs = np.nan
    
    null_diff_cdiscr = np.zeros((reps))
    null_diff_cordif = np.zeros((reps))
    null_diff_manova = np.zeros((reps))
    
    y_unique, unique_idx = np.unique(y, return_index=True)

    for i in range(reps):
        z_reord = permute_z_respect_y(y, z, y_unique=y_unique, unique_idx=unique_idx)
        try:
            null_diff_cdiscr[i] = statistic(x2, y, z_reord)[2] - statistic(x1, y, z_reord)[2]
        except:
            null_diff_cdiscr[i] = np.nan
        try:
            null_diff_cordif[i] = dcorr_statistic(x2, z_reord) - dcorr_statistic(x1, z_reord)
        except:
            null_diff_cordif[i] = np.nan
        try:
            null_diff_manova[i] = manova_statistic(x2, y, z_reord) - manova_statistic(x1, y, z_reord)
        except:
            null_diff_manova[i] = np.nan
    pvalue_cdiscr = ((null_diff_cdiscr >= ddif_obs).sum() + 1)/(reps + 1)
    pvalue_dcor = ((null_diff_cordif >= cordif_obs).sum() + 1)/(reps + 1)
    pvalue_manova = ((null_diff_manova >= manovadif_obs).sum() + 1)/(reps + 1)

    return pvalue_cdiscr, pvalue_dcor, pvalue_manova