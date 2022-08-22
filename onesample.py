import hyppo
from hyppo.discrim._utils import _CheckInputs
from hyppo.independence import Dcorr
from dask.distributed import Client, progress
import numpy as np
from statsmodels.multivariate.manova import MANOVA
import pandas as pd

def statistic(x, y, z, is_distance=False, remove_isolates=True):
    r"""
    Calculates the independence test statistic.

    Parameters
    ----------
    x, y, z : ndarray of float
        Input data matrices.
    """
    # TODO: add something that checks whether z respects y
    # that is, for each *unique* individual in y, that all
    # of that unique individual's class labels are the same
    check_input = _CheckInputs(
        [x],
        y,
        is_dist=is_distance,
        remove_isolates=remove_isolates,
    )
    x, y = check_input()

    x = np.asarray(x[0])
    y = y
    z = z

    weights = []
    zs = np.unique(z)
    K = len(zs)
    N = x.shape[0]

    weights = np.zeros((K, K))
    discrs = np.zeros((K, K))
    for i, z1 in enumerate(zs):
        Nz1 = (z == z1).sum()
        for j, z2 in enumerate(zs):
            if z1 == z2:
                Nz2 = Nz1 - 1
            else:
                Nz2 = (z == z2).sum()
            weights[i, j] = Nz1*Nz2/(N*(N - 1))
            discrs[i, j] = _statistic_zzp(x, y, z, z1=z1, z2=z2)
    w = within_sub_discr(discrs, weights)
    b = between_sub_discr(discrs, weights)
    ratio = b/w
    return w, b, ratio

def _statistic_zzp(x, y, z, z1, z2):
    r"""
    Calulates the independence test statistic.

    Parameters
    ----------
    x, y : ndarray of float
        Input data matrices.
    """
    rdfs = []
    # isolate analysis to only elements from classes z1 or z2
    idx_z1z2 = np.where(np.logical_or(z == z1, z == z2))[0]
    y_z1z2 = y[idx_z1z2]
    z_z1z2 = z[idx_z1z2]
    for i in idx_z1z2:
        # the class label of object i
        z_i = z[i]
        y_i = y[i]
        # the individual label of object i
        ind_i = y[i]
        # get all of the distances from i to other items that have class
        # of z1 or z2, where the individual label is the same
        Dii = x[i][idx_z1z2][y_z1z2 == ind_i]
        if z_i == z1:
            z_oth = z2
        else:
            z_oth = z1
        # get all of the distances from i to other items that have
        # class of z1 or z2, where the individual label is different
        # and the class is the class that object i is not
        Dij = x[i][idx_z1z2][np.logical_and(z_z1z2 == z_oth, y_z1z2 != y_i)]

        rdf = [1 - ((Dij < d).sum() + 0.5 * (Dij == d).sum()) / Dij.size for d in Dii]
        rdfs.append(rdf)
    stat = np.array(rdfs).mean()
    return stat

def permute_z_respect_y(y, z, y_unique=None, unique_idx=None):
    if (y_unique is None) or (unique_idx is None):
        y_unique, unique_idx = np.unique(y, return_index=True)
    z_unique = z[unique_idx]
    z_unique_permuted = np.random.permutation(z_unique)
    z_permuted = np.zeros((len(y)))
    for i, yi in enumerate(y_unique):
        z_permuted[y == yi] = z_unique_permuted[i]
    return z_permuted
    
def one_sample_test(x, y, z, reps=1000, is_distance=False, remove_isolates=True):
    r"""
    Calculates the independence test statistic.

    Parameters
    ----------
    x, y, z : ndarray of float
        Input data matrices.
    """
    check_input = _CheckInputs(
        [x],
        y,
        is_dist=is_distance,
        remove_isolates=remove_isolates,
    )
    x, y = check_input()

    x = np.asarray(x[0])
    y = y
    z = z
    
    y_unique, unique_idx = np.unique(y, return_index=True)

    sample_stat = statistic(x, y, z, is_distance=True, remove_isolates=False)[2]
    
    # run permutation test
    null_stat = np.zeros((reps))
    for i in range(reps):
        null_stat[i] = statistic(x, y, permute_z_respect_y(y, z, y_unique=y_unique, unique_idx=unique_idx),
                                is_distance=True, remove_isolates=False)[2]
    pvalue = ((null_stat >= sample_stat).sum() + 1)/(reps + 1)
    return sample_stat, pvalue, null_stat

def manova_statistic():
    try:
        indep = pd.DataFrame({"Individual": y, "Class": z})
        model = MANOVA(x, indep)
        test = model.mv_test()
        return test.results["x1"]["stat"]["Value"][0]
    except:
        return np.nan
    
def dcorr_statistic():
    return Dcorr().statistic(x, z)

def one_sample_manova(x, y, z):
    try:
        indep = pd.DataFrame({"Individual": y, "Class": z})
        model = MANOVA(x, indep)
        test = model.mv_test()
        return test.results["x1"]["stat"]["Pr > F"][0]
    except:
        return np.nan

def one_sample_dcorr(x, z, reps=1000, is_distance=False):
    return Dcorr().test(x, z, reps=reps)

def within_sub_discr(discrs, weights):
    return (np.diag(weights)*np.diag(discrs)/(np.diag(weights).sum())).sum()

def between_sub_discr(discrs, weights):
    K = discrs.shape[0]
    return (np.extract(1 - np.eye(K), weights)*np.extract(1 - np.eye(K), discrs)/(np.extract(1 - np.eye(K), weights).sum())).sum()