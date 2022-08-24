import numpy as np


def make_labels(Nsub, T, ndim, K):
    N = Nsub*T
    ylab = [[i for j in range(0, T)] for i in range(0, Nsub)]
    y = np.concatenate(ylab)
    zsublab = np.random.choice(K, size=Nsub)
    z = np.concatenate([[zi for j in range(0, T)] for zi in zsublab])
    Z = np.zeros((N, K))
    for i, zi in enumerate(z):
        Z[i, zi] = 1
    return y, z, Z, N

def gaussian_data(N, Nsub, K, y, Z, ndim, effect_size=1):
    # class effects
    beta_ks = []
    for k in range(0, K):
        beta_ks.append(2*effect_size*k/np.sqrt(ndim)*np.ones((ndim, 1)))
    B = np.hstack(beta_ks)

    # subject-specific effects are normally distributed
    vs = np.random.normal(size=(Nsub, ndim))

    sub_idx = np.zeros((N, Nsub))
    for i, yi in enumerate(y):
        sub_idx[i, yi] = 1
    V = sub_idx @ vs

    X = V + Z@B.transpose() + np.random.normal(size=(N, ndim))
    return X

def sample_sphere(npoints, ndim, radius_sphere=1):
    # points on sphere

    V_sphere = np.random.normal(0, 1, (npoints, ndim))
    norms = np.linalg.norm(V_sphere, axis=1)
    norms = norms.reshape(-1, 1).repeat(V_sphere.shape[1], axis=1)
    return V_sphere/norms * radius_sphere * np.sqrt(ndim)

def sample_ball(npoints, ndim, radius_ball=1):
    # points in ball
    radius_ball = 1

    V_ball = np.random.normal(0, 1, (npoints, ndim))
    norms = np.linalg.norm(V_ball, axis=1)[:, np.newaxis]
    V_ball /= norms
    uniform_points = np.random.uniform(size=npoints)[:, np.newaxis]
    new_radii = np.power(uniform_points, 1/ndim)
    V_ball *= new_radii
    return V_ball * radius_ball
    
def ball_disc_data(z_persub, y, ndim, N, Nsub, effect_size=1):
    V = np.zeros((Nsub, ndim))
    
    radius_ball = 1
    
    rad_sphere = (radius_ball*(1 + effect_size/np.sqrt(ndim)))
    V[z_persub == 1] = sample_sphere(int((z_persub == 1).sum()), ndim, 
                                     radius_sphere=rad_sphere)

    V[z_persub == 0] = sample_ball(int((z_persub == 0).sum()), ndim, 
                                        radius_ball=radius_ball)
    
    sub_idx = np.zeros((N, Nsub))
    for i, yi in enumerate(y):
        sub_idx[i, yi] = 1
    X = sub_idx @ V + np.random.normal(0, .4, size=(N, ndim))
    return X