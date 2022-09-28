import numpy as np

import logging

logger = logging.getLogger("finite_time_differences")

def finite_time_differences(
            t_in,
            x_in,
            half_width: int=3
        ):
    """
    Central finite time differences with uniform monodimensional grid spacing in time. The
    accuracy can be set via the half_width. 

    INPUT
    t_in is a vector of n_instances
    x_in is a matrix of n_variables x n_instances. (Finite differences are implemented columnwise).

    OUTPU
    t is a vector of n_instances - 2*half_width
    x is a matrix with the variables known at time t
    dx_dt is the matrix with the variables derivatives at time t
    """
    if half_width>4:
        logger.info('The cofficients for this accuracy are not present in the current implementation. The finite difference is computed with accuracy O(Dt^8)')
        half_width = 4

    # Coefficients for the numerical derivative
    coeff_mat = np.array([[1/2, 2/3, 3/4, 4/5], [0, -1/12, -3/20, -1/5], [0, 0, 1/60, 4/105], [0, 0, 0, -1/280]], dtype=float)

    # Computation of the finite differences
    n_instances = x_in.shape[1]
    base_int = range(half_width, n_instances-half_width)
    x = x_in[:, base_int]
    dx = np.zeros(x.shape)
    for idx in range(1,half_width+1):
        base_int_p = range(half_width+idx, n_instances-half_width+idx)
        base_int_m = range(half_width-idx, n_instances-half_width-idx)
        dx = dx + coeff_mat[idx-1, half_width-1] * (x_in[:, base_int_p] - x_in[:, base_int_m])
    if np.isscalar(t_in) == True:
        dx_dt, t = dx/t_in, t_in
    else:
        t = t_in[base_int]
        dx_dt = dx/(t[1]-t[0])

    return dx_dt, x, t