import logging
import numpy as np
# from ssmlearnpy.utils.finite_time_differences import finite_time_differences
from findiff import FinDiff
from scipy.interpolate import UnivariateSpline

#logger = logging.getlogger("shift_or_differentiate")

def shift_or_differentiate(x, t, type, accuracy = None, method="findiff"):
    """
    Prepares the data for regression of reduced dynamics.

    Parameters:
        x: list of np.ndarray, each of shape (dim, time)
        t: list of np.ndarray, each of shape (time,)
        type: 'map' or 'flow'
        accuracy: 
            - if method == 'findiff': stencil accuracy (default 8)
            - if method == 'spline': spline degree k (default 3)
        method: differentiation method ('findiff' or 'spline')

    Returns:
        X, y: lists of np.ndarray
    """
    if(type == 'map'):
        X, y = [], []
        #logger.info("Shift data for discrete time dynamical system")
        for i_traj in range(len(x)):
            X.append( x[i_traj][:, :-1])
            y.append( x[i_traj][:, 1:])
            
    elif(type == 'flow'):
        #logger.info("Differentiate data for continuous time dynamical system")
        X, y = [], []
        for i_traj in range(len(x)):
            traj = np.array(x[i_traj])
            time = np.array(t[i_traj])
            dt = time[1] - time[0]

            if method == 'findiff':
                accuracy = accuracy or 8
                fd = FinDiff(1, dt, 1, acc=accuracy)
                dx_dt_traj = fd(traj)
            elif method == 'spline':
                accuracy = accuracy or 3
                dx_dt_traj = np.zeros_like(traj)
                for d in range(traj.shape[0]):
                    spline = UnivariateSpline(time, traj[d, :], s=1e-10, k=accuracy)
                    dx_dt_traj[d, :] = spline.derivative()(time)
            else:
                raise NotImplementedError(f"Method '{method}' not supported.")
            X.append(x[i_traj])
            y.append(dx_dt_traj)
    else:
        raise NotImplementedError(
            (
                f"{type} not available, please specify a type that has "
                f"already been implemented, otherwise raise an issue to the developers"
            )
        )
    return X, y 