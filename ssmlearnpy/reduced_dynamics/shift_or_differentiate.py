import logging
import numpy as np
#from ssmlearnpy.utils.finite_time_differences import finite_time_differences
from findiff import FinDiff

#logger = logging.getlogger("shift_or_differentiate")

def shift_or_differentiate(x, t, type, accuracy = 4):
    """
    The function prepares the data for regression of the reduced dynamics. 
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
           # create finite difference object    
            fd = FinDiff(1, t[i_traj][1] - t[i_traj][0], 1, acc = accuracy) # first order differential with given accuracy
            dx_dt_traj = fd(np.array(x[i_traj])) # findiff differentiates along axis 0.
            X.append(x[i_traj])
            y.append( dx_dt_traj)
    else:
        raise NotImplementedError(
            (
                f"{type} not available, please specify a type that has "
                f"already been implemented, otherwise raise an issue to the developers"
            )
        )
    return X, y 