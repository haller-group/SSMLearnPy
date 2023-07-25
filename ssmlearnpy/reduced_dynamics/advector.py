from scipy.integrate import solve_ivp
from ssmlearnpy.utils.iterate_map import iterate_map
import numpy as np


class TimeStepper:
    """
    General class to advect trajectories

    Parameters:
        - n_type: type of the reduced dynamics. Can be 'map' or 'flow'
        - dynamics: should be a function returning the vector field or the map
    """
    def __init__(self, n_type, n_dynamics, n_dt = 1):
        self.type = n_type
        self.dynamics = n_dynamics
        self.dt = n_dt # for the map

    def advect(self, timespan, x0, p0 = None, **integration_args):
        l_x0 = len(x0)
        if self.type == 'flow':
            def vector_field(t,x):
                if p0 is not None: 
                    x = np.concatenate((x, p0.reshape(1,-1)))
                dx_dt = self.dynamics(x.reshape(1,l_x0))
                return dx_dt
            if len(timespan) == 2:
                sol = solve_ivp(vector_field, timespan, x0, **integration_args)
            else:
                sol = solve_ivp(vector_field, [timespan[0], timespan[-1]], x0, t_eval = timespan, **integration_args)
            return sol.t, sol.y, sol.success
        if self.type == 'map':
            success = True
            iterations = [int(x/self.dt) for x in timespan]
            numberof_iterations = iterations[-1] - iterations[0]
            if p0 is not None:
                def function_to_iterate(x):
                    xx = np.concatenate((x, p0.reshape(1,-1)))
                    return self.dynamics(xx.reshape(1,-1))[:l_x0, 0]
            else:
                function_to_iterate = self.dynamics
            sol = iterate_map(function_to_iterate, numberof_iterations, x0=x0)

            if len(iterations) == 2:
                time_at_iterations = np.arange(iterations[0], iterations[1], 1) * self.dt
            else:
                time_at_iterations = timespan

            if np.sum(np.isnan(sol)) > 0:
                success = False
            return time_at_iterations, sol, success


def advect(dynamics, t, x, dynamics_type,  parameters = None, **integration_args):
    """
    Wrapper function to advect trajectories

    Parameters:
        - dynamics: should be a function returning the vector field or the map
        - t: list of time points for each trajectory
        - x: list of trajectories
        - dynamics_type: type of the reduced dynamics. Can be 'map' or 'flow'
        - parameters: parameters to be forwarded to the dynamics
        - integration_args: arguments to be forwarded to solve_ivp

    Returns:
        - t_predict: list of time points for each trajectory
        - x_predict: list of predicted trajectories
    """
    time_stepper = TimeStepper(dynamics_type, dynamics, t[0][1] - t[0][0])
    x_predict,  t_predict = [], []
    for i_traj in range(len(x)):
        if parameters is not None:
            t_traj, x_traj, success_traj = time_stepper.advect(t[i_traj], x[i_traj][:,0], parameters[i_traj], **integration_args)
        else:
            t_traj, x_traj, success_traj = time_stepper.advect(t[i_traj], x[i_traj][:,0], **integration_args)
        x_predict.append(x_traj)
        t_predict.append(t_traj)

        if not success_traj:
            print("Time stepping failed for trajectory number %s" %(i_traj+1))

    return t_predict, x_predict

