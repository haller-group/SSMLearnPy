from scipy.integrate import solve_ivp
from ssmlearnpy.utils import ridge
import numpy as np
def vectorfield(t,x):
    # duffing oscillator with fixed points at x=0 and x=+/-1.
    # x[0] is position, x[1] is velocity
    return np.array([x[1], -x[0]-x[0]**3 - 0.05*x[1]])

def test_constrained_regression():
    # Test constrained ridge regression 

    # Generate data
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    ic_0 = np.array([0.1, 0])
    ic_1 = np.array([-0.1, 0])
    sol_0 = solve_ivp(vectorfield, [t[0], t[-1]], ic_0, t_eval=t)
    sol_1 = solve_ivp(vectorfield, [t[0], t[-1]], ic_1, t_eval=t)
    x = np.vstack([sol_0.y, sol_1.y])
    y = np.vstack([np.diff(sol_0.y)/dt, np.diff(sol_1.y)/dt])

    # Fit model
    
    # Get latent coordinates
    ssm = SSMLearn(



    
    