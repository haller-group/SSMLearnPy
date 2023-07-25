import numpy as np

def iterate_map(fun, iterations, x0):
    l_x0 = len(x0)
    x = np.zeros((len(x0), iterations+1))
    x[:,0] = x0
    for iter in range(iterations):
        x[:, iter+1] = fun(x[:, iter].reshape(1,-1))
    return x


