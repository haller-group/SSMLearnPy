from scipy.io import loadmat
from ssmlearnpy import SSMLearn
from ssmlearnpy.reduced_dynamics.reduced_dynamics import advect
import numpy as np
import matplotlib.pyplot as plt



mat = loadmat('dataRe135.mat')
n_traj = 2

t_full = [mat['aData'][i][0].flatten() for i in range(n_traj)]
x_full = [mat['aData'][i][1] for i in range(n_traj)]
x_reduced = [np.sqrt(mat['xData'][i][1]) for i in range(n_traj)]

ssm = SSMLearn(t = t_full, x = x_full, reduced_coordinates = x_reduced,
    im_dim=2 
)

print(len(ssm.emb_data))
print(ssm.emb_data.keys())
print(ssm.emb_data)
# for i_traj in range(n_traj):
#     plt.plot(ssm.emb_data['time'][i_traj].flatten(), ssm.emb_data['reduced_coordinates'][i_traj][0,:])
# plt.show()

ssm.get_parametrization(poly_degree=2)    
ssm.get_reduced_dynamics(poly_degree=4, type = 'map')
    

t_predict, x_predict  = advect(
    dynamics=ssm.reduced_dynamics.predict, 
    t=t_full, 
    x=x_reduced, 
    type='map'
)


for i_traj in range(n_traj):
    plt.plot(
        t_full[i_traj], 
        x_reduced[i_traj][0,:],
        'k'
    )

    plt.plot(
        t_predict[i_traj], 
        x_predict[i_traj][0,:],
        'r--'
    )
plt.show()    

