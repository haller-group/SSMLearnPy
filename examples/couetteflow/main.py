from scipy.io import loadmat
from ssmlearnpy import SSMLearn
from ssmlearnpy.reduced_dynamics.advector import advect
import numpy as np
import matplotlib.pyplot as plt



mat = loadmat('dataRe135.mat')
n_traj = 2

t_full = [mat['aData'][i][0].flatten() for i in range(n_traj)]
x_full = [mat['aData'][i][1] for i in range(n_traj)]
x_reduced = [np.sqrt(mat['xData'][i][1]) for i in range(n_traj)]

ini_Time, end_Time = 400, 1400
for i_traj in range(n_traj):
    t_i, x_i, x_r_i = t_full[i_traj], x_full[i_traj], x_reduced[i_traj]
    idx_Ini, idx_End = int(np.sum(t_i<ini_Time)), int(np.sum(t_i<end_Time))
    t_full[i_traj] = t_i[idx_Ini:idx_End]
    x_full[i_traj] = x_i[:,idx_Ini:idx_End]
    x_reduced[i_traj] = x_r_i[:,idx_Ini:idx_End]

ssm = SSMLearn(
    t = t_full, 
    x = x_full, 
    reduced_coordinates = x_reduced,
    ssm_dim=2, 
    dynamics_type = 'map'
)

# print(len(ssm.emb_data))
# print(ssm.emb_data.keys())
# print(ssm.emb_data)
# for i_traj in range(n_traj):
#     plt.plot(ssm.emb_data['time'][i_traj].flatten(), ssm.emb_data['reduced_coordinates'][i_traj][0,:])
# plt.show()

ssm.get_parametrization(poly_degree=2)    
ssm.get_reduced_dynamics(poly_degree=4)
ssm.predict_reduced_dynamics()    

# t_predict, x_predict  = advect(
#     dynamics=ssm.reduced_dynamics.predict, 
#     t=t_full, 
#     x=x_reduced, 
#     type='map'
# )

t_predict = ssm.reduced_dynamics_predictions['time']
x_predict = ssm.reduced_dynamics_predictions['reduced_coordinates']

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

