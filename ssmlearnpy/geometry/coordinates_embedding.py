import numpy as np

import logging

logger = logging.getLogger("coordinates_embedding")


def coordinates_embedding(
        t: list, 
        x: list, 
        imdim: int=None, 
        over_embedding: int=0,
        force_embedding: bool=False,
        time_stepping: int=1,
        shift_steps: int=1
    ):
    """
    Returns the n-dim. time series x into a time series of properly embedded
    coordinate system y of dimension p. Optional inputs to be specified as
    'field_name','field value'
        
    INPUT
    t - list of time vectors
    x - list of observed trajectories 
    imdim - dimension of the invariant manifold to learn
        
    OPTIONAL INPUT
    over_embedding  - augment the minimal embedding dimension with a number of
                     time delayed measurements, default 0
    force_embedding - force the embedding in the states of x, default false
    time_stepping   - time stepping in the time series, default 1
    shift_steps     - number of timesteps passed between components (but 
                     subsequent measurements are kept intact), default 1
        If varargin is set to an integer value, it it set as OverEmbedding
        
    OUTPUT
    y_data - cell array of dimension (N_traj,2) where the first column contains
        time instances (1 x mi each) and the second column the trajectories
        (p x mi each)
    opts_embdedding - options containing the embedding information

    """
    if not imdim:
        raise RuntimeError("imdim not specified for coordinates embedding")
    n_observables = x[0].shape[0] 
    n_n = int(np.ceil( (2*imdim + 1)/n_observables) + over_embedding)

    # Construct embedding coordinate system
    if n_n > 1 and force_embedding != 1:
        p = n_n * n_observables
        # Augment embdedding dimension with time delays
        if n_observables == 1:
            logger.info((
                f'The {str(p)} embedding coordinates consist of the ' +
                f'measured state and its {str(n_n-1)} time-delayed measurements.'
            ))
        else:
            logger.info((
                f'The {str(p)} embedding coordinates consist of the {str(n_observables)} ' +
                f'measured states and their {str(n_n-1)} time-delayed measurements.'
            ))
        t_y = []
        y = []
        for i_traj in range(len(x)):
            t_i = t[i_traj]
            x_i = x[i_traj]

            subsample = np.arange(start=0, stop=len(t_i), step=time_stepping)

            y_i = x_i[:, subsample]
            y_base = x_i[:, subsample]

            for i_rep in range(1, n_n):
                y_i = np.concatenate(
                    (
                        y_i,
                        np.roll(y_base, -i_rep) 
                    )   
                )
            
            y.append(
                y_i[:, :-n_n+1]
            )
            t_y.append(
                t_i[
                    subsample[:-n_n+1]
                ]
            )

    else:
        p = n_observables

        if time_stepping > 1:
            logger.info('The embedding coordinates consist of the measured states.')
            t_y = []
            y = []
            for i_traj in range(len(x)):
                t_i = t[i_traj]
                x_i = x[i_traj]
                subsample = np.arange(start=0, stop=len(t_i), step=time_stepping)
                t_y.append(t_i[subsample])
                y.append(x_i[:, subsample])

        else:
            t_y = t
            y = x

    opts_embdedding = {
        'imdim' : imdim,
        'over_embedding': over_embedding,
        'force_embedding': force_embedding,
        'time_stepping' : time_stepping,
        'shift_steps' : shift_steps,
        'embedding_space_dim': p
    }
    

    return t_y, y, opts_embdedding