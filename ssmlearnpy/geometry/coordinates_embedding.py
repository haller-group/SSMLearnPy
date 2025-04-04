import numpy as np

import logging

logger = logging.getLogger("coordinates_embedding")


def coordinates_embedding(
    t: list,
    x: list,
    imdim: int = None,
    offset: np.ndarray = None,
    over_embedding: int = 0,
    force_embedding: bool = False,
    time_stepping: int = 1,
    shift_steps: int = 1,
):
    """
    Returns the n-dim. time series x into a time series of properly embedded
    coordinate system y of dimension p. Optional inputs to be specified as
    'field_name','field value'

    Parameters:
    t : list of time vectors
    x : list of observed trajectories
    imdim - dimension of the invariant manifold to learn

    offsets (optional): list containing an offset vector for each trajectory. The offset should shift
                        the fixed point to the origin once subtracted. If not provided, the fixed point
                        is assumed to be at the origin.
    over_embedding (optional): augment the minimal embedding dimension with a number of
                     time delayed measurements, default 0
    force_embedding (optional): force the embedding in the states of x, default false
    time_stepping   (optional): time stepping in the time series, default 1
    shift_steps     (optional): number of timesteps passed between components (but
                     subsequent measurements are kept intact), default 1

    Returns:
    t_y : list of time vectors

    y : cell array of dimension (N_traj,2) where the first column contains
        time instances (1 x mi each) and the second column the trajectories
        (p x mi each)
    opts_embdedding : options containing the embedding information, including new offset vector if offset was provided

    """
    if not imdim:
        raise RuntimeError("imdim not specified for coordinates embedding")
    n_observables = x[0].shape[0]
    n_n = int(np.ceil((2 * imdim + 1) / n_observables) + over_embedding)
    if offset is None:
        _offset = np.zeros(n_observables)
    else:
        _offset = offset

    t_y = []
    y = []
    embedded_offset = None

    # Construct embedding coordinate system
    if n_n > 1 and force_embedding != 1:
        p = n_n * n_observables
        # Augment embdedding dimension with time delays
        if n_observables == 1:
            logger.info(
                (
                    f"The {str(p)} embedding coordinates consist of the "
                    + f"measured state and its {str(n_n-1)} time-delayed measurements."
                )
            )
        else:
            logger.info(
                (
                    f"The {str(p)} embedding coordinates consist of the {str(n_observables)} "
                    + f"measured states and their {str(n_n-1)} time-delayed measurements."
                )
            )

        for i_traj in range(len(x)):
            t_i = t[i_traj]
            x_i = x[i_traj]
            # offset = _offsets[i_traj].astype(float)

            subsample = np.arange(start=0, stop=len(t_i), step=time_stepping)

            y_i = x_i[:, subsample]
            y_base = x_i[:, subsample]

            for i_rep in range(1, n_n):
                y_i = np.concatenate((y_i, np.roll(y_base, -i_rep * shift_steps)))

            y.append(y_i[:, : -n_n * shift_steps + 1])
            t_y.append(t_i[subsample[: -n_n * shift_steps + 1]])
        embedded_offset = np.tile(_offset, n_n)

    else:
        p = n_observables

        if time_stepping > 1:
            logger.info("The embedding coordinates consist of the measured states.")
            for i_traj in range(len(x)):
                t_i = t[i_traj]
                x_i = x[i_traj]
                subsample = np.arange(start=0, stop=len(t_i), step=time_stepping)
                t_y.append(t_i[subsample])
                y.append(x_i[:, subsample])

        else:
            t_y = t
            y = x
        embedded_offset = _offset

    # To improve consistency with existing code, we return None rather than zero offset if no offset is provided
    if offset is None:
        embedded_offset = None

    opts_embdedding = {
        "imdim": imdim,
        "over_embedding": over_embedding,
        "force_embedding": force_embedding,
        "time_stepping": time_stepping,
        "shift_steps": shift_steps,
        "embedding_space_dim": p,
        "embedded_offset": embedded_offset,
    }

    return t_y, y, opts_embdedding
