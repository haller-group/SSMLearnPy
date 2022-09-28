import logging

import numpy as np

def compute_errors(reference,
            prediction,
            metric='NTE'
    ):
    if metric == 'NTE':
        error_fun = nte_error
    elif metric == 'NMTE':
        error_fun = nmte_error
    elif metric == 'TE':
        error_fun = te_error
    elif metric == 'MTE':
        error_fun = mte_error
    else:
        raise NotImplementedError(
            (
                f"{metric} not implemented, please specify an error metric that have "
                f"already been implemented, otherwise raise an issue to the developers"
            )
        )

    errors = []
    for i_elem in range(len(reference)):
        if np.sum(np.isnan(prediction[i_elem])) > 0:
            error_i = np.empty(reference[i_elem].shape, dtype=float), 
            error_i.fill(np.nan)
        else:
            error_i = error_fun(reference[i_elem],prediction[i_elem])
        errors.append(error_i)

    return errors

def nte_error(
    x_reference,
    x_prediction
    ):
    x_norm = np.max(np.sum(np.square(x_reference), axis=0))
    x_error = np.sqrt(np.sum(np.square(x_reference-x_prediction), axis=0)) / x_norm
    return x_error

def nmte_error(
    x_reference,
    x_prediction
    ):
    x_norm = np.max(np.sum(np.square(x_reference), axis=0))
    x_error = np.mean(np.sqrt(np.sum(np.square(x_reference-x_prediction), axis=0))) / x_norm
    return x_error

def te_error(
    x_reference,
    x_prediction
    ):
    x_error = np.sqrt(np.sum(np.square(x_reference-x_prediction), axis=0))
    return x_error

def mte_error(
    x_reference,
    x_prediction
    ):
    x_error = np.mean(np.sqrt(np.sum(np.square(x_reference-x_prediction), axis=0)))
    return x_error