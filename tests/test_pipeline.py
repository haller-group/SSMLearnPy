import numpy as np
from ssmlearnpy.utils.data import SSMDataAttribute, SSMData
from ssmlearnpy.main.main import SSMLearn
from copy import deepcopy
import ipdb

def test_pipeline():
    t = np.linspace(0, 100, 10000)
    signal = np.exp(-0.1*t)*np.sin(t)

    input_data = SSMDataAttribute(
        data=[signal.reshape(1,-1)],
        time=[t],
    )

    ssm = SSMLearn(
        data = SSMData(inputs=input_data),
    )

    ssm.fit()
    test_data = deepcopy(ssm.data)
    ssm.predict()

    assert np.all(ssm.data.embedded.data[0] == test_data.embedded.data[0]), "Prediction should not mutate the data"
    
    assert ssm.config.ssm_dim == 2, "SSM dimension should be calculated to be 2"
    assert ssm.config.coordinates_embeddings_args.shift_steps is not None, "Shift steps should be calculated by the model"

    assert np.allclose(ssm.data.advected_normal_coordinates.reconstructed_embedding[0], ssm.data.embedded.data[0], atol=2e-3), \
        "Reconstructed embedding should match the original embedded data"

if __name__ == "__main__":
    test_pipeline()