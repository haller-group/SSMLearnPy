import numpy as np
from ssmlearnpy.utils.data import SSMDataAttribute, SSMData
from ssmlearnpy.main.main import SSMLearn
from ssmlearnpy.utils.config import SSMConfig
from copy import deepcopy
import ipdb

def test_pipeline():
    t = np.linspace(0, 100, 10000)
    signal = np.exp(-0.1*t)*np.sin(t)

    input_data = SSMDataAttribute(
        data=[signal.reshape(1,-1)],
        time=[t],
    )

    # Note that config does not have to be passed to SSMLearn, as it will be created automatically.
    # We just do it here for the asserts
    config = SSMConfig()

    assert config.ssm_dim is None, "SSM dimension should be None before fitting"
    assert config.coordinates_embeddings_args.shift_steps is None, "Shift steps should be None before fitting"

    ssm = SSMLearn(
        data = SSMData(inputs=input_data),
        config=config,
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