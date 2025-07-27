from scipy.integrate import solve_ivp
from ssmlearnpy.utils import ridge
import numpy as np
from ssmlearnpy.geometry.dimensionality_reduction import reduce_dimensions, LinearChart
from ssmlearnpy.utils.data import SSMDataAttribute, SSMData
from ssmlearnpy.utils.config import SSMConfig, BaseRegressionConfig

from ssmlearnpy import SSMLearn

def test_LinearChart():
    lc = LinearChart(2)
    data = np.random.rand(3,10)
    lc.fit(data)
    data2 = np.random.rand(3, 10000)
    reduced = lc.predict(data2)
    assert reduced.shape == (2, 10000)

def test_geometry():

    input_data = SSMDataAttribute(
        data=[np.random.rand(3, 1000)],
        time=[np.linspace(0, 10, 1000)],
    )
    ssm_dim = 2

    config = SSMConfig(
        ssm_dim=ssm_dim,
        dynamics_type='flow',
    )

    ssm = SSMLearn(
    data = SSMData(inputs = input_data),
    config=config,
    )

    ssm.fit_geometry(regression_args= BaseRegressionConfig(poly_degree=1))
    
    embedded_dim = ssm.data.embedded.data[0].shape[0]

    
    encoded = ssm.encode(np.random.rand(embedded_dim, 100))
    assert encoded.shape == (ssm_dim, 100)
    decoded = ssm.decode(encoded)
    assert decoded.shape == (embedded_dim, 100)

def test_geometry_explicit():
    input_data = SSMDataAttribute(
        data=[np.random.rand(3, 1000)],
        time=[np.linspace(0, 10, 1000)],
    )

    ssm_dim = 2

    config = SSMConfig(
        ssm_dim=ssm_dim,
        dynamics_type='flow',
    )

    ssm = SSMLearn(
    data = SSMData(inputs = input_data),
    config=config,
    )

    ssm.fit_encoder(method = "linearchart")
    ssm.fit_decoder(regression_args = BaseRegressionConfig(poly_degree=2))

    embedded_dim = ssm.data.embedded.data[0].shape[0]

    encoded = ssm.encode(np.random.rand(embedded_dim, 100))
    print(encoded.shape)
    assert encoded.shape == (ssm_dim, 100)
    decoded = ssm.decode(encoded)
    assert decoded.shape == (embedded_dim, 100)


if __name__ == '__main__':
    test_LinearChart()
    test_geometry()
    test_geometry_explicit()