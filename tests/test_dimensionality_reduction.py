from scipy.integrate import solve_ivp
from ssmlearnpy.utils import ridge
import numpy as np
from ssmlearnpy.geometry.dimensionality_reduction import reduce_dimensions, LinearChart

from ssmlearnpy import SSMLearn

def test_LinearChart():
    lc = LinearChart(2)
    data = np.random.rand(3,10)
    lc.fit(data)
    data2 = np.random.rand(3, 10000)
    reduced = lc.predict(data2)
    assert reduced.shape == (2, 10000)

def test_encoder():
    ssm = SSMLearn(
    t = [np.linspace(0, 10, 1000)], 
    x = [np.random.rand(3, 1000)], 
    derive_embdedding=False,
    ssm_dim=2, 
    dynamics_type = 'flow',
    )
    ssm.get_reduced_coordinates()
    
    ssm.get_parametrization(poly_degree = 1)
    
    encoded = ssm.encode(np.random.rand(3, 100))
    assert encoded.shape == (2, 100)
    decoded = ssm.decode(encoded)
    assert decoded.shape == (3, 100)

def test_encoder_implicit():
    ssm = SSMLearn(
    t = [np.linspace(0, 10, 1000)], 
    x = [np.random.rand(3, 1000)], 
    derive_embdedding=False,
    ssm_dim=2, 
    dynamics_type = 'flow',
    )
    ssm.get_parametrization(poly_degree = 2)
    
    encoded = ssm.encode(np.random.rand(3, 100))
    print(encoded.shape)
    assert encoded.shape == (2, 100)
    decoded = ssm.decode(encoded)
    assert decoded.shape == (3, 100)


if __name__ == '__main__':
    test_LinearChart()
    test_encoder()
    test_encoder_implicit()