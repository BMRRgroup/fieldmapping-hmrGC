import pytest
import numpy as np
import numba
import h5py
from hmrGC.constants import GYRO
import os

numba.set_num_threads(4)
#os.environ['NUMBA_DISABLE_JIT'] = '0'

def pytest_addoption(parser):
    """
    Add CLI options to `pytest` to pass those options to the test cases.
    These options are used in `pytest_generate_tests`.
    """
    parser.addoption('--use_gpu')


@pytest.fixture(params=['1', '0'])
def test_gpu(request, pytestconfig):
    """
    Determine if GPU mode should be tested
    """
    use_gpu = pytestconfig.getoption("use_gpu")
    if use_gpu is None:
        use_gpu == 'True'

    if request.param == '1' and use_gpu == 'False':
        return False
    os.environ['BMRR_USE_GPU'] = request.param
    return True


# Water-fat phantom for Dixon MR imaging
@pytest.fixture()
def signal():
    with h5py.File('./tests/data/water_fat_num_phantom.h5', 'r') as f:
        signal = np.array(f['ImDataParams']['signal'], dtype=np.complex64)
    return signal


# Water-fat-silicone phantom for Dixon MR imaging
@pytest.fixture()
def signal_water_fat_silicone():
    with h5py.File('./tests/data/water_fat_silicone_num_phantom.h5', 'r') as f:
        signal = np.array(f['ImDataParams']['signal'], dtype=np.complex64)
    return signal


@pytest.fixture()
def mask():
    mask = np.ones((20, 20, 1), dtype=np.bool_)
    mask[0, :3] = 0
    return mask

# Test for minium required number of params and additional params
@pytest.fixture()
def params(request):
    params = {}
    with h5py.File('./tests/data/water_fat_silicone_num_phantom.h5', 'r') as f:
        params['fieldStrength_T'] = f['ImDataParams'].attrs['fieldStrength_T']
        params['TE_s'] = np.array(f['ImDataParams']['TE_s'])
        params['centerFreq_Hz'] = f['ImDataParams'].attrs['centerFreq_Hz']
        params['voxelSize_mm'] = f['ImDataParams'].attrs['voxelSize_mm']

    params['siliconePeak_ppm'] = -4.9
    params['FatModel'] = {'freqs_ppm': np.array([-3.8, -3.4, -3.1, -2.68,
                                                 -2.46, -1.95, -0.5,  0.49,
                                                 0.59]),
                          'relAmps': np.array([0.08991009, 0.58341658, 0.05994006,
                                               0.08491508, 0.05994006, 0.01498501,
                                               0.03996004, 0.00999001, 0.05694306]),
                          'name': 'Ren marrow'}
    params['sampling_stepsize_fm'] = 1
    params['sampling_stepsize_r2s'] = 5
    #params['range_fm'] = [-500, 500]
    params['range_r2s'] = [0, 500]
    return params


@pytest.fixture()
def fieldmap():
    with h5py.File('./tests/data/water_fat_num_phantom.h5', 'r') as f:
        fieldmap = np.array(f['SimulationParams']['fieldmap_Hz'])
    return fieldmap


@pytest.fixture()
def r2starmap():
    with h5py.File('./tests/data/water_fat_num_phantom.h5', 'r') as f:
        r2starmap = np.array(f['SimulationParams']['R2s_Hz'])
    return r2starmap
