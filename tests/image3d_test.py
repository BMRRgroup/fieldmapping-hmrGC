import pytest
import copy
import numpy as np

from hmrGC.image3d import Image3D, pad_array3d
from hmrGC.helper import move2cpu


def test_image3d_set_options_default(signal, mask, params):
    g = Image3D(signal, mask, params, '../tests/image3d_test.json')
    assert g.options['test_param'] == False
    assert g.options['isotropic_weighting_factor'] == [1, 1, 1]


def test_image3d_set_methods_default(signal, mask, params):
    g = Image3D(signal, mask, params, '../tests/image3d_test.json')
    assert '1_layer' in g.methods.keys()
    assert '2_layer_unwrapping' in g.methods.keys()
    assert g.methods['1_layer'][0]['test_param'] == True
    assert g.methods['2_layer_unwrapping'][0]['test_param'] == True
    g.methods['1_layer'][0]['test_param'] = False
    g.methods['2_layer_unwrapping'][0]['test_param'] = False
    assert g.methods['1_layer'][0]['test_param'] == False
    assert g.methods['2_layer_unwrapping'][0]['test_param'] == False
    g.set_methods_default(method_name='2_layer_unwrapping')
    assert g.methods['1_layer'][0]['test_param'] == False
    assert g.methods['2_layer_unwrapping'][0]['test_param'] == True


def test_image3d_set_mask_and_signal_original(signal, mask, params):
    g = Image3D(signal, mask, params, '../tests/image3d_test.json')
    np.testing.assert_array_equal(g.mask, g._mask_original)
    np.testing.assert_array_equal(g.signal, g._signal_original)
    g.signal = np.zeros_like(g.signal)
    assert np.any(np.not_equal(g.signal, g._signal_original))
    if 'voxelSize_mm' in params:
        np.testing.assert_array_equal(params['voxelSize_mm'],
                                      g._voxel_size_original)


def test_image3d_change_resolution(signal, mask, params, test_gpu):
    if test_gpu:
        g = Image3D(signal, mask, params, '../tests/image3d_test.json')
        g.options['voxelSize_mm'] = np.multiply(g.params['voxelSize_mm'],
                                                np.array([5/3, 5/3, 5/3]))
        g.change_resolution()
        assert np.array_equal(g._signal_original, signal)
        assert np.array_equal(g._mask_original, mask)
        assert g.signal.shape == (12, 12, 1, 3)
        assert g.mask.shape == (12, 12, 1)
        if 'voxelSize_mm' in params:
            assert np.array_equal(g.params['voxelSize_mm'],
                                np.multiply(params['voxelSize_mm'],
                                            np.array([5/3, 5/3, 1])))

        g.test_map = np.ones((12, 12, 1), dtype=np.float32)
        del g.options['voxelSize_mm']
        g.change_resolution()
        assert g.signal.shape == (20, 20, 1, 3)
        assert g.mask.shape == (20, 20, 1)
        assert g.test_map.shape == (20, 20, 1)
        assert np.array_equal(g.signal, signal)
        assert np.array_equal(g.mask, mask)

        g = Image3D(signal, mask, params, '../tests/image3d_test.json')
        g.options['voxelSize_mm'] = np.multiply(g.params['voxelSize_mm'],
                                                np.array([5/2, 5/2, 5/2]))
        g.change_resolution()
        g.test_map = np.ones_like(g.mask, dtype=np.float32)
        g.change_resolution()
        assert g._prior[-1].shape == (8, 8, 1)
        assert g._prior_original[-1].shape == (20, 20, 1)
        assert g.mask.shape == (8, 8, 1)
        assert g.signal.shape == (8, 8, 1, 3)
        if 'voxelSize_mm' in params:
            assert np.array_equal(g.params['voxelSize_mm'],
                                np.multiply(params['voxelSize_mm'],
                                            np.array([5/2, 5/2, 1])))

        g = Image3D(signal, mask, params, '../tests/image3d_test.json')
        g.change_resolution()
        g.options['voxelSize_mm'] = np.multiply(g.params['voxelSize_mm'],
                                                np.array([5/2, 5/2, 5/2]))
        g.test_map = np.ones_like(g.mask, dtype=np.float32)
        g.change_resolution()
        assert g._prior[-1].shape == (8, 8, 1)
        assert g._prior_original[-1].shape == (20, 20, 1)
        assert g.mask.shape == (8, 8, 1)
        assert g.signal.shape == (8, 8, 1, 3)
        if 'voxelSize_mm' in params:
            assert np.array_equal(g.params['voxelSize_mm'],
                                np.multiply(params['voxelSize_mm'],
                                            np.array([5/2, 5/2, 1])))


def test_image3d_get_prior(signal, mask, params, test_gpu):
    if test_gpu:
        g = Image3D(signal, mask, params, '../tests/image3d_test.json')
        signal = np.abs(signal)
        g._prior.append(signal[...,0])
        g._prior.append(np.zeros_like(signal[...,0]))
        g.options['prior'] = {}
        g.options['prior']['layer_for_range'] = 0
        g.options['prior']['layer_for_insert'] = [1, 0]
        prior_range, prior_insert = g._get_prior()
        np.testing.assert_array_equal(signal[...,0][mask], move2cpu(prior_range))
        np.testing.assert_array_equal(signal[...,0][mask], move2cpu(prior_insert[:,1]))
        np.testing.assert_array_equal(np.zeros_like(signal[...,0])[mask], move2cpu(prior_insert[:,0]))


def test_image3d_trim_and_pad(signal, mask, params, test_gpu):
    if test_gpu:
        arr = signal
        arr_shape = np.array(arr.shape)
        padsize = [5, 5, 5]
        arr_new_shape = np.append((arr_shape[:3]+2*np.array(padsize)),
                                     arr_shape[-1]).astype(np.int32)
        arr_new = np.zeros(arr_new_shape, dtype=np.complex64)
        for i in range(arr.shape[-1]):
            arr_new[..., i] = pad_array3d(arr[..., i], padsize)
        arr_mask = pad_array3d(mask, padsize)

        g = Image3D(arr_new, arr_mask, params, '../tests/image3d_test.json')
        g.trim_and_pad([0, 0, 0])
        np.testing.assert_almost_equal(g.signal[mask], signal[mask], decimal=5)
        np.testing.assert_array_equal(g.mask, mask)
        g.revert_trim_and_pad()
        np.testing.assert_almost_equal(g.signal[arr_mask], arr_new[arr_mask], decimal=5)
        np.testing.assert_array_equal(g.mask, arr_mask)
        g.trim_and_pad([5, 5, 5])
        np.testing.assert_almost_equal(g.signal[arr_mask], arr_new[arr_mask], decimal=5)
        np.testing.assert_array_equal(g.mask, arr_mask)
        g.revert_trim_and_pad()
        np.testing.assert_almost_equal(g.signal[arr_mask], arr_new[arr_mask], decimal=5)
        np.testing.assert_array_equal(g.mask, arr_mask)
        g.test_map = np.zeros_like(arr_new[..., 0])
        g.trim_and_pad([0, 0, 0])
        np.testing.assert_almost_equal(g.signal[mask], signal[mask], decimal=5)
        np.testing.assert_array_equal(g.mask, mask)
        np.testing.assert_array_equal(g.test_map, np.zeros_like(signal[..., 0]))
        g.revert_trim_and_pad()
        np.testing.assert_almost_equal(g.signal[arr_mask], arr_new[arr_mask], decimal=5)
        np.testing.assert_array_equal(g.mask, arr_mask)
        np.testing.assert_array_equal(g.test_map, np.zeros_like(arr_new[..., 0]))

def test_image3d_phase_inpaint(signal, mask, params):
    mask = np.ones_like(mask)
    signal = np.ones_like(signal)
    signal *= np.exp(1j*np.pi/2)
    mask[2, 2, :] = 0
    signal[2, 2, :, :] = 1
    g = Image3D(signal, mask, params, '../tests/image3d_test.json')
    g.phase_inpaint()
    assert np.sum(np.angle(g.signal[2,2,:,:])-np.pi/2) < 0.1
