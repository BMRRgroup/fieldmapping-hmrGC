import pytest
import copy
import numpy as np

from hmrGC.dixon_imaging.multi_echo import MultiEcho

tol = 10


def test_multi_echo_init(signal, mask, params):
    params['TE_s'] = params['TE_s'][:3]
    del params['siliconePeak_ppm']

    g = MultiEcho(signal, mask, params)
    assert g._signal_masked.shape == (20*20-3, 3)
    _signal_masked = signal[mask]
    assert np.array_equal(g._signal_masked, _signal_masked)
    assert g._num_voxel == 20*20
    assert g._mask_reshaped.shape[0] == g._num_voxel

    fieldmap = np.zeros_like(mask)
    g.fieldmap = fieldmap
    assert g._fieldmap_masked.shape[0] == 20*20-3

    r2starmap = np.zeros_like(mask)
    g.r2starmap = r2starmap
    assert g._r2starmap_masked.shape[0] == 20*20-3

    assert np.allclose(g.range_fm_ppm, np.array([-6.52520032,  6.52520032]))
    assert g.sampling_stepsize_fm == 2
    assert g.sampling_stepsize_r2s == 2
    assert g.range_r2s == [0, 500]
    assert g._num_fm == np.ceil(np.sum(np.abs(g.range_fm)) / g.sampling_stepsize_fm + 1)
    assert g._num_r2s == np.ceil(np.sum(g.range_r2s) / g.sampling_stepsize_r2s + 1)
    assert g._gridspacing_fm == 1.998401278976819


def test_multi_echo_methods(signal, mask, params):
    params['TE_s'] = params['TE_s'][:3]
    del params['siliconePeak_ppm']

    g = MultiEcho(signal, mask, params)
    assert len(g.methods['single-res']) == 3
    assert len(g.methods['multi-res']) == 5
    assert len(g.methods['breast']) == 8


def test_multi_echo_set_phi_matrix(signal_water_fat_silicone, mask, params):
    g = MultiEcho(signal_water_fat_silicone, mask, params)
    opt_default = copy.deepcopy(g.options)
    assert g.options['signal_model'] == 'WFS'
    assert g.phi.shape == (4, 3)
    assert np.sum(g.phi[:, 0] - 1) == 0
    g.options['signal_model'] = 'WF'
    g.set_phi_matrix()
    assert g.params['signal_model'] == 'WFS'
    assert g.phi.shape == (4, 2)
    assert np.sum(g.phi[:, 0] - 1) == 0
    g.options['signal_model'] = 'W'
    g.set_phi_matrix()
    assert g.params['signal_model'] == 'WFS'
    assert g.phi.shape == (4, 1)
    assert np.sum(g.phi[:, 0] - 1) == 0
    g.set_options_default()
    for key in opt_default.keys():
        assert g.options[key] == opt_default[key]


def test_multi_echo_get_residual_fm(signal, mask, params, test_gpu):
    if test_gpu:
        params['TE_s'] = params['TE_s'][:3]
        del params['siliconePeak_ppm']

        g = MultiEcho(signal, mask, params)
        g.range_r2s = params['range_r2s']
        g.sampling_stepsize_fm = params['sampling_stepsize_fm']
        g.sampling_stepsize_r2s = params['sampling_stepsize_r2s']
        residual = g.get_residual_fm()
        assert residual.shape[:3] == signal.shape[:3]
        psis = np.linspace(g.range_fm[0],
                           g.range_fm[1],
                           g._num_fm)

        r1 = residual[0, -1, 0, :][np.round(psis) == 50-8]
        r2 = residual[0, -1, 0, :][np.round(psis) == 50]
        r3 = residual[0, -1, 0, :][np.round(psis) == 50+8]
        assert r2 < r1
        assert r2 < r3
        assert r2 < 1e-1


def test_multi_echo_set_fieldmap(signal, mask, params, fieldmap, test_gpu):
    if test_gpu:
        params['TE_s'] = params['TE_s'][:3]
        del params['siliconePeak_ppm']

        g = MultiEcho(signal, mask, params)
        g.range_r2s = params['range_r2s']
        g.sampling_stepsize_fm = params['sampling_stepsize_fm']
        g.sampling_stepsize_r2s = params['sampling_stepsize_r2s']

        minima_arrays = g._get_minima_arrays()
        assert minima_arrays['voxel_id'].shape == minima_arrays['minima_ind'].shape \
            == minima_arrays['cost'].shape
        voxel_id_unique = np.unique(minima_arrays['voxel_id'])
        assert np.array_equal(voxel_id_unique, np.arange(20*20-3))

        g.set_fieldmap()
        assert np.abs(g.maxflow - 77822) <= tol
        assert np.mean(np.abs(g.fieldmap-fieldmap)[mask]) < 3*params['sampling_stepsize_fm']


def test_multi_echo_perform(signal_water_fat_silicone, signal, mask, params,
                            fieldmap, test_gpu):
    if test_gpu:
        g = MultiEcho(signal_water_fat_silicone, mask, params)
        g.params['voxelSize_mm'] = [2, 2, 2]
        g.range_r2s = params['range_r2s']
        g.sampling_stepsize_fm = params['sampling_stepsize_fm']
        g.sampling_stepsize_r2s = params['sampling_stepsize_r2s']
        g.perform('breast')
        assert np.abs(g.maxflow - 127558) <= tol
        assert np.mean(np.abs(g.fieldmap-fieldmap)[mask]) < 3*params['sampling_stepsize_fm']

        params['TE_s'] = params['TE_s'][:3]
        del params['siliconePeak_ppm']
        del params['signal_model']

        g = MultiEcho(signal, mask, params)
        g.params['voxelSize_mm'] = [2, 2, 2]
        g.range_r2s = params['range_r2s']
        g.sampling_stepsize_fm = params['sampling_stepsize_fm']
        g.sampling_stepsize_r2s = params['sampling_stepsize_r2s']
        g.perform('multi-res')
        assert g.options['signal_model'] == g.params['signal_model']
        assert np.abs(g.maxflow - 80396) <= tol
        assert np.mean(np.abs(g.fieldmap-fieldmap)[mask]) < 3*params['sampling_stepsize_fm']

        g = MultiEcho(signal, mask, params)
        g.params['voxelSize_mm'] = [2, 2, 2]
        g.range_r2s = params['range_r2s']
        g.sampling_stepsize_fm = params['sampling_stepsize_fm']
        g.sampling_stepsize_r2s = params['sampling_stepsize_r2s']
        g.perform('single-res')
        assert np.abs(g.maxflow - 77822) <= tol
        assert np.mean(np.abs(g.fieldmap-fieldmap)[mask]) < 3*params['sampling_stepsize_fm']


def test_multi_echo_set_fieldmap_with_prior(signal, mask, params, fieldmap,
                                            test_gpu):
    if test_gpu:
        params['TE_s'] = params['TE_s'][:3]
        del params['siliconePeak_ppm']

        g = MultiEcho(signal, mask, params)
        g.range_r2s = params['range_r2s']
        g.sampling_stepsize_fm = params['sampling_stepsize_fm']
        g.sampling_stepsize_r2s = params['sampling_stepsize_r2s']
        g._prior.append(fieldmap)
        g.options['prior'] = {}
        g.options['prior']['layer_for_range'] = 0
        g.options['prior']['layer_for_insert'] = [0]
        g.options['prior']['neighborhood_for_range'] = np.array([150/g.params['centerFreq_Hz']*1e6,
                                                                 150/g.params['centerFreq_Hz']*1e6])
        g.options['prior']['neighborhood_for_insert'] = np.array([10/g.params['centerFreq_Hz']*1e6,
                                                                  10/g.params['centerFreq_Hz']*1e6])

        minima_arrays = g._get_minima_arrays()
        assert minima_arrays['voxel_id'].shape == minima_arrays['minima_ind'].shape \
            == minima_arrays['cost'].shape
        voxel_id_unique = np.unique(minima_arrays['voxel_id'])
        assert np.array_equal(voxel_id_unique, np.arange(20*20-3))

        g.set_fieldmap()
        assert np.abs(g.maxflow - 79094) <= tol
        assert np.mean(np.abs(g.fieldmap-fieldmap)[mask]) < 2*params['sampling_stepsize_fm']


def test_multi_echo_get_nodes_arrays(signal, mask, params):
    params['TE_s'] = params['TE_s'][:3]
    del params['siliconePeak_ppm']

    g = MultiEcho(signal, mask, params)
    minima_arrays = g._get_minima_arrays()
    nodes_arrays = g._get_nodes_arrays(minima_arrays)
    intra_column_mask = nodes_arrays['intra_column_mask']
    intra_column_cost = nodes_arrays['intra_column_cost']
    inter_column_cost = nodes_arrays['inter_column_cost']
    assert intra_column_mask.shape == intra_column_cost.shape == \
        inter_column_cost.shape
    assert np.sum(np.abs(intra_column_cost[~intra_column_mask])) == 0
    assert np.sum(np.abs(inter_column_cost[~intra_column_mask])) == 0


def test_multi_echo_get_range_from_arr(signal, mask, params):
    params['TE_s'] = params['TE_s'][:3]
    del params['siliconePeak_ppm']

    g = MultiEcho(signal, mask, params)
    g.fieldmap = np.ones_like(mask, dtype=np.float32)
    g._fieldmap_original = np.ones_like(mask, dtype=np.float32)
    g.options['prior'] = {}
    g.options['prior']['neighborhood_for_range'] = [2.5, 2.5]
    g._prior.append(g.fieldmap)
    g.range_fm = g._get_range_from_arr(g._prior[0][g.mask])
    assert g.range_fm[0] == np.around(1-2.5*g.params['centerFreq_Hz']*1e-6-5*g._gridspacing_fm)
    assert g.range_fm[1] == np.around(1+2.5*g.params['centerFreq_Hz']*1e-6+5*g._gridspacing_fm)


def test_multi_echo_set_r2starmap(signal, mask, params, fieldmap, test_gpu):
    if test_gpu:
        params['TE_s'] = params['TE_s'][:3]
        del params['siliconePeak_ppm']

        g = MultiEcho(signal, mask, params)
        g.fieldmap = fieldmap
        g.set_r2starmap()
        assert g.r2starmap.shape == g.fieldmap.shape
        assert np.mean(g._r2starmap_masked-40) < 4*params['sampling_stepsize_r2s']


def test_multi_echo_set_images(signal_water_fat_silicone, mask, params, fieldmap, r2starmap,
                               test_gpu):
    if test_gpu:
        fatfraction_percent = np.linspace(0, 100, 20)
        siliconefraction_percent = np.linspace(0, 100, 20)
        zeros_empty_dim = np.zeros(1)
        xx, yy, zz = np.meshgrid(fatfraction_percent, siliconefraction_percent, zeros_empty_dim)
        water_params = (100-xx-yy)/100
        water = np.where(xx + yy <= 100, water_params, water_params[::-1,::-1,:])
        xx_params = xx/100
        yy_params = yy/100
        fat = np.where(xx + yy <= 100, xx_params, yy_params[::-1, :, :])
        silicone = np.where(xx + yy <= 100, yy_params, xx_params[:, ::-1, :])

        g = MultiEcho(signal_water_fat_silicone, mask, params)
        g.fieldmap = fieldmap
        g.set_images()
        assert g.images['water'].shape == (20, 20, 1)
        assert g.images['fat'].shape == (20, 20, 1)
        assert g.images['silicone'].shape == (20, 20, 1)
        assert np.mean(np.abs(g.images['water'] - water)) < 1
        assert np.mean(np.abs(g.images['fat'] - water)) < 1
        assert np.mean(np.abs(g.images['silicone'] - water)) < 1

        g.r2starmap = r2starmap
        g.set_images()
        assert g.images['water'].shape == (20, 20, 1)
        assert g.images['fat'].shape == (20, 20, 1)
        assert g.images['silicone'].shape == (20, 20, 1)
        assert np.mean(np.abs(g.images['water'] - water)) < 1
        assert np.mean(np.abs(g.images['fat'] - water)) < 1
        assert np.mean(np.abs(g.images['silicone'] - water)) < 1
