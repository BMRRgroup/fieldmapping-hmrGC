import numpy as np
import scipy.ndimage as ndi
try:
    import cupyx.scipy.ndimage as cp_ndi
except ModuleNotFoundError:
    import scipy.ndimage as cp_ndi

from hmrGC.helper import ChunkArr, get_arr_chunks, \
    njit_residuals, njit_residuals_prior, xp_function, \
    scale_array2interval, scale_array2interval_along1D, free_mempool, \
    CUDA_OutOfMemory, move2cpu, move2gpu
from hmrGC.dixon_imaging.helper import calculate_pdsf_percent, \
    calculate_pdff_percent
from hmrGC.graph_cut import GraphCut
from hmrGC.constants import RESCALE, GYRO, PPM2HZ
from hmrGC.image3d import Image3D


class MultiEcho(Image3D):
    """
    Water-fat(-silicone) separation using hierarchical
    multi-resolution graph-cuts

    """

    def __init__(self, signal, mask, params):
        """Initialize MultiEcho object

        :param signal: complex array of shape (nx, ny, nz, nTE),
        complex signal array
        :param mask: boolean array of shape (nx, ny, nz), signal mask
        :param params: dict, scan parameters

        """
        # Set signal_model
        if 'signal_model' not in params:
            if 'siliconePeak_ppm' in params and 'FatModel' in params:
                params['signal_model'] = 'WFS'
            elif 'FatModel' in params:
                params['signal_model'] = 'WF'
            else:
                params['signal_model'] = 'W'
        Image3D.__init__(self, signal, mask, params,
                         'dixon_imaging/multi_echo.json')

        # Check input data and dimensions
        assert len(self.params['TE_s']) == self.signal.shape[-1]
        assert self.mask.shape == self.signal.shape[:3]
        if params['signal_model'] == 'W' or params['signal_model'] == 'S':
            assert len(self.params['TE_s']) > 1
        elif params['signal_model'] == 'WF':
            assert len(self.params['TE_s']) > 2
        elif params['signal_model'] == 'WFS':
            assert len(self.params['TE_s']) > 3
        else:
            print('Signal model not implemented!')

        # Set default field-map range
        range_fm_ppm = [-2.5, 2.5]
        if 'period' not in self.params:
            TE_s = self.params['TE_s']
            dTE = np.diff(TE_s)
            if np.sum(np.abs(dTE-dTE[0])) < 1e-5:
                self._params['period'] = np.abs(1/(TE_s[1] - TE_s[0])) * 1e6 \
                    / self.params['centerFreq_Hz']

                range_fm_ppm = [-self.params['period'],
                                self.params['period']]
        self.range_fm_ppm = np.array(range_fm_ppm)
        # Set default R2star range
        self.range_r2s = [0, 500]
        # Set default sampling stepsizes for field-map and R2star
        self.sampling_stepsize_fm = 2
        self.sampling_stepsize_r2s = 2
        # Default: Species images are r2star corrected
        self.r2star_correction = True

    @property
    def fieldmap(self):
        """float array of shape (nx, ny, nz), field-map"""
        return self._fieldmap

    @fieldmap.setter
    def fieldmap(self, val):
        self._fieldmap = val
        if hasattr(self, 'mask'):
            self._fieldmap_masked = val[self.mask]

    @property
    def r2starmap(self):
        """array of shape (nx, ny, nz), r2star-map"""
        return self._r2starmap

    @r2starmap.setter
    def r2starmap(self, val):
        self._r2starmap = val
        if hasattr(self, 'mask'):
            self._r2starmap_masked = val[self.mask]

    @property
    def sampling_stepsize_fm(self):
        """float, sampling stepsize for fieldmap; *default* 2 Hz"""
        return self._sampling_stepsize_fm

    @sampling_stepsize_fm.setter
    def sampling_stepsize_fm(self, val):
        self._sampling_stepsize_fm = val
        if hasattr(self, 'range_fm'):
            self._num_fm = int(np.ceil(np.ceil(np.diff(self.range_fm)) /
                                       self.sampling_stepsize_fm)) + 1
            self._gridspacing_fm = float(np.diff(self.range_fm) /
                                         (self._num_fm - 1))

    @property
    def range_fm(self):
        """array, sampling range for fieldmap"""
        return self._range_fm

    @range_fm.setter
    def range_fm(self, val):
        self._range_fm = val
        self._range_fm_ppm = val/self.params['centerFreq_Hz']*1e6
        if hasattr(self, 'sampling_stepsize_fm'):
            self._num_fm = int(np.ceil(np.ceil(np.diff(self.range_fm)) /
                                       self.sampling_stepsize_fm)) + 1
            self._gridspacing_fm = float(np.diff(self.range_fm) /
                                         (self._num_fm - 1))

    @property
    def range_fm_ppm(self):
        """array, sampling range for fieldmap in ppm"""
        return self._range_fm_ppm

    @range_fm_ppm.setter
    def range_fm_ppm(self, val):
        self.range_fm = val * self.params['centerFreq_Hz'] * 1e-6

    @property
    def sampling_stepsize_r2s(self):
        """float, sampling stepsize for r2starmap; *default* 2 Hz"""
        return self._sampling_stepsize_r2s

    @sampling_stepsize_r2s.setter
    def sampling_stepsize_r2s(self, val):
        self._sampling_stepsize_r2s = val
        if hasattr(self, 'range_r2s'):
            self._num_r2s = int(np.ceil(np.ceil(np.diff(self.range_r2s)) /
                                        self.sampling_stepsize_r2s)) + 1

    @property
    def range_r2s(self):
        """array, sampling range for r2starmap; *default* [0, 500] Hz"""
        return self._range_r2s

    @range_r2s.setter
    def range_r2s(self, val):
        self._range_r2s = val
        if hasattr(self, 'sampling_stepsize_r2s'):
            self._num_r2s = int(np.ceil(np.ceil(np.diff(self.range_r2s)) /
                                        self.sampling_stepsize_r2s)) + 1

    @property
    def r2star_correction(self):
        """bool, True if images should be r2star corrected"""
        return self._r2star_correction

    @r2star_correction.setter
    def r2star_correction(self, val):
        self._r2star_correction = bool(val)

    def set_options_default(self):
        Image3D.set_options_default(self)
        self._options['signal_model'] = self.params['signal_model']
        self.set_phi_matrix()

    @xp_function
    def get_residual_fm(self, xp=np):
        """
        Calculate residuals for the field-map parameter
        """
        p_matrix = move2gpu(self._get_p_matrix('fm'), xp)
        signal = move2gpu(self._signal_masked, xp)

        chunksize = signal.shape[0]
        reduce_gpu_mem = False

        # Split calculation in chunks
        while True:
            try:
                num_chunks = int(np.ceil(signal.shape[0]/chunksize))
                residuals = ChunkArr('float32', reduce_gpu_mem,
                                     shape=(0, p_matrix.shape[0]), xp=xp)

                for i in range(num_chunks):
                    signal_chunk = get_arr_chunks(chunksize, signal, i, num_chunks)

                    # Calculate residuals
                    if xp == np:
                        residuals_chunk = njit_residuals(signal_chunk, p_matrix)
                    else:
                        residuals_chunk = xp.sum(xp.abs(xp.dot(signal_chunk,
                                                                p_matrix)) ** 2,
                                                    axis=-1, dtype=xp.float32)

                    residuals.concatenate(residuals_chunk)
                break
            except CUDA_OutOfMemory:
                reduce_gpu_mem = True
                chunksize = chunksize //2

        residuals = residuals.to_numpy()
        residual_fm = np.zeros((self.mask.shape[0], self.mask.shape[1],
                                self.mask.shape[2], residuals.shape[-1]),
                               dtype=np.float32)
        residual_fm[self.mask, :] = residuals
        return residual_fm

    @xp_function
    def set_fieldmap(self, xp=np):
        """
        Calculate VARPRO minima, solve the graph-cut algorithm and
        compute the field-map based on the mincut partition
        """
        # Get minima arrays and convert to nodes arrays
        minima_arrays = self._get_minima_arrays()
        nodes_arrays = self._get_nodes_arrays(minima_arrays)
        # Perform graph cut
        g = GraphCut(nodes_arrays)
        g.use_gpu = self.use_gpu
        g.intra_column_scaling *= 1/self.options['reg_param']
        g.isotropic_scaling = self.options['isotropic_weighting_factor']
        g.voxel_weighting_intra_column = self.options['noise_weighting_intra_edges']
        g.voxel_weighting_inter_column = self.options['noise_weighting_inter_edges']
        g.set_edges_and_tedges()
        g.mincut()
        self.maxflow = g.maxflow
        map_masked = g.get_map()

        # Reconstruct fieldmap based on the mincut-minima
        node2fm = nodes_arrays['inter_column_cost'].copy()
        node2fm = move2gpu(node2fm * self.params['centerFreq_Hz'] * 1e-6 /
                           PPM2HZ + self.range_fm[0], xp)
        fieldmap_masked = node2fm[xp.arange(len(map_masked)), map_masked]
        fieldmap = xp.zeros_like(move2gpu(self.mask, xp), dtype=xp.float32)
        fieldmap[self.mask] = fieldmap_masked
        self._fieldmap_masked = move2cpu(fieldmap_masked, xp)
        self._fieldmap = move2cpu(xp.reshape(fieldmap, self.mask.shape), xp)

    @xp_function
    def set_r2starmap(self, xp=np):
        """
        Calculate r2starmap based on self.fieldmap
        """
        p_matrix = move2gpu(self._get_p_matrix('r2s'), xp)
        signal = move2gpu(self._signal_masked, xp)
        fieldmap = move2gpu(self._fieldmap_masked, xp)
        te = move2gpu(self.params['TE_s'], xp)

        # Undo effect of fieldmap
        _fieldmap = xp.repeat(fieldmap[:, xp.newaxis], len(te), axis=-1)
        _te = xp.repeat(te[xp.newaxis, :], len(_fieldmap), axis=0)
        decompose_fieldmap = \
            xp.exp(-2.0j * xp.pi * xp.multiply(_fieldmap, _te)).astype(xp.complex64)
        signal = xp.multiply(signal, decompose_fieldmap)

        chunksize = signal.shape[0]
        reduce_gpu_mem = False
        while True:
            try:
                num_chunks = int(np.ceil(signal.shape[0]/chunksize))
                r2starmap = ChunkArr('float32', reduce_gpu_mem, xp=xp)

                for i in range(num_chunks):
                    signal_chunk = get_arr_chunks(chunksize, signal,
                                                  i, num_chunks)

                    # Calculate residuals
                    if xp == np:
                        residuals = njit_residuals(signal_chunk, p_matrix)
                    else:
                        residuals = xp.sum(xp.abs(xp.dot(signal_chunk,
                                                         p_matrix)) ** 2,
                                           axis=-1, dtype=xp.float32)

                    # Reconstruct r2starmap
                    r2s = xp.linspace(self.range_r2s[0], self.range_r2s[1],
                                      self._num_r2s, dtype=xp.float32)
                    r2starmap_chunk = r2s[xp.argmin(residuals, axis = -1)]

                    r2starmap.concatenate(r2starmap_chunk)
                break
            except CUDA_OutOfMemory:
                reduce_gpu_mem = True
                chunksize = chunksize // 2

        self._r2starmap_masked = r2starmap.to_numpy()
        mask = move2gpu(self._mask_reshaped, xp)
        r2starmap = xp.zeros_like(mask, dtype=xp.float32)
        r2starmap[mask] = self._r2starmap_masked
        self._r2starmap = move2cpu(xp.reshape(r2starmap, self.mask.shape), xp)

    @xp_function
    def set_images(self, xp=np):
        """
        Calculate species images based on self.fieldmap (and self.r2starmap)
        """
        fieldmap = move2gpu(self._fieldmap_masked, xp)
        te = move2gpu(self.params['TE_s'], xp)
        signal = move2gpu(self._signal_masked, xp)

        inv_phi = xp.linalg.pinv(move2gpu(self.phi, xp))

        if hasattr(self, '_r2starmap_masked') and self.r2star_correction:
            r2starmap = move2gpu(self._r2starmap_masked, xp)
            _r2starmap = xp.repeat(r2starmap[:, xp.newaxis], len(te), axis=-1)
            _te = xp.repeat(te[xp.newaxis, :], len(_r2starmap), axis=0)
            decompose_r2starmap = \
                xp.exp(-xp.multiply(_r2starmap, _te)).astype(xp.complex64)
            signal = xp.multiply(xp.reciprocal(decompose_r2starmap), signal)

        chunksize = signal.shape[0]
        reduce_gpu_mem = False
        while True:
            try:
                num_chunks = int(np.ceil(signal.shape[0]/chunksize))
                result = ChunkArr('complex64', reduce_gpu_mem, shape=(0, inv_phi.shape[0]), xp=xp)
                for i in range(num_chunks):
                    signal_chunk = get_arr_chunks(chunksize, signal, i, num_chunks)
                    fieldmap_chunk = get_arr_chunks(chunksize, fieldmap,
                                                    i, num_chunks)

                    _fieldmap = xp.repeat(fieldmap_chunk[:, xp.newaxis],
                                          len(te), axis=-1)
                    _te = xp.repeat(te[xp.newaxis, :], len(_fieldmap), axis=0)
                    decompose_fieldmap = \
                        xp.exp(2.0j * xp.pi * xp.multiply(_fieldmap, _te)).astype(xp.complex64)
                    signal_chunk = \
                        xp.multiply(xp.reciprocal(decompose_fieldmap), signal_chunk).transpose()
                    result_chunk = xp.dot(inv_phi, signal_chunk).transpose()

                    result.concatenate(result_chunk)
                result = move2gpu(result.to_numpy(), xp)
                break
            except CUDA_OutOfMemory:
                reduce_gpu_mem = True
                chunksize = chunksize // 2

        mask = move2gpu(self._mask_reshaped, xp)
        images = {}
        images['water'] = xp.zeros_like(mask, dtype=xp.complex64)
        images['water'][mask] = result[:, 0]
        images['water'] = move2cpu(xp.reshape(images['water'], self.mask.shape), xp)
        if self.options['signal_model'] == 'WF' or self.options['signal_model'] == 'WFS':
            images['fat'] = xp.zeros_like(mask, dtype=xp.complex64)
            images['fat'][mask] = result[:, 1]
            images['fat'] = move2cpu(xp.reshape(images['fat'], self.mask.shape), xp)
        if self.options['signal_model'] == 'WFS':
            images['silicone'] = xp.zeros_like(mask, dtype=xp.complex64)
            images['silicone'][mask] = result[:, 2]
            images['silicone'] = move2cpu(xp.reshape(images['silicone'],
                                                     self.mask.shape), xp)

        if self.options['signal_model'] != 'W':
            if hasattr(self, '_r2starmap_masked') and not self.r2star_correction:
                    g = MultiEcho(self.signal, self.mask, self.params)
                    g.options['signal_model'] = self.options['signal_model']
                    g.set_phi_matrix()
                    g.fieldmap = self.fieldmap
                    g.r2starmap = self.r2starmap
                    g.set_images()
                    pdff = g.images['fatFraction_percent']
            else:
                if self.options['signal_model'] == 'WFS':
                    pdff = calculate_pdff_percent(images['water'], images['fat'],
                                                  images['silicone'])
                elif self.options['signal_model'] == 'WF':
                    pdff = calculate_pdff_percent(images['water'], images['fat'])

            images['fatFraction_percent'] = pdff

        self.images = images

    def set_phi_matrix(self):
        """
        Set phi matrix containing the signal model
        """
        if self.options['signal_model'] == 'WF':
            num_species = 2
        elif self.options['signal_model'] == 'WFS':
            num_species = 3
        elif self.options['signal_model'] == 'W' or self.options['signal_model'] == 'S':
            num_species = 1

        te_ = self.params['TE_s']
        field_strength = self.params['fieldStrength_T']
        phi = np.ones((len(te_), num_species)).astype(np.complex64)
        for ii, te in enumerate(te_):
            if self.options['signal_model'] == 'WF' or self.options['signal_model'] == 'WFS':
                freqs_ppm = np.array(self.params['FatModel']['freqs_ppm'])
                rel_amps = np.array(self.params['FatModel']['relAmps'])
                delta_f = [0, *GYRO*freqs_ppm*field_strength]
                fatphasor = 0.0
                for i, amp in enumerate(rel_amps):
                    fatphasor += (amp * np.exp(2.0j * np.pi * delta_f[i+1] * te))
                phi[ii, 1] = fatphasor
            if self.options['signal_model'] == 'WFS':
                delta_f = GYRO * np.array(self.params['siliconePeak_ppm']) * \
                    field_strength
                phi[ii, 2] = np.exp(2.0j * np.pi * delta_f * te)
            if self.options['signal_model'] == 'S':
                delta_f = GYRO * self.params['siliconePeak_ppm'] * \
                    field_strength
                phi[ii, 0] = np.exp(2.0j * np.pi * delta_f * te)
        self.phi = phi

    def _layer_fieldmap(self):
        if self.options['prior'] is not False:
            prior_range = self._prior[self.options['prior']['layer_for_range']]
            self.range_fm = self._get_range_from_arr(prior_range[self.mask])

        self.set_fieldmap()

    def _layer_r2starmap(self):
        if self.options['signal_model'] == 'WFS':
            if 'voxel-dependant_signal_model' in self.options and \
            self.options['voxel-dependant_signal_model']:
                wf = MultiEcho(self.signal, self._mask_wf, self.params)
                wf.options['signal_model'] = 'WF'
                wf.set_phi_matrix()
                wf.fieldmap = self.fieldmap
                wf.set_r2starmap()
                mask_wfs = ~self._mask_wf
                mask_wfs[~self.mask] = False
                wfs = MultiEcho(self.signal, mask_wfs, self.params)
                wfs.fieldmap = self.fieldmap
                wfs.set_r2starmap()

                r2starmap = np.zeros_like(self.mask, dtype=np.float32)
                r2starmap[self._mask_wf] = wf.r2starmap[self._mask_wf]
                r2starmap[~self._mask_wf] = wfs.r2starmap[~self._mask_wf]
                self.r2starmap = r2starmap
            else:
                self.set_r2starmap()
        else:
            self.set_r2starmap()

    def _layer_images(self):
        self.set_images()
        images = self.images
        if self.options['signal_model'] == 'WFS':
            if 'voxel-dependant_signal_model' in self.options and \
            self.options['voxel-dependant_signal_model']:
                wf = MultiEcho(self.signal, self._mask_wf, self.params)
                wf.r2star_correction = self.r2star_correction
                wf.options['signal_model'] = 'WF'
                wf.set_phi_matrix()
                wf.fieldmap = self.fieldmap
                if hasattr(self, 'r2starmap'):
                    wf.r2starmap = self.r2starmap
                wf.set_images()
                images_wf = wf.images

                images['water'][self._mask_wf] = images_wf['water'][self._mask_wf]
                images['fat'][self._mask_wf] = images_wf['fat'][self._mask_wf]
                images['fatFraction_percent'][self._mask_wf] = images_wf['fatFraction_percent'][self._mask_wf]
        self.images = images

    def _add_voxeldependant_signal_models(self):
        if self.params['signal_model'] == 'WFS':
            g = MultiEcho(self.signal, self.mask, self.params)
            g.fieldmap = self.fieldmap
            g.set_images()
            images  = g.images

            threshold = 10
            pdsf = calculate_pdsf_percent(images['water'], images['fat'],
                                          images['silicone'])

            mask_wf = self.mask.copy()
            mask_wf[pdsf > threshold] = False
            self._mask_wf = mask_wf
            pdsf_masked = pdsf[self.mask]
            self._prior_mask = np.zeros_like(pdsf_masked, dtype=np.bool_)
            self._prior_mask[pdsf_masked > threshold] = True
        else:
            self._break_layer = True

    def _set_prior_neighborhood(self):
        layer_name = self.options['name']
        margin = 0.4
        if 'period' in self.params:
            neighborhood = np.array([margin, self._get_spacings('W', 'F')[0] + margin])
            self.options['prior']['neighborhood_for_range'] = neighborhood
        else:
            print('No TE period information. Please specify "neighborhood for_range".')

    @xp_function
    def _check_for_silicone_only_regions(self, xp=np):
        do_silicone_image = self.options['do_silicone_image']
        if self.options['do_silicone_image'] == 'auto' or \
           self.options['modify_prior_insert']:
            self.fieldmap = self._prior[-1].copy()
            self.set_images()
            images = self.images
            pdsf = move2gpu(calculate_pdsf_percent(images['water'], images['fat'],
                                                   images['silicone']), xp)
            silicone_mask = xp.zeros_like(pdsf)
            silicone_mask[pdsf > 60] = 1
            if self.use_gpu:
                xp_ndi = cp_ndi
            else:
                xp_ndi = ndi

            # check for silicone only in the center of the FOV
            silicone_mask_reduced = silicone_mask[:,:,pdsf.shape[-1]//2-10:pdsf.shape[-1]//2+11]
            label_objects, nb_labels = xp_ndi.label(silicone_mask_reduced)

            sizes = xp.bincount(label_objects.ravel())[1:] / np.sum(self.mask[:,:,pdsf.shape[-1]//2-10:pdsf.shape[-1]//2+11])
            if self.options['do_silicone_image'] == 'auto':
                if len(sizes) > 0:
                    if np.max(sizes) > 1e-2:
                        do_silicone_image = True
                        print('Silicone implants detected...', end=" ", flush=True)
                    else:
                        do_silicone_image = False
                else:
                    do_silicone_image = False

            if self.options['modify_prior_insert'] and do_silicone_image:
                # check for silicone only in the center of the FOV
                label_objects, nb_labels = xp_ndi.label(silicone_mask)
                sizes = xp.bincount(label_objects.ravel())[1:] / self._fieldmap_masked.shape[0]
                if len(sizes) > 0:
                    label_objects = move2cpu(label_objects, xp)
                    arg_sizes = move2cpu(xp.argsort(sizes)+1, xp)
                    if sizes[arg_sizes[-1]-1] > 1e-2:
                        self._prior[-2][label_objects == arg_sizes[-1]] = \
                            self._prior[-1][label_objects == arg_sizes[-1]]
                    if len(sizes) > 1 and sizes[arg_sizes[-2]-1] > 1e-2:
                        self._prior[-2][label_objects == arg_sizes[-2]] = \
                            self._prior[-1][label_objects == arg_sizes[-2]]

        if do_silicone_image:
            self._break_layer = False
            self.methods['breast'][4]['prior']['layer_for_range'] = 3
            self.methods['breast'][4]['prior']['layer_for_insert'] = [3]
        else:
            self.params['signal_model'] = 'WF'
            self.options['signal_model'] = 'WF'
            self.fieldmap = self._prior[-2]
            self._break_layer = True
            self.methods['breast'][4]['prior']['layer_for_range'] = 1
            self.methods['breast'][4]['prior']['layer_for_insert'] = [1]

    @xp_function
    def _get_p_matrix(self, residual_type, xp=np):
        """
        Calculate projection matrix for the specified signal model
        """
        te = move2gpu(self.params['TE_s'], xp)

        phi_inv_phi = xp.dot(move2gpu(self.phi, xp), xp.linalg.pinv(move2gpu(self.phi, xp)))

        if residual_type == 'fm':
            fm = xp.linspace(self.range_fm[0], self.range_fm[1], self._num_fm,
                             dtype=xp.float32)
            lamda, inv_lamda = self._get_lamda_fieldmap(te, fm)
            len_param = len(fm)
        elif residual_type == 'r2s':
            r2s = xp.linspace(self.range_r2s[0], self.range_r2s[1], self._num_r2s,
                              dtype=xp.float32)
            lamda, inv_lamda = self._get_lamda_r2starmap(te, r2s)
            len_param = len(r2s)

        p_matrix = xp.repeat(xp.identity(len(te))[xp.newaxis, :, :], len_param, axis=0) - \
            xp.matmul(xp.dot(lamda, phi_inv_phi), inv_lamda)
        return np.ascontiguousarray(move2cpu(p_matrix.transpose(0, 2, 1), xp).astype(np.complex64))

    @xp_function
    def _get_lamda_fieldmap(self, te, fm, xp=np):
        diag = xp.diag(xp.exp(2.0j * xp.pi * te)).astype(xp.complex64)
        diag = xp.repeat(diag[xp.newaxis, :, :], len(fm), axis=0)
        power = xp.repeat(fm[:, xp.newaxis], len(te), axis=-1)
        power = xp.repeat(power[:, :, xp.newaxis], len(te), axis=-1)
        lamda = xp.zeros_like(diag)
        lamda[diag != 0] = xp.power(diag[diag != 0], power[diag != 0])
        return lamda, lamda.conj()

    @xp_function
    def _get_lamda_r2starmap(self, te, r2s, xp=np):
        diag = xp.diag(xp.exp(-te)).astype(xp.complex64)
        diag = xp.repeat(diag[xp.newaxis, :, :], len(r2s), axis=0)
        inv_diag = xp.diag(xp.exp(te)).astype(xp.complex64)
        inv_diag = xp.repeat(inv_diag[xp.newaxis, :, :], len(r2s), axis=0)
        power = xp.repeat(r2s[:, xp.newaxis], len(te), axis=-1)
        power = xp.repeat(power[:, :, xp.newaxis], len(te), axis=-1)
        lamda = xp.zeros_like(diag)
        lamda[diag != 0] = xp.power(diag[diag != 0], power[diag != 0])
        inv_lamda = xp.zeros_like(inv_diag)
        inv_lamda[inv_diag != 0] = xp.power(inv_diag[inv_diag != 0],
                                            power[inv_diag != 0])
        return lamda, inv_lamda

    @xp_function
    def _get_minima_arrays(self, xp=np):
        """
        Calculate residue for the field-map parameter and extract minima
        """
        if self.options['prior'] is False:
            use_prior = False
        else:
            use_prior = True

        p_matrix = move2gpu(self._get_p_matrix('fm'), xp)
        signal = move2gpu(self._signal_masked, xp)

        chunksize = signal.shape[0]
        reduce_gpu_mem = False
        prior, prior_insert = self._get_prior()
        if prior is not None:
            prior -= self.range_fm[0]
            prior = xp.around(prior / self._gridspacing_fm).astype(xp.int32)
        if prior_insert is not None:
            prior_insert -= self.range_fm[0]
            prior_insert = \
                xp.around(prior_insert / self._gridspacing_fm).astype(xp.int32)
        prior_mask = None
        if use_prior:
            if 'prior_mask' in self.options['prior'] and self.options['prior']['prior_mask']:
                prior_mask = self._prior_mask

        while True:
            try:
                num_chunks = int(np.ceil(signal.shape[0]/chunksize))
                voxel_id = ChunkArr('uint32', reduce_gpu_mem, xp=xp)
                minima_ind = ChunkArr('uint32', reduce_gpu_mem, xp=xp)
                cost = ChunkArr('float32', reduce_gpu_mem, xp=xp)
                for i in range(num_chunks):
                    voxel_id, minima_ind, cost = \
                        self.__chunk_func_get_minima_arrays(use_prior, chunksize,
                                                            i, num_chunks, signal,
                                                            p_matrix, voxel_id,
                                                            minima_ind, cost,
                                                            prior, prior_insert,
                                                            prior_mask, xp)
                break
            except CUDA_OutOfMemory:
                reduce_gpu_mem = True
                chunksize = chunksize //2

        minima_arrays = {}
        minima_arrays['voxel_id'] = voxel_id.to_numpy()
        minima_arrays['minima_ind'] = minima_ind.to_numpy()
        minima_arrays['cost'] = cost.to_numpy()
        return minima_arrays

    def __chunk_func_get_minima_arrays(self, use_prior, chunksize, i, num_chunks,
                                       signal, p_matrix, voxel_id, minima_ind,
                                       cost, prior, prior_insert, prior_mask, xp):
        signal_chunk = get_arr_chunks(chunksize, signal, i, num_chunks)
        if use_prior:
            prior_chunk = get_arr_chunks(chunksize, prior, i, num_chunks)
            layer_for_insert = self.options['prior']['layer_for_insert']
            neighborhood_for_range = np.array(self.options['prior']['neighborhood_for_range']) \
                                              * self.params['centerFreq_Hz'] * 1e-6
            neighborhood_for_range = np.around(neighborhood_for_range /
                                               self._gridspacing_fm).astype(np.int32)
            if len(layer_for_insert) > 0:
                neighborhood_for_insert = np.array(self.options['prior']['neighborhood_for_insert']) \
                                                   * self.params['centerFreq_Hz'] * 1e-6
                neighborhood_for_insert = np.around(neighborhood_for_insert /
                                                    self._gridspacing_fm).astype(np.int32)
                prior_insert_chunk = get_arr_chunks(chunksize, prior_insert, i,
                                                    num_chunks)
                if prior_mask is not None:
                    prior_mask_chunk = get_arr_chunks(chunksize, prior_mask, i, num_chunks)
                else:
                    prior_mask_chunk = None

        # Calculate residuals
        if xp == np:
            if use_prior:
                prior_chunk = prior_chunk.astype(np.int32)
                residuals = njit_residuals_prior(signal_chunk, p_matrix, prior_chunk,
                                                 neighborhood_for_range)
            else:
                residuals = njit_residuals(signal_chunk, p_matrix)
        else:
            residuals = xp.sum(xp.abs(xp.dot(signal_chunk, p_matrix)) ** 2, axis=-1,
                               dtype=xp.float32)
            if use_prior:
                ind2_neighbor = xp.repeat(prior_chunk[:, xp.newaxis] -
                                          neighborhood_for_range[0],
                                          np.sum(neighborhood_for_range)+1,
                                          axis=-1)
                ind2_neighbor = ind2_neighbor + \
                    xp.repeat(xp.arange(ind2_neighbor.shape[-1])[xp.newaxis, :],
                              ind2_neighbor.shape[0], axis=0)
                ind1_neighbor = xp.repeat(xp.arange(len(prior_chunk))[:, xp.newaxis],
                                          ind2_neighbor.shape[-1], axis=-1)
                residuals = xp.reshape(residuals[ind1_neighbor.flatten(),
                                                 ind2_neighbor.flatten()],
                                       (signal_chunk.shape[0],
                                        np.sum(neighborhood_for_range)+1))

        # Calculate extrema
        extrema = xp.diff(xp.sign(xp.diff(residuals, axis=-1)), axis=-1)

        # Use only distinct minima
        voxel_id_chunk, maxima_ind_chunk = xp.where(extrema < 0)
        ind2_neighbor = xp.repeat(maxima_ind_chunk[:, xp.newaxis] -
                                  int(np.ceil(self.options['min_min-max_distance_Hz'] / self._gridspacing_fm)),
                                  2 * int(np.ceil(self.options['min_min-max_distance_Hz'] / self._gridspacing_fm)) + 1,
                                  axis=-1)
        ind2_neighbor = ind2_neighbor + xp.repeat(xp.arange(ind2_neighbor.shape[-1])[xp.newaxis, :],
                                                  ind2_neighbor.shape[0], axis=0)
        ind1_neighbor = xp.repeat(voxel_id_chunk[:, xp.newaxis],
                                  ind2_neighbor.shape[-1], axis=-1)
        ind2_neighbor[ind2_neighbor >= extrema.shape[-1]] = extrema.shape[-1] - 1
        ind2_neighbor[ind2_neighbor < 0] = 0
        extrema[ind1_neighbor.flatten(), ind2_neighbor.flatten()] = 0

        if use_prior:
            if len(layer_for_insert) > 0:
                # Look for similar minima in the neighborhood
                if prior_mask_chunk is not None:
                    extrema[prior_mask_chunk, :] =  0
                for j in range(prior_insert_chunk.shape[-1]):
                    voxel_id_chunk, minima_ind_chunk = xp.where(extrema > 0)
                    voxel_id_no_minima_chunk = xp.where(np.sum(extrema > 0, axis=-1) == 0)

                    ind = prior_insert_chunk[:, j] - (prior_chunk -
                                                      neighborhood_for_range[0])
                    nearby_minima = ((minima_ind_chunk-ind[voxel_id_chunk]) \
                        < neighborhood_for_insert[1]) & \
                        (minima_ind_chunk-ind[voxel_id_chunk] > - neighborhood_for_insert[0])
                    nearby_minima[ind[voxel_id_chunk] < 1] = True
                    nearby_minima[ind[voxel_id_chunk] > np.sum(neighborhood_for_range)] = True
                    try:
                        voxel_nearby_minima = xp.unique(voxel_id_chunk[~nearby_minima])
                        extrema[voxel_nearby_minima,
                                ind.astype(xp.int32)[voxel_nearby_minima]] = 1
                    except IndexError:
                        pass
                    # Include voxel with no minima
                    ind[ind < 1] = 0
                    ind[ind > np.sum(neighborhood_for_range)-2] = 0
                    extrema[tuple(voxel_id_no_minima_chunk)[0],
                            ind.astype(xp.int32)[voxel_id_no_minima_chunk]] = 1
                    extrema[:, 0] = 0

        # Find all local minima
        voxel_id_chunk, minima_ind_chunk = xp.where(extrema > 0)
        voxel_id_chunk = voxel_id_chunk.astype(xp.uint32)
        if use_prior:
            cost_chunk = residuals[voxel_id_chunk, minima_ind_chunk + 1]
            indent_minima_ind_chunk = prior_chunk - \
                neighborhood_for_range[0]
            minima_ind_chunk = minima_ind_chunk + \
                indent_minima_ind_chunk[voxel_id_chunk] + 1
            minima_ind_chunk.astype(xp.uint32)
        else:
            minima_ind_chunk = minima_ind_chunk.astype(xp.uint32) + 1
            cost_chunk = residuals[voxel_id_chunk, minima_ind_chunk]

        voxel_id.concatenate(voxel_id_chunk+i*chunksize)
        minima_ind.concatenate(minima_ind_chunk)
        cost.concatenate(cost_chunk)
        return voxel_id, minima_ind, cost

    @xp_function
    def _get_nodes_arrays(self, minima_arrays, xp=np):
        # Find unique indices and calculate node indices
        voxel_id = move2gpu(minima_arrays['voxel_id'], xp)
        minima_ind = move2gpu(minima_arrays['minima_ind'], xp)
        cost = move2gpu(minima_arrays['cost'], xp)
        signal = move2gpu(self._signal_masked.copy(), xp)
        mask = move2gpu(self._mask_reshaped.copy(), xp)

        unique, num_minima_per_voxel = xp.unique(voxel_id, return_counts=True)
        max_minima_per_voxel = int(xp.max(num_minima_per_voxel))
        helper_mask = xp.zeros(signal.shape[0], dtype=xp.bool_)
        helper_mask[unique] = True
        helper_mask2 = mask.copy()
        helper_mask2[mask] = helper_mask
        mask[~helper_mask2] = False
        signal = signal[helper_mask]
        self._signal_masked = move2cpu(signal, xp)
        self.mask = move2cpu(xp.reshape(mask, self.mask.shape), xp)

        # Calculate noise weighting
        MIP = xp.sum(xp.abs(signal), axis=-1, dtype=xp.float32)
        noise_weighting = move2gpu(scale_array2interval(move2cpu(MIP, xp), RESCALE, xp), xp)

        # Create arrays to represent local minima information
        nodes_arrays = {}
        intra_column_mask = (xp.repeat(xp.arange(max_minima_per_voxel,
                                                 dtype=xp.uint32)
                                [:, xp.newaxis], signal.shape[0], axis=-1) - \
                             num_minima_per_voxel < 0).transpose()
        nodes_arrays['intra_column_mask'] = move2cpu(intra_column_mask, xp)
        intra_column_cost = xp.ones((len(num_minima_per_voxel), max_minima_per_voxel),
                                    dtype=xp.float32) * xp.nan
        intra_column_cost[intra_column_mask] = cost
        intra_column_cost[num_minima_per_voxel > 1] = \
            move2gpu(scale_array2interval_along1D(move2cpu(intra_column_cost[num_minima_per_voxel > 1], xp),
                                                  RESCALE, xp), xp)
        intra_column_cost[num_minima_per_voxel == 1] = 1
        intra_column_cost[~intra_column_mask] = 0
        nodes_arrays['intra_column_cost'] = move2cpu(intra_column_cost, xp)
        inter_column_cost = xp.zeros_like(intra_column_cost, dtype=xp.float32)
        # Correct for fieldstrength
        inter_column_cost[intra_column_mask] = minima_ind * self._gridspacing_fm \
            / self.params['centerFreq_Hz'] * 1e6 * PPM2HZ
        inter_column_cost[intra_column_mask] = inter_column_cost[intra_column_mask]
        nodes_arrays['inter_column_cost'] = move2cpu(inter_column_cost, xp)
        nodes_arrays['voxel_mask'] = self.mask
        nodes_arrays['voxel_weighting'] = move2cpu(noise_weighting, xp)
        return nodes_arrays

    @xp_function
    def _get_range_from_arr(self, arr, xp=np):
        xp = np
        arr = move2gpu(arr, xp)
        neighborhood_for_range = np.array(self.options['prior']['neighborhood_for_range']) \
                                          * self.params['centerFreq_Hz'] * 1e-6
        return move2cpu(xp.array([xp.around(xp.min(arr) - \
                                            neighborhood_for_range[0]-5*self._gridspacing_fm),
                                  xp.around(xp.max(arr) + \
                                            neighborhood_for_range[1]+5*self._gridspacing_fm)]), xp)

    def _get_spacings(self, start, stop):
        if start == 'F' and stop == 'W':
            return self._get_spacings(stop, start)[::-1]
        if start == 'S' and stop == 'W':
            return self._get_spacings(stop, start)[::-1]
        if start == 'S' and stop == 'F':
            return self._get_spacings(stop, start)[::-1]

        if start == 'W' and stop == 'F':
            max_freq_ppm = np.abs(self.params['FatModel']['freqs_ppm']
                                  [np.argmax(self.params['FatModel']['relAmps'])])
            spacing_w_f = max_freq_ppm % self.params['period']
            return np.array([self.params['period'] - spacing_w_f, spacing_w_f])
        if start == 'W' and stop == 'S':
            silicone_freq_ppm = np.abs(self.params['siliconePeak_ppm'])
            spacing_w_s = silicone_freq_ppm % self.params['period']
            return np.array([self.params['period'] - spacing_w_s, spacing_w_s])
        if start == 'F' and stop == 'S':
            max_freq_ppm = np.abs(self.params['FatModel']['freqs_ppm']
                                  [np.argmax(self.params['FatModel']['relAmps'])])
            silicone_freq_ppm = np.abs(self.params['siliconePeak_ppm'])
            diff_peak = silicone_freq_ppm - max_freq_ppm
            spacing_f_s = diff_peak % self.params['period']
            return np.array([self.params['period'] - spacing_f_s, spacing_f_s])

    def _set_options_layer(self, options_method, index):
        Image3D._set_options_layer(self, options_method, index)
        self.set_phi_matrix()
