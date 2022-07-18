import numpy as np
import json
import os
import sys
import time
import copy
from scipy.ndimage import gaussian_filter
from hmrGC.helper import xp_function,  move2cpu, move2gpu, \
    free_mempool
from scipy.ndimage import zoom
import cv2 as cv


class Image3D(object):
    """
    3D multi-resolution image processing
    """

    def __init__(self, signal, mask, params, json_path):
        if os.environ.get('BMRR_USE_GPU') == '1':
            self.use_gpu = True
        else:
            self.use_gpu = False

        # Save mask and signal arrays (in 1D and 3D)
        signal = signal.astype(np.complex64)
        mask = mask.astype(np.bool_)
        self.mask = copy.deepcopy(mask)
        self.signal = copy.deepcopy(signal)
        self.params = copy.deepcopy(params)

        # Check for input data
        if 'voxelSize_mm' not in self.params:
            print('Voxel size not in params.')
            print('Multi-res methods are not available.')
            self.params['voxelSize_mm'] = [1, 1, 1]

        d = os.path.dirname(sys.modules[__name__].__file__)
        with open(os.path.join(d, json_path), "r") as read_file:
            self._json_file = json.load(read_file)
        self.set_options_default()
        self.set_methods_default()

        self._set_mask_and_signal_original()

        self._prior = []
        self._prior_original = []
        self.verbose = True

    @property
    def signal(self):
        """array of shape (nx, ny, nz, nTE), signal"""
        return self._signal

    @signal.setter
    def signal(self, val):
        self._signal = val
        if hasattr(self, '_mask_reshaped'):
            self._signal_masked = val[self.mask]

    @property
    def mask(self):
        """boolean array of shape (nx, ny, nz), signal mask"""
        return self._mask

    @mask.setter
    def mask(self, val):
        self._mask = val
        self._num_voxel = val.shape[0]*val.shape[1]*val.shape[2]
        self._mask_reshaped = np.reshape(val, self._num_voxel)

    @property
    def options(self):
        """dict, runtime options"""
        return self._options

    @options.setter
    def options(self, val):
        self._options = val

    @property
    def methods(self):
        return self._methods

    @methods.setter
    def methods(self, val):
        self._methods = val

    @property
    def params(self):
        """dict, scan parameters"""
        return self._params

    @params.setter
    def params(self, val):
        self._params = val

    @property
    def verbose(self):
        """boolean, verbose mode; *default* True"""
        return self._verbose

    @verbose.setter
    def verbose(self, val):
        self._verbose = val

    def set_options_default(self):
        """Set key-value pairs in runtime options (self.options)

        TODO
        """
        self.options = self._json_file['default_layer_options'].copy()
        self._set_isotropic_weighting_default()

    def set_methods_default(self, method_name=None):
        if method_name is None:
            self.methods = copy.deepcopy(self._json_file['methods'])
        else:
            self.methods[method_name] = copy.deepcopy(self._json_file['methods'][method_name])

    def _set_isotropic_weighting_default(self):
        voxelSize = self.params['voxelSize_mm']
        self.options['isotropic_weighting_factor'] = \
            [1.0, np.around(voxelSize[0]/voxelSize[1], 2),
                np.around(voxelSize[0]/voxelSize[2], 2)]

    def _set_mask_and_signal_original(self):
        self._mask_original = self.mask.copy()
        arrays2downsample = self._json_file['arrays2downsample']
        for arr_dict in arrays2downsample:
            arr_name = arr_dict['name']
            try:
                arr = getattr(self, arr_name)
            except AttributeError:
                continue
            setattr(self, "_{}_original".format(arr_name), arr.copy())
            self._voxel_size_original = self.params['voxelSize_mm']

    def perform(self, method=None):
        if method is None:
            if self._json_file['class'] == 'Multi-echo':
                method = self._json_file['default_method'][self.params['signal_model']]
            else:
                method = self._json_file['default_method']
        if self.verbose:
            class_name = self._json_file['class']
            print(f'{class_name} method: {method}')
        options_start = self.options.copy()
        options_method = self.methods[method]

        self.trim_and_pad((0, 0, 0))

        for i in range(len(options_method)):
            self._break_layer = False
            if 'repeat' in options_method[i]:
                repeat = options_method[i]['repeat']
            else:
                repeat = 1

            for j in range(repeat):
                if self.verbose:
                    layer_name = options_method[i]['name']
                    self._verbose_start(f'{i+1}/{len(options_method)}: {layer_name}...')

                self._set_options_layer(options_method, i)
                self.change_resolution()
                if "run_functions" in self.options:
                    for function in self.options['run_functions']:
                        func = getattr(self, function)
                        func()
                if self._break_layer:
                    if self.verbose:
                        self._verbose_stop()
                    break
                layer_type = options_method[i]['type']
                layer = getattr(self, f'_layer_{layer_type}')
                layer()
                if self.verbose:
                    self._verbose_stop()

            if self._break_layer:
                continue

        self.revert_trim_and_pad()
        self.options = options_start
        free_mempool()

    @xp_function
    def change_resolution(self, xp=np):
        graph_map = self._json_file['graph_map']
        arrays2downsample = [{'name': 'mask', 'method': 'zoom'}]
        arrays2downsample = arrays2downsample + self._json_file['arrays2downsample']
        in_factor = np.array(self._mask_original.shape) / np.array(self.mask.shape)
        if 'voxelSize_mm' in self.options:
            out_factor = self.voxelSize2downsampling(self.options['voxelSize_mm'])
        else:
            out_factor = [1, 1, 1]

        if hasattr(self, graph_map):
            arr = getattr(self, graph_map)
            if not np.array_equal(in_factor, [1, 1, 1]):
                if self.options['smoothing']:
                    mask_filtered = gaussian_filter(self.mask.astype(np.float32),
                                                    sigma=self.options['smoothing_sigma'])
                    arr = gaussian_filter(arr,
                                          sigma=self.options['smoothing_sigma'])
                    arr[mask_filtered != 0] = arr[mask_filtered != 0]/mask_filtered[mask_filtered != 0]

                arr[~self.mask] = np.nan
                arr = inpaint(arr, ~np.isfinite(arr))
                arr[~np.isfinite(arr)] = 0
                if np.array_equal(in_factor, out_factor):
                    self._prior.append(arr.copy())
                arr = zoom_arr(arr, in_factor, order=1, mode='round')
                self._prior_original.append(arr.copy())
                if not np.array_equal(in_factor, out_factor):
                    self._prior.append(arr.copy())
            else:
                if self.options['smoothing']:
                    mask_filtered = gaussian_filter(self.mask.astype(np.float32),
                                                    sigma=self.options['smoothing_sigma'])
                    arr = gaussian_filter(arr,
                                          sigma=self.options['smoothing_sigma'])
                    arr[mask_filtered != 0] = arr[mask_filtered != 0]/mask_filtered[mask_filtered != 0]

                self._prior_original.append(arr.copy())
                self._prior.append(arr.copy())

        if not np.array_equal(in_factor, out_factor):
            if not np.array_equal(out_factor, [1, 1, 1]):
                factor = 1/out_factor
                for arr_dict in arrays2downsample:
                    arr_name = arr_dict['name']
                    arr = getattr(self, "_{}_original".format(arr_name)).copy()
                    if arr_name.startswith('mask'):
                        arr = arr.astype(np.float32)
                        arr = zoom_arr(arr, factor, order=1)
                        arr[~np.isfinite(arr)] = 0
                        arr = arr.astype(np.bool_)
                    else:
                        if arr_dict['method'] == 'downsample':
                            arr, _ = downsample(arr, out_factor, xp=xp)
                            arr[~self.mask] = np.nan
                        elif arr_dict['method'] == 'zoom':
                            arr = zoom_arr(arr, factor, order=1)
                            arr[~self.mask] = 0
                        else:
                            raise NotImplementedError
                    setattr(self, arr_name, arr)

                voxelSize_factor = np.array(self._mask_original.shape) \
                    / np.array(self.mask.shape)
                self.params['voxelSize_mm'] = \
                    np.multiply(self._voxel_size_original, voxelSize_factor)
                for i in range(len(self._prior_original)):
                    arr = self._prior_original[i].copy()
                    arr = zoom_arr(arr, factor)
                    self._prior[i] = arr
            else:
                for arr_dict in arrays2downsample:
                    arr_name = arr_dict['name']
                    arr = getattr(self, "_{}_original".format(arr_name)).copy()
                    setattr(self, arr_name, arr)
                self._prior = self._prior_original.copy()
                if hasattr(self, graph_map):
                    setattr(self, graph_map, self._prior[-1])
                self.params['voxelSize_mm'] = self._voxel_size_original

    def _get_prior(self):
        if self.options['prior'] is not False:
            prior_range = self._prior[(self.options['prior']['layer_for_range'])][self.mask]
            layer_for_insert = self.options['prior']['layer_for_insert']
            if len(layer_for_insert) > 0:
                prior_insert = np.zeros((prior_range.shape[0],
                                         len(layer_for_insert)))
                for i in range(len(layer_for_insert)):
                    prior_insert[:, i] = \
                        self._prior[layer_for_insert[i]][self.mask]
            else:
                prior_insert = None
        else:
            prior_range = None
            prior_insert = None
        return prior_range, prior_insert

    def _verbose_start(self, string):
        print(string, end=" ", flush=True)
        self.start_time = time.time()

    def _verbose_stop(self):
        print('done! ({}s)'.format(np.round(time.time() - self.start_time, 2)))

    @xp_function
    def trim_and_pad(self, padsize, xp=np):
        self.params['mask_shape'] = self.mask.shape
        self.params['padsize'] = padsize
        arr, slicing = trim_zeros(self.mask)
        self.mask = pad_array3d(arr, padsize, xp)
        self.params['slicing'] = slicing
        arr = self.signal[slicing]
        if len(arr.shape) == 3:
            appended_dim = True
            arr = arr[..., xp.newaxis]
        else:
            appended_dim = False
        arr_shape = np.array(arr.shape)
        arr_new_shape = np.append((arr_shape[:3]+2*np.array(padsize)),
                                  arr_shape[-1]).astype(np.int32)
        arr_new = xp.zeros(arr_new_shape, dtype=xp.complex64)
        for i in range(arr.shape[-1]):
            arr_new[..., i] = move2gpu(pad_array3d(move2cpu(arr[..., i], xp), padsize), xp)
        if appended_dim:
            arr = arr[..., 0]
        self.signal = move2cpu(arr_new, xp)
        for map_name in self._json_file['maps']:
            if hasattr(self, map_name):
                arr = getattr(self, map_name)[slicing]
                setattr(self, map_name, pad_array3d(arr, padsize, xp))
        if hasattr(self, 'images'):
            for key in self.images.keys():
                self.images[key] = pad_array3d(self.images[key][slicing],
                                               padsize, xp)
        self._set_mask_and_signal_original()
        self.phase_inpaint()

    @xp_function
    def revert_trim_and_pad(self, xp=np):
        shape = self.params['mask_shape']
        slicing = self.params['slicing']
        padsize = self.params['padsize']
        arr = depad_array3d(self.mask, padsize)
        self.mask = revert_trim_zeros(arr, slicing, shape)
        arr_new = xp.zeros(np.append(shape, self.signal.shape[-1]),
                           dtype=xp.complex64)
        for i in range(self.signal.shape[-1]):
            tmp_arr = depad_array3d(self.signal[..., i], padsize)
            arr_new[..., i] = move2gpu(revert_trim_zeros(tmp_arr, slicing, shape), xp)
        self.signal = move2cpu(arr_new, xp)
        for map_name in self._json_file['maps']:
            if hasattr(self, map_name):
                arr = depad_array3d(getattr(self, map_name), padsize)
                setattr(self, map_name, revert_trim_zeros(arr, slicing, shape))
        if hasattr(self, 'images'):
            for key in self.images.keys():
                arr = depad_array3d(self.images[key], padsize)
                self.images[key] = revert_trim_zeros(arr, slicing, shape)
        self._set_mask_and_signal_original()

    def voxelSize2downsampling(self, voxelSize_mm):
        voxelSize_mm = np.array(voxelSize_mm)
        voxelSize_mm_original = np.array(self._voxel_size_original)
        downsampling_factor = voxelSize_mm/voxelSize_mm_original
        downsampling_factor[downsampling_factor < 1] = 1
        return downsampling_factor

    def phase_inpaint(self):
        arr = self.signal
        scale = 1e3
        phase = inpaint(np.angle(arr)*scale, ~self.mask) / scale
        self.signal = np.abs(arr)*np.exp(1j*phase)

    def _set_options_layer(self, options_method, index):
        self.set_options_default()
        self.options.update(options_method[index])

    
def depad_array3d(arr, padsize):
    ''':param arr: numpy array :returns: depadded array symmetrically with size
    given in padsize (per dimension)

    Christof Boehm,
    christof.boehm@tum.de

    '''
    kernelSize = arr.shape
    arrSmall = arr[padsize[0]:(kernelSize[0]-padsize[0]), \
                   padsize[1]:(kernelSize[1]-padsize[1]), \
                   padsize[2]:(kernelSize[2]-padsize[2])]
    return arrSmall


def pad_array3d(arr, padsize, xp=np):
    '''
    :param arr: numpy array
    :returns: padded array symmetrically with size given in padsize (per dimension)

    Christof Boehm,
    christof.boehm@tum.de

    '''
    arr = move2gpu(arr, xp)
    arrBig = xp.pad(arr, ((padsize[0], padsize[0]), \
                          (padsize[1], padsize[1]), \
                          (padsize[2], padsize[2])), 'constant', \
                    constant_values = 0)
    return move2cpu(arrBig, xp)

def trim_zeros(arr, margin=0):
    '''
    Trim the leading and trailing zeros from a N-D array.

    :param arr: numpy array
    :param margin: how many zeros to leave as a margin
    :returns: trimmed array
    :returns: slice object

    Christof Boehm,
    christof.boehm@tum.de
    '''
    s = []
    for dim in range(arr.ndim):
        start = 0
        end = -1
        slice_ = [slice(None)]*arr.ndim

        go = True
        while go:
            slice_[dim] = start
            go = not np.any(arr[tuple(slice_)])
            start += 1
        start = max(start-1-margin, 0)

        go = True
        while go:
            slice_[dim] = end
            go = not np.any(arr[tuple(slice_)])
            end -= 1
        end = arr.shape[dim] + min(-1, end+1+margin) + 1

        s.append(slice(start,end))
    return arr[tuple(s)], tuple(s)


def revert_trim_zeros(arr, slicing, orig_shape, xp=np):
    x = orig_shape[0] - slicing[0].stop
    y = orig_shape[1] - slicing[1].stop
    z = orig_shape[2] - slicing[2].stop

    arr = move2gpu(arr)
    arrBig = xp.pad(arr, ((slicing[0].start, x), \
                          (slicing[1].start, y), \
                          (slicing[2].start, z)), 'constant', \
                    constant_values = 0)

    return move2cpu(arrBig, xp)
    
def downsample(arr, downsample_factor, xp=np):
    """Downsample array

    Downsample array by averaging neighboring voxels. If (nx % downsample_factor[0],
    ny % downsample_factor[1], nz % downsample_factor[2]) != (0, 0, 0), the array
    will be padded with zeros.

    :param arr: N-dim array of shape (nx, ny, nz, ...)
    :param downsample_factor: 3-dim array with downsampling factor (int) in [x, y, z]
    :returns:
        - N-dim array of new shape (nx/downsample_factor[0], ny/downsample_factor[1], nz/downsample_factor[2], ...)
        - Padding width (needs to be saved for revert)

    """
    arr = move2gpu(arr, xp)
    downsample_factor = xp.array(downsample_factor)

    new_shape = xp.array(arr.shape)
    new_shape[:3] = xp.ceil(xp.array(arr.shape[:3])/downsample_factor)
    pad_width = []
    for i in range(len(arr.shape)):
        if i < 3:
            pad_width_value = new_shape[i]*downsample_factor[i]-arr.shape[i]
            pad_width.append((int(pad_width_value//2), int(pad_width_value - pad_width_value//2)))
        else:
            pad_width.append((0, 0))

    arr = xp.pad(arr, pad_width, 'constant', constant_values = np.nan)
    shape = np.zeros((len(arr.shape)+3), dtype=np.uint32)
    new_shape = move2cpu(new_shape)
    shape[:6] = [new_shape[0], arr.shape[0]/new_shape[0], new_shape[1], arr.shape[1]/new_shape[1],
                 new_shape[2], arr.shape[2]/new_shape[2]]
    shape[6:] = arr.shape[3:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return move2cpu(np.nanmean(np.nanmean(np.nanmean(arr.reshape(tuple(shape)),
                                                         axis=5), axis=3), axis=1), xp), pad_width


def revert_downsample(arr, downsample_factor, pad_width, order=0):
    """Revert :meth:`bmrr_shared_helper.padding.revert_downsample`

    Use `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html>`_
    zoom function with order 0 and reverts padding

    :param arr: 3-dim or 4-dim array of shape (nx, ny, nz, ...)
    :param interpolate_factor: 3-dim array with original downsampling factor in [x, y, z]
    :param pad_width: Padding width used in :meth:`bmrr_shared_helper.padding.revert_downsample`
    :returns: 3-dim or 4-dim array of new shape (nx*interpolate_factor[0], ny*interpolate_factor[1], nz*interpolate_factor[2], ...)

    """
    zoomedArr = zoom_arr(arr, downsample_factor, order=order)

    kernel_size = zoomedArr.shape
    zoomedArr = zoomedArr[pad_width[0][0]:(kernel_size[0]-pad_width[0][1]), \
                          pad_width[1][0]:(kernel_size[1]-pad_width[1][1]), \
                          pad_width[2][0]:(kernel_size[2]-pad_width[2][1])]
    return zoomedArr


def inpaint(arr, mask):
    """
    Wraps `OpenCV <https://docs.opencv.org/master/d7/d8b/group__photo__inpaint.html#gaedd30dfa0214fec4c88138b51d678085>`_
    inpaint function with the algorithm proposed by Alexandru Telea.

    :param arr: 3-dim or 4-dim array of shape (nx, ny, nz, ...)
    :param mask: interpolation order
    :returns: inpainted array

    """
    if len(arr.shape) == 3:
        shape = 3
        arr = arr[..., np.newaxis]
    else:
        shape = 4

    # rescale
    scale = 1e3/np.nanmean(arr)
    radius = 0.5
    arr = arr * scale

    new_arr = np.zeros_like(arr)

    for i in range(arr.shape[-1]):
        for j in range(arr.shape[-2]):
            if len(mask.shape) == 4:
                new_arr[..., j, i].real = cv.inpaint(arr[..., j, i].real, np.array(mask, dtype=np.uint8)[..., j, i],
                                                     radius, cv.INPAINT_TELEA)
                if np.sum(np.iscomplex(arr)) > 0:
                    new_arr[..., j, i].imag = cv.inpaint(arr[..., j, i].imag, np.array(mask, dtype=np.uint8)[..., j, i],
                                                         radius, cv.INPAINT_TELEA)
            else:
                new_arr[..., j, i].real = cv.inpaint(arr[..., j, i].real, np.array(mask, dtype=np.uint8)[..., j],
                                                     radius, cv.INPAINT_TELEA)
                if np.sum(np.iscomplex(arr)) > 0:
                    new_arr[..., j, i].imag = cv.inpaint(arr[..., j, i].imag, np.array(mask, dtype=np.uint8)[..., j],
                                                         radius, cv.INPAINT_TELEA)

    if shape == 3:
        new_arr = new_arr[..., 0]
    return new_arr / scale


def interpolate(arr, factor, spline_order=0):
    return zoom_arr(arr, factor, order=spline_order)


def zoom_arr(arr, factor, order=2, mode='ceil'):
    """
    Wraps `scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html>`_
    zoom function. If GPU: cupy implementation is used

    :param arr: 3-dim or 4-dim array of shape (nx, ny, nz, ...)
    :param factor: 3-dim array with zoom factor in [x, y, z]
    :param order: order of the spline interpolation
    :returns: 3-dim or 4-dim array of new shape (nx*interpolate_factor[0], ny*interpolate_factor[1], nz*interpolate_factor[2], ...)

    """
    shape = np.zeros(3, dtype=np.uint16)
    grid_mode = False
    shape = np.zeros(3, dtype=np.uint16)
    for i in range(len(shape)):
        if mode == 'round':
            shape[i] = int(np.round(arr.shape[i]*factor[i]))
            if shape[i] == 0:
                shape[i] = 1
        elif mode == 'ceil':
            shape[i] = int(np.ceil(arr.shape[i]*factor[i]))
    factor = shape / arr.shape[:3]

    if len(arr.shape) == 4:
        N_params = int(arr.shape[-1])
        zoomedArr = np.zeros((shape[0], shape[1], shape[2],
                              N_params), dtype=arr.dtype)
        try:
            for i in range(N_params):
                zoomedArr[..., i] = zoom(arr[..., i], factor, mode='nearest',
                                         grid_mode=grid_mode, order=order)
        except TypeError: # workaround for scipy version with no support for complex arrays
            for i in range(N_params):
                real = zoom(arr.real[..., i], factor, mode='nearest',
                            grid_mode=grid_mode, order=order)
                imag = zoom(arr.imag[..., i], factor, mode='nearest',
                            grid_mode=grid_mode, order=order)
                zoomedArr[..., i] = real+1j*imag
    elif len(arr.shape) == 3:
        try:
            zoomedArr = zoom(arr, factor, mode='nearest',
                             grid_mode=grid_mode, order=order)
        except TypeError:
            real = zoom(arr.real, factor, mode='nearest',
                        grid_mode=grid_mode, order=order)
            imag = zoom(arr.imag, factor, mode='nearest',
                        grid_mode=grid_mode, order=order)
            zoomedArr = real+1j*imag
    return zoomedArr
