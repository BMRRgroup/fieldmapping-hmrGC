import numpy as np
from numba import njit, prange
import os
from functools import wraps
import numpy as np

try:
    import cupy as cp
    from cupy.cuda.memory import OutOfMemoryError as CUDA_OutOfMemory

    def xp_function(f):
        """GPU/CPU function decorator for class methods
        The class needs to have the attribute `self.use_gpu`

        :param f: Function with keyword argument `xp=np`
        :returns: GPU optimized function

        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            try:
                if args[0].use_gpu:
                    ret = f(*args, **kwargs, xp=cp)
                else:
                    ret = f(*args, **kwargs, xp=np)
                return ret
            except CUDA_OutOfMemory:
                print('CUDA_OutOfMemory: GPU mode has been ended.')
                print('CPU mode will be enabled.')
                ret = f(*args, **kwargs, xp=np)
                return ret
        return wrapper

    def move2cpu(x, xp=cp):
        """Returns a numpy array

        :param x: numpy/cupy array
        :param xp: used python module (numpy or cupy)
        :returns: numpy array

        """
        if xp == cp:
            if isinstance(x, np.ndarray):
                y = x
            elif isinstance(x, cp.ndarray):
                y = cp.asnumpy(x)
            return y
        elif xp == np:
            return x

    def move2gpu(x, xp=cp):
        """Returns a cupy array

        :param x: numpy/cupy array
        :param xp: used python module (numpy or cupy)
        :returns: cupy array

        """
        if xp == cp:
            if isinstance(x, np.ndarray):
                y = cp.asarray(x)
            elif isinstance(x, cp.ndarray):
                y = x
            return y
        elif xp == np:
            return x

    def free_mempool():
        """Clear the default cupy mempool"""
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

except ModuleNotFoundError:
    import numpy as xp

    class CUDA_OutOfMemory(Exception):
        pass

    def xp_function(f):
        """GPU/CPU function decorator

        :param f: Function with keyword argument `xp=np`
        :returns: GPU optimized function

        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            ret = f(*args, **kwargs, xp=np)
            return ret
        return wrapper

    def move2cpu(x, xp=np):
        """Returns a numpy array

        :param x: numpy/cupy array
        :param xp: used python module (numpy or cupy)
        :returns: numpy array

        """
        return x

    def move2gpu(x, xp=np):
        """Returns a cupy array

        :param x: numpy/cupy array
        :param xp: used python module (numpy or cupy)
        :returns: cupy array

        """
        return x

    def free_mempool():
        """Clear the default cupy mempool"""
        pass


def scale_array2interval_along1D(array, intervalVector, xp=xp):
    array = move2gpu(array, xp=xp)
    intervalVector = xp.array(intervalVector).astype(np.float32)

    newValueRange = (xp.max(intervalVector) - xp.min(intervalVector))*xp.ones_like(array)

    valueRange = xp.repeat(xp.array(xp.nanmax(array, axis=-1) - xp.nanmin(array, axis=-1))[:, xp.newaxis],
                           array.shape[-1], axis=-1)

    rescaleSlope = xp.ones_like(array)
    rescaleSlope[valueRange != 0] = xp.divide(newValueRange[valueRange != 0],
                                               valueRange[valueRange != 0])

    arrayScaled = xp.multiply(rescaleSlope, array)

    # shift by intercept
    rescaleIntercept = xp.repeat(xp.array(xp.min(intervalVector) - xp.nanmin(arrayScaled, axis=-1))[:, xp.newaxis],
                                 array.shape[-1], axis=-1)
    arrayScaled += rescaleIntercept

    return move2cpu(arrayScaled, xp)


def scale_array2interval(array, intervalVector, xp=xp):
    array = move2gpu(array, xp)
    intervalVector = xp.array(intervalVector).astype(np.float32)
    newValueRange = xp.max(intervalVector) - xp.min(intervalVector)
    valueRange = xp.max(array) - xp.min(array)

    if valueRange == 0:
        rescaleSlope = 1
    else:
        rescaleSlope = newValueRange / valueRange

    arrayScaled = rescaleSlope * array

    # shift by intercept
    rescaleIntercept = xp.min(intervalVector) - xp.min(arrayScaled)
    arrayScaled += rescaleIntercept

    return move2cpu(arrayScaled, xp)


@njit(parallel=True)
def njit_residuals(signal, p_matrix):
    residuals = np.zeros((signal.shape[0], p_matrix.shape[0]), dtype=np.float32)
    for i in prange(p_matrix.shape[0]):
        p_matrix_chunk = np.ascontiguousarray(p_matrix[i, :, :])
        residuals[:, i] = np.sum(np.abs(np.dot(signal, p_matrix_chunk)) ** 2,
                                 axis=-1)
    return residuals


@njit(parallel=True)
def njit_residuals_prior(signal, p_matrix, prior, neighborhood):
    residuals = np.zeros((signal.shape[0], np.sum(neighborhood)+1), dtype=np.float32)
    for i in prange(signal.shape[0]):
        p_matrix_prior = p_matrix[prior[i]-neighborhood[0]:prior[i]+neighborhood[1]+1]
        shape = p_matrix_prior.shape
        p_matrix_prior = np.ascontiguousarray(np.transpose(p_matrix_prior, (1, 0, 2)))
        p_matrix_prior = p_matrix_prior.reshape(shape[1], shape[0]*shape[2])
        residuals[i, :] = np.sum(np.abs(np.reshape(np.dot(signal[i], p_matrix_prior),
                                                (shape[0], shape[2]))) ** 2, axis=-1)
    return residuals


def get_arr_chunks(chunksize, arr, i, num_chunks):
    if i == num_chunks-1:
        arr_chunk = arr[i*chunksize:]
    else:
        arr_chunk = arr[i*chunksize:(i+1)*chunksize]
    return arr_chunk


class ChunkArr():
    def __init__(self, dtype, reduce_gpu_mem, shape=None, xp=np):
        self.reduce_gpu_mem = reduce_gpu_mem
        self.xp = xp

        if self.reduce_gpu_mem:
            if shape is None:
                self.arr = np.array([], np.dtype(dtype))
            else:
                self.arr = np.zeros(shape, np.dtype(dtype))
        else:
            if shape is None:
                self.arr = xp.array([], xp.dtype(dtype))
            else:
                self.arr = xp.zeros(shape, xp.dtype(dtype))

    def concatenate(self, arr_chunk):
        if self.reduce_gpu_mem:
            self.arr = np.concatenate((self.arr, move2cpu(arr_chunk)))
        else:
            self.arr = self.xp.concatenate((self.arr, arr_chunk))

    def to_numpy(self):
        return move2cpu(self.arr)
