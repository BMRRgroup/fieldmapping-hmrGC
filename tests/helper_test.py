import pytest
import os
import numpy as np
try:
    import cupy as cp
except ModuleNotFoundError:
    import numpy as cp

from hmrGC.helper import get_arr_chunks, ChunkArr, move2cpu


def test_get_arr_chunks(test_gpu):
    if test_gpu:
        if os.environ['BMRR_USE_GPU'] == '1':
            xp = cp
        else:
            xp = np
        arr = xp.ones((10, 10))
        arr_new = xp.ones((0, 10))
        chunksize = 2
        num_chunks = int(np.ceil(arr.shape[0]/chunksize))
        for i in range(num_chunks):
            arr_chunk = get_arr_chunks(chunksize, arr, i, num_chunks)
            if i < num_chunks - 1:
                assert arr_chunk.shape[0] == chunksize
            arr_new = xp.concatenate((arr_new, arr_chunk))
        assert xp.array_equal(arr, arr_new)


def test_chunkarr_concatenate(test_gpu):
    if test_gpu:
        if os.environ['BMRR_USE_GPU'] == '1':
            xp = cp
        else:
            xp = np
        arr = xp.ones(10)
        chunkarr = ChunkArr('int32', reduce_gpu_mem=True, shape=None, xp=xp)
        chunkarr.concatenate(arr)
        assert chunkarr.arr.shape == arr.shape
        assert xp.array_equal(chunkarr.arr, move2cpu(arr))

        arr = xp.ones(10)
        chunkarr = ChunkArr('int32', reduce_gpu_mem=False, shape=None, xp=xp)
        chunkarr.concatenate(arr)
        assert chunkarr.arr.shape == arr.shape

        arr = xp.ones((10,10))
        chunkarr = ChunkArr('int32', reduce_gpu_mem=False, shape=(0,10), xp=xp)
        chunkarr.concatenate(arr)
        assert chunkarr.arr.shape == arr.shape
        chunkarr.concatenate(arr)
        assert chunkarr.arr.shape == (20, 10)

        arr = xp.ones((10,10))
        chunkarr = ChunkArr('int32', reduce_gpu_mem=True, shape=(0,10), xp=xp)
        chunkarr.concatenate(arr)
        assert chunkarr.arr.shape == arr.shape
        chunkarr.concatenate(arr)
        assert chunkarr.arr.shape == (20, 10)


def test_chunkarr_to_numpy(test_gpu):
    if test_gpu:
        if os.environ['BMRR_USE_GPU'] == '1':
            xp = cp
        else:
            xp = np
        arr = xp.ones(10, dtype=np.int32)
        chunkarr = ChunkArr('int32', reduce_gpu_mem=True, shape=None, xp=xp)
        chunkarr.concatenate(arr)
        arr_numpy = chunkarr.to_numpy()
        assert arr_numpy.dtype == np.int32
