"""
Python bindings for the Compresso labeled image compression algorithm.

B. Matejek, D. Haehn, F. Lekschas, M. Mitzenmacher, and H. Pfister.
"Compresso: Efficient Compression of Segmentation Data for Connectomics".
Springer: Intl. Conf. on Medical Image Computing and Computer-Assisted Intervention.
2017.

https://vcg.seas.harvard.edu/publications/compresso-efficient-compression-of-segmentation-data-for-connectomics
https://github.com/vcg/compresso

PyPI Distribution: 
https://github.com/seung-lab/compresso

License: MIT
"""

cimport cython
cimport numpy as np
import numpy as np
import ctypes
from libc.stdint cimport (
  int8_t, int16_t, int32_t, int64_t,
  uint8_t, uint16_t, uint32_t, uint64_t,
)

cdef extern from "compresso.hxx" namespace "compresso":
    uint64_t *Compress(
        uint64_t *data, 
        size_t sx, size_t sy, size_t sz, 
        size_t xstep, size_t ystep, size_t zstep
    )
    uint64_t *Decompress(uint64_t *compressed_data)

def compress(data):
    """
    Compress a three dimensional numpy array containing image segmentation
    using the Compresso algorithm.
    
    Returns: compressed bytes b'...'
    """
    from math import ceil

    # reshape the data into one dimension
    sx, sy, sz = data.shape
    (zstep, ystep, xstep) = (1, 8, 8)
    header_size = 9

    nzblocks = int(ceil(float(sz) / zstep))
    nyblocks = int(ceil(float(sy) / ystep))
    nxblocks = int(ceil(float(sx) / xstep))
    nblocks = nzblocks * nyblocks * nxblocks

    cdef np.ndarray[uint64_t, ndim=3] cpp_data = np.asfortranarray(data, dtype=np.uint64)
    cdef uint64_t *cpp_compressed_data = Compress(
        &(cpp_data[0,0,0]), sx, sy, sz, xstep, ystep, zstep
    )
    length = header_size + cpp_compressed_data[3] + cpp_compressed_data[4] + cpp_compressed_data[5] + nblocks
    cdef uint64_t[:] tmp_compressed_data = <uint64_t[:length]> cpp_compressed_data
    compressed_data = np.asarray(tmp_compressed_data)

    # compress all the zeros in the window values

    nblocks = int(ceil(float(sz) / zstep)) * int(ceil(float(sy) / ystep)) * int(ceil(float(sx) / xstep))
    
    intro_data = compressed_data[:-nblocks]
    block_data = compressed_data[-nblocks:]
    
    if (np.max(block_data) < 2**32):
        block_data = block_data.astype(np.uint32)

    condensed_blocks = list()
    inzero = False
    prev_zero = 0
    for ie, block in enumerate(block_data):
        if block == 0:
            # start counting zeros
            if not inzero:
                inzero = True
                prev_zero = ie
        else:
            if inzero:
                # add information for the previous zero segment
                condensed_blocks.append((ie - prev_zero) * 2 + 1)
                inzero = False
            condensed_blocks.append(block * 2)

    condensed_blocks = np.array(condensed_blocks).astype(np.uint32)

    return intro_data.tobytes() + condensed_blocks.tobytes()

def decompress(data):
    """
    Decompress a compresso encoded byte stream into a three dimensional 
    numpy array containing image segmentation.

    Returns: compressed bytes b'...'
    """
    from math import ceil 

    # read the first nine bytes corresponding to the header
    header = np.frombuffer(data[0:72], dtype=np.uint64)

    cdef size_t sz = header[0]
    cdef size_t sy = header[1]
    cdef size_t sx = header[2]
    cdef size_t voxels = sx * sy * sz
    
    ids_size = int(header[3])
    values_size = int(header[4])
    locations_size = int(header[5])
    zstep = header[6]
    ystep = header[7]
    xstep = header[8]

    # get the intro data
    intro_size = 9 + ids_size + values_size + locations_size
    intro_data = np.frombuffer(data[0:intro_size*8], dtype=np.uint64)

    # get the compressed blocks
    nblocks = int(ceil(float(sz) / zstep)) * int(ceil(float(sy) / ystep)) * int(ceil(float(sx) / xstep))
    compressed_blocks = np.frombuffer(data[intro_size*8:], dtype=np.uint32)
    block_data = np.zeros(nblocks, dtype=np.uint64)

    cdef size_t index = 0
    cdef size_t nzeros = 0
    for block in compressed_blocks:
        # greater values correspond to zero blocks
        if block % 2:
            nzeros = (block  - 1) // 2
            block_data[index:index+nzeros] = 0
            index += nzeros
        else:
            block_data[index] = block // 2
            index += 1

    data = np.concatenate((intro_data, block_data))

    cdef np.ndarray[uint64_t, ndim=1] cpp_data = np.asfortranarray(data, dtype=np.uint64)
    cdef uint64_t[:] cpp_decompressed_data = <uint64_t[:voxels]> Decompress(&(cpp_data[0]))
    decompressed_data = np.reshape(np.asarray(cpp_decompressed_data), (sx, sy, sz))

    return decompressed_data