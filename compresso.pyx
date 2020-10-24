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
    uint64_t *Compress(uint64_t *data, int zres, int yres, int xres, int zstep, int ystep, int xstep)
    uint64_t *Decompress(uint64_t *compressed_data)

def compress(data):
    """
    Compress a three dimensional numpy array containing image segmentation
    using the Compresso algorithm.
    
    Returns: compressed bytes b'...'
    """
    from math import ceil

    # reshape the data into one dimension
    zres, yres, xres = data.shape
    (zstep, ystep, xstep) = (1, 8, 8)
    header_size = 9

    nzblocks = int(ceil(float(zres) / zstep))
    nyblocks = int(ceil(float(yres) / ystep))
    nxblocks = int(ceil(float(xres) / xstep))
    nblocks = nzblocks * nyblocks * nxblocks

    # call the Cython function
    cdef np.ndarray[uint64_t, ndim=3, mode='c'] cpp_data
    cpp_data = np.ascontiguousarray(data, dtype=ctypes.c_uint64)
    cdef uint64_t *cpp_compressed_data = Compress(&(cpp_data[0,0,0]), zres, yres, xres, zstep, ystep, xstep)
    length = header_size + cpp_compressed_data[3] + cpp_compressed_data[4] + cpp_compressed_data[5] + nblocks
    cdef uint64_t[:] tmp_compressed_data = <uint64_t[:length]> cpp_compressed_data
    compressed_data = np.asarray(tmp_compressed_data)

    # compress all the zeros in the window values

    nblocks = int(ceil(float(zres) / zstep)) * int(ceil(float(yres) / ystep)) * int(ceil(float(xres) / xstep))
    
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

    zres = header[0]
    yres = header[1]
    xres = header[2]
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
    nblocks = int(ceil(float(zres) / zstep)) * int(ceil(float(yres) / ystep)) * int(ceil(float(xres) / xstep))
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

    cdef np.ndarray[uint64_t, ndim=1, mode='c'] cpp_data
    cpp_data = np.ascontiguousarray(data, dtype=ctypes.c_uint64)
    n = zres * yres * xres

    cdef uint64_t[:] cpp_decompressed_data = <uint64_t[:n]> Decompress(&(cpp_data[0]))
    decompressed_data = np.reshape(np.asarray(cpp_decompressed_data), (zres, yres, xres))

    return decompressed_data