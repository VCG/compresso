cimport cython
cimport numpy as np
import numpy as np
import ctypes
import math
import sys
        
(bz, by, bx) = (8, 8, 8)
(chunkz, chunky, chunkx) = (64, 64, 64)

cdef extern from 'cpp-neuroglancer.h' namespace 'neuroglancer':
    unsigned long *Compress(unsigned long *data, int zres, int yres, int xres, int bz, int by, int bx, int origz, int origy, int origx)
    unsigned long *Decompress(unsigned long *compressed_data, int bz, int by, int bx)

from pyneuroglancer import DecodeNeuroglancer

# use cython for decompress (slower)
cython_decompress = True


########################
### DECODE FUNCTIONS ###
########################

class neuroglancer(object):

    @staticmethod
    def name():
        return 'Neuroglancer'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''Neuroglancer compression
        '''
        origz, origy, origx = data.shape

        # determine the number of chunks of data
        zres, yres, xres = data.shape

        nzchunks, nychunks, nxchunks = (int(math.ceil(float(zres) / chunkz) + 0.5), int(math.ceil(float(yres) / chunky) + 0.5), int(math.ceil(float(xres) / chunkx) + 0.5))
        
        compressed_data = np.zeros(3, dtype=np.uint64)
        compressed_data[0] = zres
        compressed_data[1] = yres
        compressed_data[2] = xres

        cdef np.ndarray[unsigned long, ndim=3, mode='c'] cpp_data
        cdef unsigned long *cpp_compressed_data
        cdef unsigned long[:] tmp_compressed_data

        # compress every chunk
        for iz in range(0, nzchunks):
            for iy in range(0, nychunks):
                for ix in range(0, nxchunks):
                    chunk = data[
                        iz * chunkz:(iz + 1) * chunkz, 
                        iy * chunky:(iy + 1) * chunky,
                        ix * chunkx:(ix + 1) * chunkx
                    ]

                    # create header variables
                    zres, yres, xres = chunk.shape
                    origz, origy, origx = zres, yres, xres

                    if zres % bz: zpad = (bz - zres % bz)
                    else: zpad = 0
                    if yres % by: ypad = (by - yres % by)
                    else: ypad = 0
                    if xres % bx: xpad = (bx - xres % bx)
                    else: xpad = 0

                    zres += zpad
                    yres += ypad
                    xres += xpad

                    padded_data = np.pad(chunk, ((0, zpad), (0, ypad), (0, xpad)), 'reflect').astype(np.uint64)
                                
                    cpp_data = np.ascontiguousarray(padded_data, dtype=ctypes.c_uint64)
                    cpp_compressed_data = Compress(&(cpp_data[0,0,0]), zres, yres, xres, bz, by, bx, origz, origy, origx)
                    length = cpp_compressed_data[0]
                    tmp_compressed_data = <unsigned long[:length]> cpp_compressed_data
                    compressed_data = np.concatenate((compressed_data, np.asarray(tmp_compressed_data)))

        return compressed_data

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''Neuroglancer decompression
        '''
        # get the uncompressed data size
        zres = data[0]
        yres = data[1]
        xres = data[2]
        data = data[3:]

        nzchunks, nychunks, nxchunks = (int(math.ceil(float(zres) / chunkz) + 0.5), int(math.ceil(float(yres) / chunky) + 0.5), int(math.ceil(float(xres) / chunkx) + 0.5))

        # create an empty decompressed array
        decompressed_data = np.zeros((zres, yres, xres), dtype=np.uint64)

        cdef np.ndarray[unsigned long, ndim=1, mode='c'] cpp_data
        cdef unsigned long[:] cpp_decompressed_chunk
        
        if not cython_decompress:
            # go through every chunk
            for iz in range(0, nzchunks):
                for iy in range(0, nychunks):
                    for ix in range(0, nxchunks):
                        # get the size of the data
                        length = int(data[0])

                        az, ay, ax = data[1], data[2], data[3]

                        gz, gy, gx = (
                            int(math.ceil(float(az) / bz)),
                            int(math.ceil(float(ay) / by)),
                            int(math.ceil(float(ax) / bx))
                        )

                        # get the total size of the header
                        nelements = gz * gy * gx

                        # later will become 3 bytes from 4
                        table_offsets = np.zeros(nelements, dtype=np.uint32)
                        nbits = np.zeros(nelements, dtype=np.uint8)
                        values_offsets = np.zeros(nelements, dtype=np.uint32)

                        ################################
                        # DECOMPRESS HEADER VALUES
                        ################################

                        # get the original data size
                        origz, origy, origx = data[4], data[5], data[6]

                        data_entries = 7
                        for ie in range(0, nelements):
                            header = long(data[data_entries])
                            table_offsets[ie] = header >> 40
                            nbits[ie] = (header << 24) >> 56
                            values_offsets[ie] = (header << 32) >> 32

                            data_entries += 1

                        ###############################
                        # DECOMPRESS ENTIRE IMAGE
                        ###############################

                        # remove the first element (length not needed)
                        decompressed_chunk = DecodeNeuroglancer(data[:length], table_offsets, nbits, values_offsets, data_entries, bz, by, bx)
                        decompressed_chunk = np.reshape(decompressed_chunk, (az, ay, ax))
 
                        decompressed_data[
                            iz * chunkz:(iz + 1) * chunkz,
                            iy * chunky:(iy + 1) * chunky,
                            ix * chunkx:(ix + 1) * chunkx
                        ] = decompressed_chunk[0:origz,0:origy,0:origx]

                        data = data[length:]

            return decompressed_data
        else:
            # go through every chunk
            for iz in range(0, nzchunks):
                for iy in range(0, nychunks):
                    for ix in range(0, nxchunks):
                        # get the size of the data
                        length = int(data[0])
                        az, ay, ax = data[1], data[2], data[3]
                        origz, origy, origx = data[4], data[5], data[6]

                        ###############################
                        # DECOMPRESS ENTIRE IMAGE
                        ###############################
                        cpp_data = np.ascontiguousarray(data[:length], dtype=ctypes.c_uint64)
                        n = az * ay * ax

                        cpp_decompressed_chunk = <unsigned long[:n]> Decompress(&(cpp_data[0]), bz, by, bx)
                        decompressed_chunk = np.reshape(np.asarray(cpp_decompressed_chunk), (az, ay, ax))
                        
                        decompressed_data[
                            iz * chunkz:(iz + 1) * chunkz,
                            iy * chunky:(iy + 1) * chunky,
                            ix * chunkx:(ix + 1) * chunkx
                        ] = decompressed_chunk[0:origz,0:origy,0:origx]

                        data = data[length:]
            
            return decompressed_data
