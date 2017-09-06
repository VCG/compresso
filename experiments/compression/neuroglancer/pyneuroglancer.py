import math
import numpy as np
import time
from numba import jit

@jit(nopython=True)
def DecodeValues(block, values, encoded_values, bz, by, bx, nbits):
    # get the number of values per 8 byte uint64
    if (nbits > 0):
        values_per_uint64 = 64 / nbits

        ie = 0
        for value in encoded_values:
            for i in range(0, values_per_uint64):
                lower_bits_to_remove = (
                    (values_per_uint64 - i - 1) * nbits
                )
                values[ie] = (
                    (value >> lower_bits_to_remove) % 2**nbits
                )
                ie += 1

    ii = 0
    # get the lookup table
    for iw in range(0, bz):
        for iv in range(0, by):
            for iu in range(0, bx):
                block[iw, iv, iu] = values[ii]
                ii += 1

    return block, values

@jit(nopython=True)
def LookupTable(decompressed_data, lookup_table, block, iz, iy, ix, bz, by, bx):
    # read the lookup label
    for iw in range(0, bz):
        for iv in range(0, by):
            for iu in range(0, bx):
                decompressed_data[iz * bz + iw, iy * by + iv,ix * bx + iu] = lookup_table[block[iw, iv, iu]]


    return decompressed_data


def DecodeNeuroglancer(data, table_offsets, nbits, values_offsets, data_entries, bz, by, bx):
    # get the size of the data
    az, ay, ax = data[1], data[2], data[3]
    gz, gy, gx = (
        int(az / bz),
        int(ay / by),
        int(ax / bx)
    )

    decompressed_data = np.zeros((az, ay, ax), dtype=np.uint64)
    
    block_size = bz * by * bx

    index = 0
    for iz in range(0, gz):
        for iy in range(0, gy):
            for ix in range(0, gx):
                # get the total number of bits needed
                uint64s_needed = (
                    nbits[index] * block_size
                ) / 64

                uint64s_needed = int(uint64s_needed + 0.5)

                # get the encoded values
                encoded_values = data[values_offsets[index]:values_offsets[index] + uint64s_needed]

                # reconstruct the block with their ids
                block = np.zeros((bz, by, bx), dtype=np.uint32)

                # decode the values based on the number of bytes needed
                values = np.zeros(block_size, dtype=np.uint32)

                block, values = DecodeValues(block, values, encoded_values, bz, by, bx, nbits[index])

                # find the number of unique elements
                nunique = len(np.unique(block))
                lookup_table = data[
                    table_offsets[index]:table_offsets[index] + nunique
                ]
                decompressed_data = LookupTable(decompressed_data, lookup_table, block, iz, iy, ix, bz, by, bx)
                
                index += 1

    return decompressed_data