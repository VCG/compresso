import math
import numpy as np
from numba import jit


#######################
# VALUE ENCODING
#######################

@jit(nopython=True)
def EncodeValues(block, unique_elements):
    max_value = unique_elements[unique_elements.shape[0] - 1]

    # map elements to ints
    element_to_label = [-1] * (max_value + 1)
    seen = 0
    for unique in unique_elements:
        element_to_label[unique] = seen
        seen += 1

    # get the number of bits needed
    nunique = len(unique_elements)

    # record the number of bits needed
    if (nunique <= pow(2, 0)):
        nbits = 0
    elif (nunique <= pow(2, 1)):
        nbits = 1
    elif (nunique <= pow(2, 2)):
        nbits = 2
    elif (nunique <= pow(2, 4)):
        nbits = 4
    elif (nunique <= pow(2, 8)):
        nbits = 8
    elif (nunique <= pow(2, 16)):
        nbits = 16
    elif (nunique <= pow(2, 32)):
        nbits = 32

    encoded_values = np.zeros(block.size, dtype=np.uint32)

    ii = 0
    for iz in range(0, block.shape[0]):
        for iy in range(0, block.shape[1]):
            for ix in range(0, block.shape[2]):
                encoded_values[ii] = element_to_label[block[iz, iy, ix]]
                ii += 1

    # get the lookup table
    lookup_table = np.zeros(nunique, dtype=np.uint64)
    for i, element in enumerate(unique_elements):
        lookup_table[i] = element

    return nbits, encoded_values, lookup_table


def ConstructLookupTable(data):
    # actual size, block size, grid dimensions
    az, ay, ax = data.shape
    bz, by, bx = (8, 64, 64)
    gz, gy, gx = (
        int(math.ceil(float(az) / bz)),
        int(math.ceil(float(ay) / by)),
        int(math.ceil(float(ax) / bx))
    )

    # get the total size of the header
    nelements = gz * gy * gx
    # 8 bytes per entry (64 bits) plus 3 for the size
    header_offset = nelements + 3

    # later will become 3 bytes from 4
    table_offsets = np.zeros(nelements, dtype=np.uint32)
    nbits = np.zeros(nelements, dtype=np.uint8)
    values_offsets = np.zeros(nelements, dtype=np.uint32)

    encoded_values = [[]] * nelements
    lookup_table = [[]] * nelements

    # start with zeros bytes for entire header
    index = 0
    nuint64 = header_offset
    for iz in range(0, gz):
        for iy in range(0, gy):
            for ix in range(0, gx):
                # get the real (ii, ij, ik) location for the corner of this
                # block
                ii, ij, ik = ix * bx, iy * by, iz * bz

                # get this particular block
                block = data[ik:ik + bz, ij:ij + by, ii:ii + bx]

                # get the number of unique elements
                unique_elements = np.unique(block)

                nbits[index], encoded_values[index], lookup_table[index] = EncodeValues(block, unique_elements)

                values_offsets[index] = nuint64
                nuint64 += nbits[index] * block.size / 64
                table_offsets[index] = nuint64
                nuint64 += lookup_table[index].size

                index += 1

    return (
        nuint64,
        table_offsets,
        nbits,
        values_offsets,
        lookup_table,
        encoded_values
    )


@jit(nopython=True)
def EncodeChunk(compressed_data, data_entries, segment, nbits):
    value = 0
    for ic, chunk in enumerate(segment):
        shift = ((len(segment) - 1) - ic) * nbits
        value += (chunk << shift)

    # add the value to the compressed data
    compressed_data[data_entries] = value

    return compressed_data


def PopulateEncodedValues(
    compressed_data,
    encoded_values,
    lookup_table,
    nbits,
    gz,
    gy,
    gx,
    data_entries
):
    index = 0
    for iz in range(0, gz):
        for iy in range(0, gy):
            for ix in range(0, gx):

                if (nbits[index] > 0):
                    # number of values per 8 bytes
                    nvalues_per_entry = 64 / nbits[index]

                    # get all of the chunks that should be together
                    encoded_chunks = np.split(
                        encoded_values[index],
                        encoded_values[index].size / nvalues_per_entry
                    )

                    for segment in encoded_chunks:
                        compressed_data = EncodeChunk(
                            compressed_data,
                            data_entries,
                            segment,
                            nbits[index]
                        )
                        data_entries += 1

                for table_entry in lookup_table[index]:
                    compressed_data[data_entries] = table_entry
                    data_entries += 1

                index += 1


########################
### DECODE FUNCTIONS ###
########################

@jit(nopython=True)
def DecodeValues(block, values, encoded_values, sizez, sizey, sizex, nbits):
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
                    (value >> lower_bits_to_remove) %
                    2**nbits
                )
                ie += 1

    ii = 0
    # get the lookup table
    for iw in range(0, sizez):
        for iv in range(0, sizey):
            for iu in range(0, sizex):
                block[iw, iv, iu] = values[ii]
                ii += 1

    return block, values


@jit(nopython=True)
def LookupTable(
    decompressed_data,
    lookup_table,
    block,
    iz,
    iy,
    ix,
    bz,
    by,
    bx,
    sizez,
    sizey,
    sizex
):
    # read the lookup label
    for iw in range(0, sizez):
        for iv in range(0, sizey):
            for iu in range(0, sizex):
                decompressed_data[
                    iz * bz + iw, iy * by + iv, ix * bx + iu
                ] = lookup_table[block[iw, iv, iu]]

    return decompressed_data


def DecodeNeuroglancer(
    data, table_offsets, nbits, values_offsets, data_entries
):
    # get the size of the data
    az, ay, ax = data[0], data[1], data[2]
    bz, by, bx = (8, 64, 64)
    gz, gy, gx = (
        int(math.ceil(float(az) / bz) + 0.5),
        int(math.ceil(float(ay) / by) + 0.5),
        int(math.ceil(float(ax) / bx) + 0.5)
    )

    decompressed_data = np.zeros((az, ay, ax), dtype=np.uint64)

    index = 0
    for iz in range(0, gz):
        for iy in range(0, gy):
            for ix in range(0, gx):
                # find the start and end values
                startx, starty, startz = ix * bx, iy * by, iz * bz
                endx, endy, endz = startx + bx, starty + by, startz + bz
                if endx > ax:
                    endx = ax
                if endy > ay:
                    endy = ay
                if endz > az:
                    endz = az
                sizex = endx - startx
                sizey = endy - starty
                sizez = int(endz - startz)

                # get the total number of bits needed
                uint64s_needed = (
                    nbits[index] * (sizex * sizey * sizez)
                ) / 64

                uint64s_needed = int(uint64s_needed + 0.5)

                # get the encoded values
                encoded_values = data[
                    values_offsets[index]:values_offsets[index] +
                    uint64s_needed
                ]

                # reconstruct the block with their ids
                block = np.zeros((sizez, sizey, sizex), dtype=np.uint32)

                # decode the values based on the number of bytes needed
                values = np.zeros((sizex * sizey * sizez), dtype=np.uint32)

                block, values = DecodeValues(
                    block,
                    values,
                    encoded_values,
                    sizez,
                    sizey,
                    sizex,
                    nbits[index]
                )

                # find the number of unique elements
                nunique = len(np.unique(block))
                lookup_table = data[
                    table_offsets[index]:table_offsets[index] + nunique
                ]
                decompressed_data = LookupTable(
                    decompressed_data,
                    lookup_table,
                    block,
                    iz,
                    iy,
                    ix,
                    bz,
                    by,
                    bx,
                    sizez,
                    sizey,
                    sizex
                )

                index += 1

    return decompressed_data


class neuroglancer(object):

    @staticmethod
    def name():
        return 'Neuroglancer'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''Neuroglancer compression
        '''

        # actual size, block size, grid dimensions
        az, ay, ax = data.shape
        bz, by, bx = (8, 64, 64)
        gz, gy, gx = (
            int(math.ceil(float(az) / bz)),
            int(math.ceil(float(ay) / by)),
            int(math.ceil(float(ax) / bx))
        )

        # get the total size of the header
        nelements = gz * gy * gx
        # 8 bytes per entry (64 bits) plus 3 for the size
        header_offset = nelements + 3

        output_size, table_offsets, nbits, values_offsets, lookup_table, encoded_values = ConstructLookupTable(data)

        compressed_data = np.zeros(output_size, dtype=np.uint64)


        ##########################
        # COMPRESSION HEADER
        ##########################

        compressed_data[0] = az
        compressed_data[1] = ay
        compressed_data[2] = ax

        data_entries = 3
        for ie in range(0, nelements):
            compressed_data[data_entries] = (
                (table_offsets[ie] << 40) +
                (nbits[ie] << 32) +
                values_offsets[ie]
            )
            data_entries += 1


        ########################################
        # LOOKUP TABLES AND ENCODED VALUES
        ########################################

        PopulateEncodedValues(
            compressed_data,
            encoded_values,
            lookup_table,
            nbits,
            gz,
            gy,
            gx,
            data_entries
        )

        return compressed_data.tobytes()

    @staticmethod
    def encode(data, *args, **kwargs):

        return neuroglancer.compress(data, *args, **kwargs)

    @staticmethod
    def decode(data, *args, **kwargs):

        return neuroglancer.decompress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''Neuroglancer decompression
        '''

        data = np.fromstring(data, dtype=np.uint64)

        # get the size of the data
        az, ay, ax = data[0], data[1], data[2]
        bz, by, bx = (8, 64, 64)
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

        data_entries = 3
        for ie in range(0, nelements):
            header = long(data[data_entries])

            table_offsets[ie] = header >> 40
            nbits[ie] = (header << 24) >> 56
            values_offsets[ie] = (header << 32) >> 32

            data_entries += 1


        ###############################
        # DECOMPRESS ENTIRE IMAGE
        ###############################

        decompressed_data = DecodeNeuroglancer(
            data, table_offsets, nbits, values_offsets, data_entries
        )

        return decompressed_data
