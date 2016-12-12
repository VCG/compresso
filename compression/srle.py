import numpy as np
from delta import Delta
from pack import Pack
from rle import RLE
from util import Util


class SRLE(object):

    @staticmethod
    def name():
        '''Name
        '''

        return 'SRLE'

    @staticmethod
    def encode(img):
        '''Run-length encoding with separate structures
        '''
        if img.ndim == 3:
            raise ValueError(
                'Sorry dude, SRLE currently supports 2D encoding only'
            )

        rle_slice1 = RLE.encode(img)

        max_length = 0
        # slice1_prefix=rle_slice1[0]
        slice1_height = rle_slice1[0]
        slice1_width = rle_slice1[1]

        data_offset = 3
        prev_row_length = 0
        row_counter = 0
        # print slice1_prefix, slice1_width, slice1_height
        while row_counter < slice1_height:
            data_offset = data_offset + prev_row_length

            end_of_row = np.where(
                rle_slice1[data_offset:] == slice1_width
            )[0][0]

            current_row = rle_slice1[data_offset:data_offset + end_of_row + 2]

            prev_row_length = len(current_row)
            max_length = max(max_length, prev_row_length)

            row_counter += 1

        # now we can store as a new compressed array
        rle_slice1_compressed = np.zeros(
            (row_counter, max_length), dtype=rle_slice1.dtype
        )
        data_offset = 2
        row_counter = 0
        prev_row_length = 0
        while row_counter < slice1_height:
            data_offset = data_offset + prev_row_length

            end_of_row = np.where(
                rle_slice1[data_offset:] == slice1_width
            )[0][0]

            current_row = rle_slice1[data_offset:data_offset + end_of_row + 2]
            prev_row_length = len(current_row)

            rle_slice1_compressed[row_counter, 0:len(current_row)] = \
                current_row
            row_counter += 1

        indices = rle_slice1_compressed[:, 0::2].copy(order='C')

        #
        #
        # delta encoding for indices
        delta_indices = Delta.encode(indices)
        delta_indices = Util.to_best_type(delta_indices)

        data = rle_slice1_compressed[:, 1::2].copy(order='C')

        indices = Util.to_best_type(indices)
        data = Util.to_best_type(data)

        # returns metainfo, indices, data
        # return rle_slice1[0:3], indices, delta_indices, data
        return Util.to_best_type(rle_slice1[0:2]), delta_indices, indices, data

    @staticmethod
    def decode(shape, delta_indices, data):
        '''Run-length decoding
        '''
        # depth = 1
        height = shape[0]
        width = shape[1]

        indices = Delta.decode(delta_indices)

        p_indices = Pack.encode(indices, include_delimiter=False)[1]
        p_data = Pack.encode(data, include_delimiter=False)[1]

        full_length = 2 + len(p_indices) + len(p_data)

        out_array = np.zeros((full_length), dtype=data.dtype)

        out_array[0] = height
        out_array[1] = width

        out_array[2::2] = p_indices
        out_array[3::2] = p_data

        return RLE.decode(out_array)
