import numpy as np

class Pack(object):

    @staticmethod
    def name():
        '''Name
        '''

        return 'Pack'

    @staticmethod
    def encode(array, delimiter=0, include_delimiter=True):
        '''Pack an array along X which means skipping everything after a certain delimiter.
        '''
        original_length = array.shape[-1]

        required_length = 0

        add_one = 0
        if include_delimiter:
            add_one = 1

        for y in range(array.shape[-2]):

            first_delimiter_appearance = np.where(array[y] == delimiter)
            if len(first_delimiter_appearance[0]) == 0:
                # hit the end of the row
                required_length += array.shape[-1] + add_one

            else:
                first_delimiter_appearance = first_delimiter_appearance[0][0]

                required_length += first_delimiter_appearance + add_one

        out_array = np.zeros((required_length), dtype=array.dtype)

        out_array_pointer = 0
        for y in range(array.shape[-2]):

            first_delimiter_appearance = np.where(array[y] == delimiter)
            if len(first_delimiter_appearance[0]) == 0:
                # hit the end of the row
                cur_end = array.shape[-1] + add_one
                values = array[y,0:cur_end]
                if include_delimiter:
                    values = np.zeros((values.shape[0]+1))
                    values[0:array[y,0:cur_end].shape[0]] = array[y,0:cur_end]

            else:
                first_delimiter_appearance = first_delimiter_appearance[0][0]
                cur_end = first_delimiter_appearance + add_one
                values = array[y,0:cur_end]

            # print out_array_pointer
            out_array[out_array_pointer:out_array_pointer+cur_end] = values
            # the +1 is to store one delimiter
            out_array_pointer += cur_end

        return original_length, out_array

    @staticmethod
    def decode(original_length, array, delimiter=0, include_delimiter=True):
        '''Unpack a 1D array.
        '''

        # count the lines
        row_count = len(np.where(array == delimiter)[0])
        # print 'rowcount',row_count
        out_array = np.zeros((row_count, original_length), dtype=array.dtype)
        # print original_length
        add_one = 0
        if include_delimiter:
            add_one = 1

        array_pointer = 0
        for y in range(row_count):

            stop_byte = np.where(array == delimiter)[0][y]


            # print y,stop_byte,array_pointer#+add_one-array_pointer, array_pointer,stop_byte+add_one
            out_array[y,0:stop_byte+add_one-array_pointer] = array[array_pointer:stop_byte+add_one]
            # print out_array[y]

            array_pointer = stop_byte+add_one+1

        return out_array
