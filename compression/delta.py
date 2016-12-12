import numpy as np


class Delta(object):

    @staticmethod
    def name():
        '''Name
        '''

        return 'Delta'

    @staticmethod
    def encode(array):
        '''Delta Encoding along last axis. This only works properly if the
        values are sorted.
        '''

        out_array = array.copy()

        delta_array = np.diff(array.astype(np.int64), axis=-1)
        delta_array[delta_array < 0] = 0

        out_array[:, 1:] = delta_array

        return out_array

    @staticmethod
    def decode(array):
        '''Run-length decoding
        '''

        out_array = array.copy().astype(np.int64)
        out_array = np.cumsum(array, axis=-1)  # ohyea
        out_array[array == 0] = 0

        return out_array
