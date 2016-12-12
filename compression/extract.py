import numpy as np
from util import Util


class Extract(object):

    @staticmethod
    def name():
        '''Name
        '''

        return 'Extract'

    @staticmethod
    def encode(array):
        '''Extract values which need more bits from an array and store as list.
        Returns an uint8 array with uint16 and uint32 and uint64 lists.
        '''

        out_array = array.copy()

        ui8 = np.iinfo(np.uint8).max
        ui16 = np.iinfo(np.uint16).max
        ui32 = np.iinfo(np.uint32).max
        ui64 = np.iinfo(np.uint64).max

        values_64bit = np.where(out_array >= ui32)
        a64 = np.zeros((len(values_64bit[0])), dtype=np.uint64)

        # grab all 64 bit values from the array
        for i in range(len(values_64bit[0])):

            y = values_64bit[0][i]
            if array.ndim == 2:
                x = values_64bit[1][i]

                a64[i] = out_array[y, x]
                out_array[y, x] = ui32
            elif array.ndim == 1:
                a64[i] = out_array[y]
                out_array[y] = ui32

        values_32bit = np.where(out_array >= ui16)
        a32 = np.zeros((len(values_32bit[0])), dtype=np.uint32)

        for i in range(len(values_32bit[0])):

            y = values_32bit[0][i]
            if array.ndim == 2:
                x = values_32bit[1][i]

                a32[i] = out_array[y, x]
                out_array[y, x] = ui16
            elif array.ndim == 1:
                a32[i] = out_array[y]
                out_array[y] = ui16

        values_16bit = np.where(out_array >= ui8)
        a16 = np.zeros((len(values_16bit[0])), dtype=np.uint16)

        for i in range(len(values_16bit[0])):

            y = values_16bit[0][i]
            if array.ndim == 2:
                x = values_16bit[1][i]

                a16[i] = out_array[y, x]
                out_array[y, x] = ui8
            elif array.ndim == 1:
                a16[i] = out_array[y]
                out_array[y] = ui8

        return Util.to_best_type(out_array), a16, a32, a64

    @staticmethod
    def decode(array, a16, a32, a64):
        '''Unpack a 1D array.
        '''
        out_array = array.copy().astype(np.uint64)

        ui8 = np.iinfo(np.uint8(10))
        ui8 = ui8.max
        ui16 = np.iinfo(np.uint16(10))
        ui16 = ui16.max
        ui32 = np.iinfo(np.uint32(10))
        ui32 = ui32.max
        ui64 = np.iinfo(np.uint64(10))
        ui64 = ui64.max

        values_larger_than_8bit = np.where(array >= ui8)

        count64 = 0
        count32 = 0
        count16 = 0

        for i in range(len(values_larger_than_8bit[0])):

            y = values_larger_than_8bit[0][i]
            x = values_larger_than_8bit[1][i]

            if a16[count16] == ui32:
                # this is even larger than 16
                if a32[count32] == ui64:
                    # this is even larger than 32
                    out_array[y, x] = a64[count64]
                    count64 += 1
                else:
                    out_array[y, x] = a32[count32]
                    count32 += 1
            else:
                out_array[y, x] = a16[count16]
                count16 += 1

        return Util.to_best_type(out_array)
