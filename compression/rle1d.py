import numpy as np


class RLE1D(object):

    @staticmethod
    def name():
        '''Name
        '''

        return 'RLE 1D'

    @staticmethod
    def encode(arr):
        '''2D Run-length encoding
        '''

        rle_arr = []
        rle_len = 0
        i = rle_len

        pos, = np.where(np.diff(arr) != 0)
        pos = np.concatenate(([0], pos + 1, [len(arr)]))
        rle = [(b, arr[a]) for (a, b) in zip(pos[:-1], pos[1:])]
        rle_len += len(rle)
        rle_arr.append(rle)

        # Run-lengths
        rle_arr_rl = np.zeros(rle_len, dtype=np.uint16)

        # Segment IDs
        rle_arr_si = np.zeros(rle_len, dtype=np.uint64)

        for r in rle_arr:
            for t in r:
                rle_arr_rl[i] = t[0]
                rle_arr_si[i] = t[1]
                i += 1

        rle_arr_rl = np.concatenate(
            (rle_arr_rl[0:1], np.diff(rle_arr_rl)),
            axis=0
        )

        return np.concatenate((rle_arr_rl, rle_arr_si), axis=0)

    @staticmethod
    def decode(rle_arr):
        '''2D Run-length decoding
        '''

        rl_len = len(rle_arr) / 2

        arr_len = np.sum(rle_arr[0:rl_len])

        arr = np.zeros(arr_len, dtype=rle_arr.dtype)

        from_i = 0
        for i in range(rl_len):
            to_i = from_i + rle_arr[i]
            arr[from_i:to_i] = rle_arr[rl_len + i]
            from_i = to_i

        return arr
