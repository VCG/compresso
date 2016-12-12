import numpy as np
from numba import jit


class B2N(object):

    @staticmethod
    def name():
        '''Name
        '''

        return 'Boolean 2 number'

    @staticmethod
    @jit(nopython=True)
    def get_base_2(num, offset=0):
        powers = []
        i = 1
        j = 0

        while i <= num:
            if i & num:
                powers.append(j + offset)
            i <<= 1
            j += 1

        return powers

    @staticmethod
    def encode(img, window=[4, 4]):
        '''Encode boolean array as an array of numbers
        '''

        if img.ndim == 3:
            is3D = 1
            depth = img.shape[0]
            height = img.shape[1]
            width = img.shape[2]
        elif img.ndim == 2:
            is3D = 0
            depth = 0
            height = img.shape[0]
            width = img.shape[1]
        else:
            raise ValueError('Only 2D and 3D are supported')

        win_height = window[0]
        win_width = window[1]

        header = np.zeros(6, dtype=np.uint16)
        header[0] = is3D
        header[1] = depth
        header[2] = height
        header[3] = width
        header[4] = win_height
        header[5] = win_width

        if win_height * win_width == 64:
            out_dtype = np.uint64
        elif win_height * win_width == 32:
            out_dtype = np.uint32
        elif win_height * win_width == 16:
            out_dtype = np.uint16
        elif win_height * win_width == 8:
            out_dtype = np.uint8
        else:
            raise ValueError(
                'Wrong window size. The size must multiply to 8, 16, 32, or '
                '64.'
            )

        len = img.flatten().shape[0] / (win_height * win_width)

        borders = np.zeros(len, dtype=out_dtype)

        i = 0

        if is3D:
            for z in np.arange(depth):
                for y in range(0, height, win_height):
                    diff_lines = img[z, y:y + win_height, :]

                    for x in range(0, width, win_width):
                        pos, = np.where(
                            diff_lines[:, x:x + win_width].flatten()
                        )
                        num = out_dtype(0)

                        for p in pos:
                            num += out_dtype(2**p)

                        borders[i] = num
                        i += 1
        else:
            for y in range(0, height, win_height):
                diff_lines = img[y:y + win_height, :]

                for x in range(0, width, win_width):
                    pos, = np.where(diff_lines[:, x:x + win_width].flatten())
                    num = out_dtype(0)

                    for p in pos:
                        num += out_dtype(2**p)

                    borders[i] = num
                    i += 1

        return header.tobytes() + borders.tobytes()

    @staticmethod
    def decode(num_arr):
        '''Decode number array as boolean array
        '''

        header = np.fromstring(num_arr[:12], dtype=np.uint16)
        is3D = np.bool(header[0])
        depth = np.uint32(header[1])
        height = np.uint32(header[2])
        width = np.uint32(header[3])
        win_height = np.uint32(header[4])
        win_width = np.uint32(header[5])
        win_length = win_height * win_width

        if win_height * win_width == 64:
            border_dtype = np.uint64
        elif win_height * win_width == 32:
            border_dtype = np.uint32
        elif win_height * win_width == 16:
            border_dtype = np.uint16
        elif win_height * win_width == 8:
            border_dtype = np.uint8
        else:
            raise ValueError(
                'Wrong window size. The size must multiply to 8, 16, 32, or '
                '64.'
            )

        borders = np.fromstring(num_arr[12:], dtype=border_dtype)

        if is3D:
            border_img = np.zeros((depth, height, width), dtype=np.bool)
        else:
            border_img = np.zeros((height, width), dtype=np.bool)

        z = 0
        x = 0
        y = 0

        if is3D:
            for border in borders:
                win = np.zeros(win_length, dtype=np.bool)
                win[B2N.get_base_2(border)] = True

                border_img[z, y:y + win_height, x:x + win_width] = win.reshape(
                    win_height, win_width
                )

                x += win_width

                if x == width:
                    x = 0
                    y += win_height

                    if y == height:
                        z += 1
                        y = 0
        else:
            for border in borders:
                win = np.zeros(win_length, dtype=np.bool)
                win[B2N.get_base_2(border)] = True

                border_img[y:y + win_height, x:x + win_width] = win.reshape(
                    win_height, win_width
                )

                x += win_width

                if x == width:
                    x = 0
                    y += win_height

        return border_img
