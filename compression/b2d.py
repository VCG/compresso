import numpy as np
from collections import OrderedDict
from numba import jit
from zigzag import zigzag


class B2D(object):

    @staticmethod
    def name():
        '''Name
        '''

        return 'Border 2D'

    @staticmethod
    def add_or_extend(vx, vy, s_id, x, y):
        try:
            vx[s_id] += [x]
            vy[s_id] += [y]
        except Exception:
            vx[s_id] = [x]
            vy[s_id] = [y]

        return (vx, vy)

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
    def get_only_borders(img):
        img_borders = np.zeros(img.shape, dtype=bool)

        for y in np.arange(img.shape[0]):
            line = img[y, :]
            img_borders[y, :-1] = np.diff(line) != 0

        return img_borders

    @staticmethod
    def get_borders_stack(img, window_size=[4, 4]):
        '''Get only the borders of an image
        '''

        header = np.zeros(5, dtype=np.uint16)
        header[0] = img.shape[0]  # Depth
        header[1] = img.shape[0]  # Height
        header[2] = img.shape[1]  # Width
        header[3] = window_size[0]  # Window Height
        header[4] = window_size[1]  # Widow Width

        win_height = window_size[0]
        win_width = window_size[1]

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
        diff_lines = np.zeros((win_height, img.shape[1]), dtype=np.bool)

        i = 0
        for z in np.arange(img.shape[0]):
            for y in range(0, img.shape[1], win_height):
                # Get diff of whole line
                diff_lines[:, :-1] = np.diff(img[z, y:y + win_height, :]) != 0

                for x in range(0, img.shape[1], win_width):
                    pos, = np.where(diff_lines[:, x:x + win_width].flatten())
                    num = out_dtype(0)

                    for p in pos:
                        num += out_dtype(2**p)

                    borders[i] = num
                    i += 1

        return header.tobytes() + borders.tobytes()

    @staticmethod
    def get_borders(img, window_size=[4, 4]):
        '''Get only the borders of an image
        '''

        header = np.zeros(4, dtype=np.uint16)
        header[0] = img.shape[0]  # Height
        header[1] = img.shape[1]  # Width
        header[2] = window_size[0]  # Window Height
        header[3] = window_size[1]  # Widow Width

        win_height = window_size[0]
        win_width = window_size[1]

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
        diff_lines = np.zeros((win_height, img.shape[1]), dtype=np.bool)

        i = 0
        for y in range(0, img.shape[0], win_height):
            # Get diff of whole line
            diff_lines[:, :-1] = np.diff(img[y:y + win_height, :]) != 0

            for x in range(0, img.shape[1], win_width):
                pos, = np.where(diff_lines[:, x:x + win_width].flatten())
                num = 0

                for p in pos:
                    num += 2**p

                borders[i] = num
                i += 1

        return header.tobytes() + borders.tobytes()

    @staticmethod
    def resolve_borders(b_border_img):
        '''Resolve borders to get back the original border image
        '''

        header = np.fromstring(b_border_img[:8], dtype=np.uint16)
        height = np.uint32(header[0])
        width = np.uint32(header[1])
        win_height = np.uint32(header[2])
        win_width = np.uint32(header[3])
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

        borders = np.fromstring(b_border_img[8:], dtype=border_dtype)

        border_img = np.zeros((height, width), dtype=np.bool)

        x = 0
        y = 0
        for border in borders:
            win = np.zeros(win_length, dtype=np.bool)
            win[B2D.get_base_2(border)] = True

            border_img[y:y + win_height, x:x + win_width] = win.reshape(
                win_height, win_width
            )

            x += win_width

            if x == width:
                x = 0
                y += win_height

        return border_img

    @staticmethod
    def encode(img, zz=True):
        '''2D segment border encoding
        '''

        vx = OrderedDict()
        vy = OrderedDict()

        vec_len = 0

        # Run-length encoding by line
        for y in range(img.shape[0]):
            line = img[y, :]
            borders, = np.where(np.diff(line) != 0)

            for x in borders:
                vec_len += 1
                vx, vy = B2D.add_or_extend(vx, vy, line[x], x, y)

        # Just a sanity check for now
        assert len(vx.keys()) == len(vy.keys())

        keys = np.array(vx.keys(), dtype=np.int64)

        header = np.zeros(4, dtype=np.int64)
        header[0] = img.shape[0]  # Height
        header[1] = img.shape[1]  # Width
        header[2] = len(keys)  # Number of keys
        header[3] = vec_len  # Lenght of the X and Y vector

        start_len = np.zeros(len(keys), dtype=np.int64)

        # First step
        start_len[0] = 0  # Start x

        for i in range(1, len(keys)):
            start_len[i] = len(vx[keys[i - 1]])  # Start x

        x = np.zeros(vec_len, dtype=np.int64)
        y = np.zeros(vec_len, dtype=np.int64)

        last_len = 0

        for i, idx in enumerate(vx):
            next_len = last_len + len(vx[idx])
            x[last_len] = vx[idx][0]
            x[last_len + 1:next_len] = np.diff(np.array(vx[idx]))
            y[last_len] = vy[idx][0]
            y[last_len + 1:next_len] = np.diff(np.array(vy[idx]))
            last_len = next_len

        if zz:
            # Zigzagify that vector
            x = zigzag.encode(x).astype(np.int64)
            y = zigzag.encode(y).astype(np.int64)

        return np.concatenate(
            (
                header,
                keys,
                start_len,
                x,
                y
            ),
            axis=0
        )

    @staticmethod
    def decode(enc_img):
        '''2D segment border decoding
        '''

        header = enc_img[0:4]

        height = header[0]
        width = header[1]
        num_keys = header[2]
        vec_len = header[3]

        keys = enc_img[4:4 + num_keys]
        starts = enc_img[4 + num_keys:4 + int(num_keys * 2)]
        x = bac3[(4 + int(num_keys * 2)):int(-1 * vec_len)]
        y = bac3[int(-1 * vec_len):]

        img = np.zeros((height, width), dtype=np.uint64)

        # for key in keys:

        return enc_img
