import numpy as np
from numba import jit


class RLE2D(object):

    @staticmethod
    def name():
        '''Name
        '''

        return 'RLE 2D'

    @staticmethod
    @jit(nopython=True)
    def find_first(item, vec):
        '''Return the index of the first occurence of item in vec'''
        for i in xrange(len(vec)):
            if item == vec[i]:
                return i
        return -1

    @staticmethod
    def get_arr(img, j, dir):
        '''Get array by direction, which can either be 0=line or 1=row
        '''

        return img[j, :] if dir == 0 else img[:, j]

    @staticmethod
    def encode(img, delta_intra=False, delta_inter=False, dir=0, bl=False):
        '''2D Run-length encoding

        delta_intra = offset within one consecutive border
        delta_inter = offset among borders, e.g. border begins at prev + 2
        '''

        rle_header_size = 2  # height and width

        # Check if by line or by row is more efficient
        diff_by_row = 0
        for j in range(img.shape[0]):
            pos, = np.where(np.diff(img[j, :]) != 0)
            diff_by_row += pos.shape[0]

        diff_by_col = 0
        for j in range(img.shape[1]):
            pos, = np.where(np.diff(img[:, j]) != 0)
            diff_by_col += pos.shape[0]

        if dir is -1:
            dir = 0 if diff_by_row <= diff_by_col else 1

        rl_dtype = img.dtype

        if delta_intra:
            rl_dtype = np.int64

        if bl:
            rl_dtype = np.uint16

        rl = np.array([], dtype=rl_dtype)
        ids = np.array([], dtype=rl_dtype)

        last_rl = rl
        last_ids = ids

        # Run-length encoding by line
        for j in range(img.shape[dir]):
            x = RLE2D.get_arr(img, j, dir)
            if bl:
                pos, = np.where(np.diff(x))
            else:
                pos, = np.where(np.diff(x) != 0)

            pos = np.concatenate(([0], pos + 1, [len(x)])).astype(rl_dtype)

            curr_ids = x[pos[:-1]]
            ids = np.concatenate(
                (ids, curr_ids),
                axis=0
            )

            curr_rl = pos[1:]
            rl = np.concatenate(
                (rl, pos[1:]),
                axis=0
            )

            if delta_inter and pos.shape[0] > 2:
                rl[-(pos.shape[0] - 2):] = np.diff(pos[1:])
                curr_rl = rl[-(pos.shape[0] - 1):]

            if delta_intra:
                k = 0
                last_ids_len = last_ids.shape[0]
                for curr_id in curr_ids:
                    offset = RLE2D.find_first(
                        curr_id,
                        last_ids[
                            max(k - 3, 0):
                            min(k + 3, last_ids_len)
                        ]
                    )

                    if offset >= 0:
                        # ID found
                        offset -= min(k, 3)
                        p = last_ids_len - (k + offset)

                        # assert curr_id == last_ids[k + offset]

                        rl[-p] = last_rl[k + offset] - curr_rl[k]

                    k += 1

            last_rl = curr_rl
            last_ids = curr_ids

        rle_img_np = np.zeros(
            rle_header_size + rl.shape[0] * 2, dtype=rl_dtype
        )
        rle_img_np[0] = img.shape[0]  # Depth
        rle_img_np[1] = img.shape[1]  # Height
        rle_img_np_index = rle_header_size

        for i in np.arange(rl.shape[0]):
            rle_img_np[rle_img_np_index] = rl[i]  # Run-lengths
            rle_img_np[rle_img_np_index + 1] = ids[i]  # IDs
            rle_img_np_index += 2

        return rle_img_np

    @staticmethod
    def decode(rle_img):
        '''2D Run-length decoding
        '''

        height = rle_img[0]
        width = rle_img[1]

        img = np.zeros((height, width), dtype=rle_img.dtype)

        y = 0
        x = 0
        for k in range(2, len(rle_img), 2):
            end = rle_img[k]
            value = rle_img[k + 1]
            img[y, x:end] = value
            x = end

            if end == width:
                x = 0
                y += 1

        return img

    # @staticmethod
    # def delta()
