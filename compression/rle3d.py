import numpy as np
from numba import jit


class RLE3D(object):

    @staticmethod
    def name():
        '''Name
        '''

        return 'RLE 3D'

    @staticmethod
    @jit(nopython=True)
    def find_first(item, vec):
        '''Return the index of the first occurence of item in vec'''
        for i in xrange(len(vec)):
            if item == vec[i]:
                return i
        return -1

    @staticmethod
    def encode(img_stack, delta_3d=False, delta_2d_inter=False):
        '''3D run-length encoding
        '''

        rle_header_size = 3  # For depth, height, and width

        out_dtype = img_stack.dtype

        if delta_3d:
            out_dtype = np.int64

        rl = np.array([], dtype=out_dtype)
        ids = np.array([], dtype=out_dtype)

        last_rl = rl
        last_ids = ids

        # For every image on the stack
        for i in range(img_stack.shape[0]):
            curr_rl = np.array([], dtype=out_dtype)
            curr_ids = np.array([], dtype=out_dtype)

            # Run-length encoding by line
            for j in range(img_stack.shape[1]):
                x = img_stack[i][j, :]
                pos, = np.where(np.diff(x) != 0)
                pos = np.concatenate(([0], pos + 1, [len(x)]))

                curr_ids = np.concatenate(
                    (curr_ids, x[pos[:-1]]),
                    axis=0
                )

                curr_rl = np.concatenate(
                    (curr_rl, pos[1:]),
                    axis=0
                )

                if delta_2d_inter and pos.shape[0] > 2:
                    curr_rl[-(pos.shape[0] - 2):] = np.diff(pos[1:])

            if delta_3d:
                curr_rl_final = np.copy(curr_rl)
                wurst = 0
                k = 0
                last_ids_len = last_ids.shape[0]
                for curr_id in curr_ids:
                    offset = RLE3D.find_first(
                        curr_id,
                        last_ids[
                            max(k - 3, 0):
                            min(k + 3, last_ids_len)
                        ]
                    )

                    if offset >= 0:
                        wurst += 1
                        # ID found
                        offset -= min(k, 3)

                        # assert curr_id == last_ids[k + offset]

                        curr_rl_final[k] -= last_rl[k + offset]

                    k += 1

            else:
                curr_rl_final = curr_rl

            # Append current slice
            ids = np.concatenate((ids, curr_ids), axis=0)
            rl = np.concatenate((rl, curr_rl_final), axis=0)

            # Store last slice
            last_rl = curr_rl
            last_ids = curr_ids

        rle_img_np = np.zeros(
            rle_header_size + rl.shape[0] * 2, dtype=out_dtype
        )
        rle_img_np[0] = img_stack.shape[0]  # Depth
        rle_img_np[1] = img_stack.shape[1]  # Height
        rle_img_np[2] = img_stack.shape[2]  # Width
        rle_img_np_index = rle_header_size

        for i in np.arange(rl.shape[0]):
            rle_img_np[rle_img_np_index] = rl[i]  # Run-lengths
            rle_img_np[rle_img_np_index + 1] = ids[i]  # IDs
            rle_img_np_index += 2

        if delta_2d_inter:
            rle_img_np[rle_header_size::2] = np.concatenate(
                (
                    rle_img_np[rle_header_size:rle_header_size + 1],
                    np.diff(rle_img_np[rle_header_size::2])
                ),
                axis=0
            )

        return rle_img_np

    @staticmethod
    def decode(rle_img):
        '''3D run-length decoding
        '''

        depth = rle_img[0]
        height = rle_img[1]
        width = rle_img[2]

        img_stack = np.zeros((depth, height, width), dtype=rle_img.dtype)

        z = 0
        y = 0
        x = 0
        for k in range(3, len(rle_img), 2):
            end = rle_img[k]
            value = rle_img[k + 1]
            img_stack[z][y, x:end] = value
            x = end

            if end == width:
                x = 0
                y += 1

                if y == height:
                    z += 1
                    y = 0

        return img_stack

    @staticmethod
    def decode_ids(rle_img):
        '''3D run-length decoding of IDs only
        '''

        depth = rle_img[0]
        height = rle_img[1]
        width = rle_img[2]

        img_stack = np.zeros((depth, height, width), dtype=rle_img.dtype)

        z = 0
        y = 0
        x = 0

        largest_x = 0
        for k in range(3, len(rle_img), 2):
            end = rle_img[k]
            value = rle_img[k + 1]
            img_stack[z][y, x] = value
            x += 1

            if end == width:
                largest_x = max(x, largest_x)
                x = 0
                y += 1

                if y == height:
                    z += 1
                    y = 0

        return img_stack[:, :, 0:largest_x]
