import numpy as np
from numba import jit


class RLEV2(object):

    @staticmethod
    def name():
        '''Name
        '''

        return 'RLE v2'

    @staticmethod
    def encode(img_stack, bl=False):
        '''2D and 3D run-length encoding v2
        '''

        if img_stack.dtype == np.bool:
            rl, = np.where(img_stack.flatten())

            rl = np.concatenate(
                (
                    rl[0:1],
                    np.diff(rl)
                ),
                axis=0
            )

            header = np.array(
                [
                    1,  # Is Boolean
                    img_stack.shape[0],  # Depth
                    img_stack.shape[1],  # Height
                    img_stack.shape[2]  # Width
                ],
                dtype=np.uint16
            )

            return np.concatenate(
                (
                    header,
                    rl
                ),
                axis=0
            )

        else:
            rl, = np.where(np.diff(img_stack.flatten()))
            rl = np.concatenate(
                (
                    rl[0:1],
                    np.diff(rl)
                ),
                axis=0
            )

            header = np.array(
                [
                    1,  # Is Boolean
                    img_stack.shape[0],  # Depth
                    img_stack.shape[1],  # Height
                    img_stack.shape[2]  # Width
                ],
                dtype=np.uint16
            )

            return np.concatenate(
                (
                    header,
                    rl
                ),
                axis=0
            )

    @staticmethod
    def decode(rle_img):
        '''2D and 3D run-length decoding v2
        '''

        header = rle_img[:4]
        is_bool = np.bool(header[0])
        depth = rle_img[1]
        height = rle_img[2]
        width = rle_img[3]

        rl = rle_img[4:]

        out_dtype = np.bool if is_bool else rle_img.dtype

        z = 0
        y = 0
        x = 0

        if depth > 1:
            out = np.zeros(depth * height * width, dtype=out_dtype)

            if not is_bool:
                for k in range(3, len(rle_img), 2):
                    end = rle_img[k]
                    value = rle_img[k + 1]
                    out[z][y, x:end] = value
                    x = end

                    if end == width:
                        x = 0
                        y += 1

                        if y == height:
                            z += 1
                            y = 0

        else:
            out = np.zeros(height * width, dtype=out_dtype)

            if not is_bool:
                for k in range(2, len(rle_img), 2):
                    end = rle_img[k]
                    value = rle_img[k + 1]
                    out[y, x:end] = value
                    x = end

                    if end == width:
                        x = 0
                        y += 1

        if is_bool:
            for delta_x in rl:
                x += delta_x
                out[x] = True

        if depth > 1:
            out = out.reshape(depth, height, width)
        else:
            out = out.reshape(height, width)

        return out
