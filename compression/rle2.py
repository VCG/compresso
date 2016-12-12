import numpy as np


class RLE2(object):

    @staticmethod
    def name():
        '''Name
        '''

        return 'RLE2'

    @staticmethod
    def encode(img_stack):
        '''Run-length encoding
        '''

        rle_img = []
        rle_len = 2#3  # For depth, height, and width

        # For every image on the stack
        # for i in range(img_stack.shape[0]):
            # Run-length encoding by line
        for j in range(img_stack.shape[0]):
            # x = img_stack[i][j, :]
            x = img_stack[j, :]
            pos, = np.where(np.diff(x) != 0)
            pos = np.concatenate(([0], pos + 1, [len(x)]))
            rle = [(b, x[a]) for (a, b) in zip(pos[:-1], pos[1:])]
            rle_len += len(rle) * 2
            rle_img.append(rle)

        rle_indices = np.zeros(rle_len/2 - 1, dtype=img_stack.dtype)
        rle_values = np.zeros(rle_len/2 -1, dtype=img_stack.dtype)


        # rle_img_np[0] = img_stack.shape[0]  # Depth
        # rle_img_np[1] = img_stack.shape[1]  # Height
        # rle_img_np[2] = img_stack.shape[2]  # Width
        index = 0

        for r in rle_img:
            for t in r:
                # rle_img_np[rle_img_np_index] = t[0]
                # rle_img_np[rle_img_np_index + 1] = t[1]
                rle_indices[index] = t[0]
                rle_values[index] = t[1]
                index += 1



        return (img_stack.shape), rle_indices, rle_values

    @staticmethod
    def decode(shape, indices, values):
        '''Run-length decoding
        '''

        # depth = rle_img[0]
        height = shape[0]
        width = shape[1]

        img_stack = np.zeros((height, width), dtype=values.dtype)

        y = 0
        x = 0
        for k in range(0, len(indices)):
            # print k
            end = indices[k]
            value = values[k]
            img_stack[y, x:end] = value
            x = end

            if end == width:
                x = 0
                y += 1

                # if y == height:
                #     z += 1
                #     y = 0

        return img_stack
