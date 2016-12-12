import numpy as np


class RLED(object):

    @staticmethod
    def name():
        '''Name
        '''

        return 'RLE-D'

    @staticmethod
    def get_reverse_dict(arr):
        '''Create a reverse dictionary for an array
        '''

        return {v: i for i, v in enumerate(arr)}

    @staticmethod
    def get_unique_ids(img_stack):
        '''Get the list of unique IDs per slice
        '''

        uids, idx = np.unique(img_stack, return_index=True)

        return uids[np.argsort(idx)]

    @staticmethod
    def encode(img_stack, diff=True, sep=False):
        '''Run-length encoding
        Idea:
        - Store IDs in a dict to use lower bit integers.
        '''

        unique_ids = RLED.get_unique_ids(img_stack)
        d = RLED.get_reverse_dict(unique_ids)

        # rle_img = np.array([], dtype=np.uint32)
        rle_lens = np.array([], dtype=np.int16)
        rle_ids = np.array([], dtype=np.int16)
        rle_len = 4  # For depth, height, width, and number of unique IDs

        # For every image on the stack
        for i in range(img_stack.shape[0]):
            # Run-length encoding by line
            for j in range(img_stack.shape[1]):
                x = img_stack[i][j, :]
                pos, = np.where(np.diff(x) != 0)
                pos = np.concatenate(([0], pos + 1, [len(x)]))
                # rle = np.array(
                #     [list((l, d[x[k]])) for (k, l) in zip(pos[:-1], pos[1:])]
                # ).flatten()
                rle_lens = np.append(rle_lens, pos[1:])
                if diff:
                    rle_ids = np.append(
                        rle_ids,
                        np.append(
                            pos[0],
                            np.diff(map(lambda k: d[x[k]], pos[:-1]))
                        )
                    )
                else:
                    rle_ids = np.append(
                        rle_ids,
                        map(lambda k: d[x[k]], pos[:-1])
                    )
                # rle_img = np.append(rle_img, rle)

        rle_len += len(rle_ids) * 2 + len(unique_ids)

        rle_img_np = np.zeros(rle_len, dtype=np.int16)
        rle_img_np[0] = img_stack.shape[0]  # Depth
        rle_img_np[1] = img_stack.shape[1]  # Height
        rle_img_np[2] = img_stack.shape[2]  # Width
        rle_img_np[3] = len(unique_ids)  # Number of unique IDs
        rle_img_np_index = 4

        for id in np.nditer(unique_ids):
            rle_img_np[rle_img_np_index] = id
            rle_img_np_index += 1

        # Extract run lengths
        for rl in rle_lens:
            rle_img_np[rle_img_np_index] = rl
            rle_img_np_index += 1

        # Extract IDs
        for i in rle_ids:
            rle_img_np[rle_img_np_index] = i
            rle_img_np_index += 1

        return rle_img_np

    @staticmethod
    def decode(rle_img):
        '''Run-length decoding
        '''

        depth = rle_img[0]
        height = rle_img[1]
        width = rle_img[2]
        num_unique_ids = np.uint32(rle_img[3])

        img_stack = np.zeros((depth, height, width), dtype=rle_img.dtype)

        unique_ids = np.zeros(num_unique_ids, dtype=rle_img.dtype)

        # Get list unique IDs
        for k in range(4, num_unique_ids + 4):
            unique_ids[k - 4] = rle_img[k]

        z = 0
        y = 0
        x = 0

        rl_start = 4 + num_unique_ids
        rl_block_size = len(rle_img[rl_start:])
        rl_half_block_size = (rl_block_size / 2)

        for k in range(rl_start, rl_start + rl_half_block_size):
            end = rle_img[k]
            value = unique_ids[rle_img[k + rl_half_block_size]]
            img_stack[z][y, x:end] = value
            x = end

            if end == width:
                x = 0
                y += 1

                if y == height:
                    z += 1
                    y = 0

        return img_stack
