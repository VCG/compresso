import numpy as np


class RLEA(object):
    '''Advanced run-length encoding

    It stores segment IDs and run-lengths with different bit integers. This
    saves about 1/3 of space BEFORE compression. Unfortunately after
    compression this advantage is reverted.
    '''

    @staticmethod
    def name():
        '''Name
        '''

        return 'RLEA'

    @staticmethod
    def encode(img):
        '''Run-length encoding
        Ideas:
        - Separately encode segment IDs and run-lengths as run-lengths are
          likely to be much smaller, e.g., 1024.
        - Test whether lines or columns lead to a higher compression
        - Store segment IDs in a dictionary and use internal IDs that map to a
          segment ID. Similar to what Neuroglancer does
        '''

        rle_img = []
        rle_len = 0

        # For every image on the stack
        for i in range(img.shape[0]):
            # Run-length encoding by line
            for j in range(img.shape[1]):
                x = img[i][j, :]
                pos, = np.where(np.diff(x) != 0)
                pos = np.concatenate(([0], pos + 1, [len(x)]))
                rle = [(b, x[a]) for (a, b) in zip(pos[:-1], pos[1:])]
                rle_len += len(rle)
                rle_img.append(rle)

        rle_img_header = np.zeros(4, dtype=np.uint16)
        rle_img_header[0] = img.shape[0]  # Depth
        rle_img_header[1] = img.shape[1]  # Height
        rle_img_header[2] = img.shape[2]  # Width
        rle_img_header[3] = rle_len  # Length of the other two arrays

        # Run-lengths
        rle_img_rl = np.zeros(rle_len, dtype=np.uint16)

        # Segment IDs
        rle_img_si = np.zeros(rle_len, dtype=np.uint64)

        _index = 0

        for r in rle_img:
            for t in r:
                rle_img_rl[_index] = t[0]
                rle_img_si[_index] = t[1]
                _index += 1

        return (
            rle_img_header.tobytes() +
            rle_img_rl.tobytes() +
            rle_img_si.tobytes()
        )

    @staticmethod
    def decode(rle_img):
        '''Run-length decoding
        '''

        # 4 (values) * 16 (bit) / 8 (byte) = 8
        header = np.fromstring(rle_img[:8], dtype=np.uint16)

        depth = header[0]
        height = header[1]
        width = header[2]
        rle_len = header[3]

        rl = np.fromstring(rle_img[8: 8 + (rle_len * 2)], dtype=np.uint16)
        si = np.fromstring(rle_img[8 + (rle_len * 2):], dtype=np.uint64)

        out = np.zeros((depth, height, width), dtype=np.uint64)

        z = 0
        y = 0
        x = 0
        for k in range(rle_len):
            end = rl[k]
            value = si[k]
            out[z][y, x:end] = value
            x = end

            if end == width:
                x = 0
                y += 1

                if y == height:
                    z += 1
                    y = 0

        return out
