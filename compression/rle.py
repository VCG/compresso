import numpy as np
from rle1d import RLE1D
from rle2d import RLE2D
from rle3d import RLE3D


class RLE(object):

    @staticmethod
    def name():
        '''Name
        '''

        return 'RLE'

    @staticmethod
    def encode(data):
        '''Run-length encoding
        '''

        if data.ndim == 1:
            return np.insert(RLE1D.encode(data), 0, 1)

        if data.ndim == 2:
            return np.insert(RLE2D.encode(data), 0, 2)

        if data.ndim == 3:
            return np.insert(RLE3D.encode(data), 0, 3)

        raise ValueError('RLE only supports 1, 2, or 3 dimensions')

    @staticmethod
    def decode(rle_img):
        '''Run-length decoding
        '''

        if rle_img[0] == 1:
            return RLE1D.decode(rle_img[1:])

        if rle_img[0] == 2:
            return RLE2D.decode(rle_img[1:])

        if rle_img[0] == 3:
            return RLE3D.decode(rle_img[1:])

        raise ValueError('RLE only supports 1, 2, or 3 dimensions')
