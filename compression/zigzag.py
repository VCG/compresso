import numpy as np


class zigzag(object):

    @staticmethod
    def name():
        '''Name
        '''

        return 'ZigZag'

    @staticmethod
    def get_non_neg_num(num):
        '''Gets the positive number according to the zigzag encoding.

        For example:
         0 => 0
         1 => 1
        -1 => 2
         2 => 3
        -2 => 4
        etc.
        '''

        return abs(num) * 2 - (1 if np.sign(num) > 0 else 0)

    @staticmethod
    def get_original_num(num):
        '''Get the original number.

        For example:
        0 => 0
        1 => 1
        2 => -1
        3 => 2
        4 => -2
        etc.
        '''

        m = num % 2

        return (num // 2 + m) * (-1 if m == 0 else 1)

    @staticmethod
    def encode(array):
        '''Decode an numpy array of positive and negative integers as zigzag
        lists.
        '''

        zigzagify = np.vectorize(zigzag.get_non_neg_num)

        return zigzagify(array).astype(np.uint64)

    @staticmethod
    def decode(array):
        '''Decode a zigzag numpy array
        '''

        revertify = np.vectorize(zigzag.get_original_num)

        return revertify(array.astype(np.int64))
