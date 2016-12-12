import numpy as np


class HashNums(object):

    @staticmethod
    def get_reverse_dict(arr):
        '''Create a reverse dictionary for an array
        '''

        return {v: i for i, v in enumerate(arr)}

    @staticmethod
    def get_unique_nums(nums):
        '''Get the list of unique numbers.
        '''

        u_nums, idx = np.unique(nums, return_index=True)

        return u_nums[np.argsort(idx)]

    @staticmethod
    def hash(nums):
        '''Creates a hash table for unique numbers and replaces all occureces
        in the given array.
        '''

        u_nums = HashNums.get_unique_nums(nums)
        d = HashNums.get_reverse_dict(u_nums)

        return (u_nums, np.vectorize(lambda x: d[x])(np.copy(nums)))

    @staticmethod
    def unhash(u_nums, h_nums):
        '''Unhashes a hashed array of number given the hash table of unique
        numbers.
        '''

        return np.vectorize(lambda x: u_nums[x])(np.copy(h_nums))
