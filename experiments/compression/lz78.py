class lz78(object):

    @staticmethod
    def name():
        return 'LZ78'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''LZ78 compression
        '''

        d, word = {0: ''}, 0
        dyn_d = (
            lambda d, key: d.get(key) or d.__setitem__(key, len(d)) or 0
        )

        return [
            token for
            char in
            data for
            token in
            [(word, char)] for
            word in [dyn_d(d, token)] if not word
        ] + [(word, '')]

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''LZ78 decompression
        '''

        d, j = {0: ''}, ''.join
        dyn_d = (
            lambda d, value: d.__setitem__(len(d), value) or value
        )

        return j([dyn_d(d, d[codeword] + char) for (codeword, char) in data])
