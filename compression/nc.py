class NC(object):

    @staticmethod
    def name():
        '''No Encoding
        '''

        return 'NC'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''No encoding
        '''

        return data

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''No decoding
        '''

        return data
