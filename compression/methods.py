import backports.lzma as lzma
import brotli
import bz2
import lzf
import lzo
import lzw
import numpy as np
import zlib
import zopfli.zlib as zopfli
# We might want ti check as well https://pypi.python.org/pypi/zstandard
import zstd

from boundary_encoding import boundary_encoding
from variable_encoding import variable_encoding
from jpeg import jpeg
from lz78 import lz78
from neuroglancer import neuroglancer
from png_compress import png_compress
from x264 import x264_compress


class BOCKWURST(object):
    @staticmethod
    def name():
        return 'Bockwurst'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''Bockwurst compression
        '''

        return boundary_encoding.compress(data, compress=True)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''Bockwurst decompression
        '''

        return boundary_encoding.decompress(data, compress=True)

    @staticmethod
    def decode(data, *args, **kwargs):
        '''Bockwurst decoding
        '''

        return boundary_encoding.decompress(data, compress=False)

    @staticmethod
    def encode(data, *args, **kwargs):
        '''Bockwurst encoding
        '''

        return boundary_encoding.compress(data, compress=False)


class BROTLI(object):

    @staticmethod
    def name():
        return 'Brotli'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''Brotli compression
        '''

        return brotli.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''Brotli decompression
        '''

        return brotli.decompress(data, *args, **kwargs)


class BZ2(object):

    @staticmethod
    def name():
        return 'BZip2'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''BZip2 compression
        '''

        return bz2.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''BZip2 decompression
        '''

        return bz2.decompress(data, *args, **kwargs)


class JPEG2000(object):

    @staticmethod
    def name():
        return 'JPEG2000'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''JPEG2000 compression
        '''

        return jpeg.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''JPEG2000 decompression
        '''

        return jpeg.decompress(data, *args, **kwargs)

    @staticmethod
    def encode(data, *args, **kwargs):
        '''JPEG2000 compression
        '''

        return jpeg.compress(data, *args, **kwargs)

    @staticmethod
    def decode(data, *args, **kwargs):
        '''JPEG2000 decompression
        '''

        return jpeg.decompress(data, *args, **kwargs)


class LZF(object):

    @staticmethod
    def name():
        return 'LZF'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''LZF compression
        '''

        return lzf.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''LZF decompression
        '''

        # TODO - hard coded in -> fix this
        return lzf.decompress(data, int(2048 * 2048 * 300 * 64))


class LZMA(object):

    @staticmethod
    def name():
        return 'LZMA'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''LZMA compression
        '''

        return lzma.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''LZMA decompression
        '''

        return lzma.decompress(data, *args, **kwargs)


class LZO(object):

    @staticmethod
    def name():
        return 'LZO'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''LZO compression
        '''

        return lzo.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''LZO decompression
        '''

        return lzo.decompress(data, *args, **kwargs)


class LZW(object):

    @staticmethod
    def name():
        return 'LZW'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''LZW compression
        '''

        if type(data) is np.ndarray:
            str_data = data.tobytes()
        elif type(data) is str:
            str_data = data
        else:
            raise ValueError('Data type not supported')

        return b''.join(lzw.compress(str_data))

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''LZW decompression
        '''

        return b''.join(lzw.decompress(data))


class LZ78(object):

    @staticmethod
    def name():
        return 'LZ78'

    @staticmethod
    def compress(data, *args, **kwargs):
        ''' LZ78 compression
        '''

        if type(data) is np.ndarray:
            str_data = data.tobytes()
        elif type(data) is str:
            str_data = data
        else:
            raise ValueError('Data type not supported')

        return lz78.compress(str_data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        ''' LZ78 decompression
        '''

        return lz78.decompress(data, *args, **kwargs)


class NC(object):

    @staticmethod
    def name():
        '''No Compression
        '''

        return 'NC'

    @staticmethod
    def compress(data):
        '''No encoding
        '''

        return data

    @staticmethod
    def decompress(data):
        '''No decoding
        '''

        return data


class NG(object):
    @staticmethod
    def name():
        return 'Neuroglancer'

    @staticmethod
    def encode(data, *args, **kwargs):
        '''Neuroglancer compression
        '''

        return neuroglancer.encode(data, *args, **kwargs)

    @staticmethod
    def decode(data, *args, **kwargs):
        '''Neuroglancer decompression
        '''

        return neuroglancer.decode(data, *args, **kwargs)

    @staticmethod
    def compress(data, *args, **kwargs):
        '''Neuroglancer compression
        '''

        return neuroglancer.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''Neuroglancer decompression
        '''

        return neuroglancer.decompress(data, *args, **kwargs)


class PNG(object):

    @staticmethod
    def name():
        return 'PNG'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''PNG compression
        '''

        return png_compress.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''PNG decompression
        '''

        return png_compress.decompress(data, *args, **kwargs)

    @staticmethod
    def encode(data, *args, **kwargs):
        '''PNG compression
        '''

        return png_compress.compress(data, *args, **kwargs)

    @staticmethod
    def decode(data, *args, **kwargs):
        '''PNG decompression
        '''

        return png_compress.decompress(data, *args, **kwargs)


class BOUNDARY_ENCODING(object):
    @staticmethod
    def name():
        return 'Boundary Encoding'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''Boundary Encoding compression
        '''

        return boundary_encoding.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''Boundary Encoding decompression
        '''

        return boundary_encoding.decompress(data, *args, **kwargs)


class VARIABLE_ENCODING(object):
    @staticmethod
    def name():
        return 'Variable Encoding'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''Variable Encoding compression
        '''

        return variable_encoding.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''Variable Encoding decompression
        '''

        return variable_encoding.decompress(data, *args, **kwargs)


class X264(object):

    @staticmethod
    def name():
        return 'X.264'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''X.264 compression
        '''

        return x264_compress.compress(data)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''Zlib decompression
        '''

        return x264_compress.decompress(data)

    @staticmethod
    def encode(data, *args, **kwargs):
        '''X.264 compression
        '''

        return x264_compress.compress(data)

    @staticmethod
    def decode(data, *args, **kwargs):
        '''Zlib decompression
        '''

        return x264_compress.decompress(data)


class ZLIB(object):

    @staticmethod
    def name():
        return 'Zlib'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''Zlib compression
        '''

        return zlib.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''Zlib decompression
        '''

        return zlib.decompress(data, *args, **kwargs)


class ZOPFLI(object):

    @staticmethod
    def name():
        return 'Zopfli'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''Zopfli compression
        '''

        return zopfli.compress(data, numiterations=5)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''Zopfli decompression
        '''

        return zlib.decompress(data, *args, **kwargs)


class ZSTD(object):

    @staticmethod
    def name():
        return 'ZStandard'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''Zstandard compression
        '''
        return zstd.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''Zstandard decompression
        '''

        return zstd.decompress(data, *args, **kwargs)
