import numpy as np

# general-purpose compression imports
import brotli
import bz2
import lzf
import backports.lzma as lzma
import lzo
import zlib
import zstd

# image compression imports
from jpeg import jpeg
from _png import _png

# video compression imports
from x264 import x264

# cython files
from compresso import compresso
from neuroglancer import neuroglancer
from lz78 import lz78

# segmentation specific files
from boundary_encoding import boundary_encoding
from variable_encoding import variable_encoding

class NONE(object):
    @staticmethod
    def name():
        return 'None'

    @staticmethod
    def compress(data):
        return data

    @staticmethod
    def decompress(data):
        return data

    
#############################
### SEGMENTATION SPECIFIC ###
#############################

class COMPRESSO(object):
    @staticmethod
    def name():
        return compresso.compresso.name()

    @staticmethod
    def decompress(data, *args, **kwargs):
        return compresso.compresso.decompress(data, *args, **kwargs)

    @staticmethod
    def compress(data, *args, **kwargs):
        return compresso.compresso.compress(data, *args, **kwargs)


class BOUNDARY_ENCODING(object):
    @staticmethod
    def name():
        return 'Boundary Encoding'

    @staticmethod
    def compress(data, *args, **kwargs):
        return boundary_encoding.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        return boundary_encoding.decompress(data, *args, **kwargs)


class NEUROGLANCER(object):
    @staticmethod
    def name():
        return 'Neuroglancer'

    @staticmethod
    def compress(data, *args, **kwargs):
        return neuroglancer.neuroglancer.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        return neuroglancer.neuroglancer.decompress(data, *args, **kwargs)


class VARIABLE_ENCODING(object):
    @staticmethod
    def name():
        return 'Variable Encoding'

    @staticmethod
    def compress(data, *args, **kwargs):
        return variable_encoding.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        return variable_encoding.decompress(data, *args, **kwargs)


#######################
### GENERAL PURPOSE ###
#######################


class BROTLI(object):

    @staticmethod
    def name():
        return 'Brotli'

    @staticmethod
    def compress(data, *args, **kwargs):
        return brotli.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        return brotli.decompress(data, *args, **kwargs)


class BZ2(object):

    @staticmethod
    def name():
        return 'BZip2'

    @staticmethod
    def compress(data, *args, **kwargs):
        return bz2.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        return bz2.decompress(data, *args, **kwargs)


class LZ78(object):

    @staticmethod
    def name():
        return 'LZ78'

    @staticmethod
    def compress(data, *args, **kwargs):
        if type(data) is np.ndarray:
            str_data = data.tobytes()
        elif type(data) is str:
            str_data = data
        else:
            raise ValueError('Data type not supported')

        dictionary = lz78.lz78.compress(str_data, *args, **kwargs)

        array = np.zeros(len(dictionary), dtype=np.uint32)
        retry = False
        for ie, entry in enumerate(dictionary):
            if entry[1] == '':
                if (entry[0] >= 2**24):
                    retry = True
                    break
                array[ie] = (entry[0] << 8)
            else:
                if (entry[0] >= 2**24):
                    retry =  True
                    break
                array[ie] = (entry[0] << 8) + ord(entry[1])
                
        if not retry: return array
        else:
            array = np.zeros(len(dictionary), dtype=np.uint64)
            for ie, entry in enumerate(dictionary):
                if entry[1] == '':
                    array[ie] = (entry[0] << 8)
                else:
                    array[ie] = (entry[0] << 8) + ord(entry[1])
            return array

    @staticmethod
    def decompress(data, *args, **kwargs):
        dictionary = list()

        for ie, entry in enumerate(data):
            int_value = long(entry) / (2**8)
            if ie == data.size - 1:
                char_value = ''
            else:
                char_value = chr(long(entry) % (2**8))

            dictionary.append((int_value, char_value))

        return lz78.lz78.decompress(dictionary, *args, **kwargs)


class LZF(object):

    @staticmethod
    def name():
        return 'LZF'

    @staticmethod
    def compress(data, *args, **kwargs):
        return lzf.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        return lzf.decompress(data, args[0])


class LZMA(object):

    @staticmethod
    def name():
        return 'LZMA'

    @staticmethod
    def compress(data, *args, **kwargs):
        return lzma.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        return lzma.decompress(data, *args, **kwargs)


class LZO(object):

    @staticmethod
    def name():
        return 'LZO'

    @staticmethod
    def compress(data, *args, **kwargs):
        return lzo.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        return lzo.decompress(data, *args, **kwargs)


class LZW(object):

    @staticmethod
    def name():
        return 'LZW'

    @staticmethod
    def compress(data, *args, **kwargs):        
        if type(data) is np.ndarray:
            str_data = data.tobytes()
        elif type(data) is str:
            str_data = data
        else:
            raise ValueError('Data type not supported')

        # create an empty dictionary
        dict_size = 2**8
        dictionary = dict((chr(i), i) for i in xrange(dict_size))
        
        w = ''
        result = []
        for c in str_data:
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                dictionary[wc] = dict_size
                dict_size += 1
                w = c

        if w:
            result.append(dictionary[w])

        return np.array(result, dtype=np.uint32)

    @staticmethod
    def decompress(data, *args, **kwargs):
        from cStringIO import StringIO

        data = list(data)

        dict_size = 256
        dictionary = dict((i, chr(i)) for i in xrange(dict_size))
        
        result = StringIO()
        w = chr(data.pop(0))
        result.write(w)

        for k in data:
            if k in dictionary:
                entry = dictionary[k]
            elif k == dict_size:
                entry = w + w[0]
            else:
                raise ValueError('Bad compressed k: %s' % k)
            result.write(entry)

            # Add w+entry[0] to the dictionary
            dictionary[dict_size] = str(w + entry[0])
            dict_size += 1

            w = entry

        return result.getvalue()

class ZLIB(object):

    @staticmethod
    def name():
        return 'Zlib'

    @staticmethod
    def compress(data, *args, **kwargs):
        return zlib.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        return zlib.decompress(data, *args, **kwargs)


class ZSTD(object):

    @staticmethod
    def name():
        return 'ZStandard'

    @staticmethod
    def compress(data, *args, **kwargs):
        return zstd.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        return zstd.decompress(data, *args, **kwargs)


#########################
### IMAGE COMPRESSION ###
#########################

class JPEG2000(object):

    @staticmethod
    def name():
        return 'JPEG2000'

    @staticmethod
    def compress(data, *args, **kwargs):
        return jpeg.compress(data, *args, **kwargs)

    @staticmethod
    def decompress(data, *args, **kwargs):
        return jpeg.decompress(data, *args, **kwargs)


class PNG(object):

    @staticmethod
    def name():
        return 'PNG'

    @staticmethod
    def compress(data, *args, **kwargs):
        return _png.compress(data)

    @staticmethod
    def decompress(data, *args, **kwargs):
        return _png.decompress(data)


#########################
### VIDEO COMPRESSION ###
#########################

class X264(object):

    @staticmethod
    def name():
        return 'X.264'

    @staticmethod
    def compress(data, *args, **kwargs):
        return x264.compress(data)

    @staticmethod
    def decompress(data, *args, **kwargs):
        return x264.decompress(data)

