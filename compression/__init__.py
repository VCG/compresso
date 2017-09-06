from util import Util
from rle import RLE
from methods import (
    # general purpose
    BROTLI,
    BZ2,
    LZ78,
    LZF,
    LZMA,
    LZO,
    LZW,
    ZLIB,
    ZSTD,    
    # image specific
    JPEG2000,
    PNG,
    # segmentation specific
    COMPRESSO,
    BOUNDARY_ENCODING,
    NEUROGLANCER,
    VARIABLE_ENCODING,    
    # video specific
    X264,
    # default
    NONE
)
