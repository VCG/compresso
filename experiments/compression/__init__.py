from util import Util
from compresso import compresso
from neuroglancer import neuroglancer
from methods import (
    # general purpose
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
    NEUROGLANCER,
    # video specific
    X264,
    # default
    NONE
)
