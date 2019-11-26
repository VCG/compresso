import numpy as np
import compresso

DTYPES = [
  np.int8, np.int16, np.int32, np.int64,
  np.uint8, np.uint16, np.uint32, np.uint64,
]

def test_compress_decompress():
  for dtype in DTYPES:
    labels = np.random.randint(0, 25, size=(100, 200, 150))
    reconstituted = compresso.decompress(compresso.compress(labels))
    assert np.all(labels == reconstituted)
