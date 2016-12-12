# Bockwurst: Efficient Compression of Segmentation Data For Connectomics

![Segmentations](/experiments/figures/cyl_logo_mri.png?raw=true)

> Recent advances in connectomics produce very large datasets with automatic labeling. The resulting segmentation volumes need compression for efficient storage and transfer. Such segmentation data consists of large low-frequency, high-bit regions with structured boundaries and is very different from conventional image data. As a result, general purpose compression tools do not properly exploit these characteristics. In this paper, we present Bockwurst, a new compression scheme for segmentation data that outperforms any existing method. Our method uses a sliding window approach to exploit redundancy across border regions in 2D and 3D. This enables efficient encoding of full segmentation volumes to a fraction of their original data size. We also study existing compression methods and provide a detailed evaluation on multiple connectomics datasets. To demonstrate generalizability, we include performance evaluation on a labeled brain MRI dataset.

## Requirements

- Python 2.7
- PIP
- virtualenv with virtualenvwrapper

## Setup

```bash
git clone https://github.com/rhoana/bockwurst && cd bockwurst
mkvirtualenv -a $(pwd) bockwurst
pip install -r requirements.txt
```

## Compress Your Segmentation Stack

In order to test Bockwurst on your own data simply use

```
import compression as C

# With LZ78
C.LZ78.compress(C.BOCKWURST(<NUMPY-ARRAY>, compress=False))

# With included LZMA compression
C.BOCKWURST.compress(<NUMPY-ARRAY>)
```

## Experiments

The experiments are defined in `experiments/`. Start Jupyter Notebooks (`jupyter notebook`) and open either of the four options:

- AC3
- CREMI
- CYL
- MRI

Make sure you specify the correct path to the data. The data itself can be found here:

- AC3: <http://www.openconnectomeproject.org/kasthuri11> _(Kasthuri et al. Saturated reconstruction of a volume of neocortex. Cell 2015.)_
- CREMI: <http://www.cremi.org>
- CYL: <http://www.openconnectomeproject.org/kasthuri11> _(Kasthuri et al. Saturated reconstruction of a volume of neocortex. Cell 2015.)_
- MRI: This is an unpublished dataset. Once it's available we will link it here.

### Results for Cylinder

**Compression Performance**

![Cylinder Compression Performance](/experiments/figures/cyl_performance.png?raw=true)

While run-length encoding provides the fastest compression and decompression speed, Bockwurst in combination with LZ78 outperforms any other compression scheme.

**Encoding Performance**

![Cylinder Compressed Bytes](/experiments/figures/cyl_encoding_performance.png?raw=true)

The same is true for encoding only. Here we distinguish between compression and encoding to highlight system requirements for random access, which is only available before compression.
