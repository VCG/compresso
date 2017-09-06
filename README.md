# Compresso: Efficient Compression of Segmentation Data For Connectomics

[![Paper](https://img.shields.io/badge/paper-accepted-red.svg?colorB=f52ef0)](https://vcg.seas.harvard.edu/publications/compresso-efficient-compression-of-segmentation-data-for-connectomics)
[![MICCAI](https://img.shields.io/badge/presentation-MICCAI%202017-red.svg?colorB=135f89)](http://www.miccai2017.org/schedule)
[![doi](https://img.shields.io/badge/used%20by-rhoana-red.svg?colorB=2bf55b)](http://www.rhoana.org)
[![Bockwurst](https://img.shields.io/badge/bockwurst-tasty-red.svg?colorB=ff9f3a)](#compress-your-segmentation-stack)

![Segmentations](/banner.png?raw=true)

> Recent advances in connectomics produce very large datasets with automatic labeling. The resulting segmentation volumes need compression for efficient storage and transfer. Such segmentation data consists of large low-frequency, high-bit regions with structured boundaries and is very different from conventional image data. As a result, general purpose compression tools do not properly exploit these characteristics. In this paper, we present Compresso, a new compression scheme for segmentation data that outperforms any existing method. Our method uses a sliding window approach to exploit redundancy across border regions in 2D and 3D. This enables efficient encoding of full segmentation volumes to a fraction of their original data size. We also study existing compression methods and provide a detailed evaluation on multiple connectomics datasets. To demonstrate generalizability, we include performance evaluation on a labeled brain MRI dataset.

## Requirements

- Python 2.7
- PIP
- virtualenv with virtualenvwrapper

## Setup

```bash
git clone https://github.com/rhoana/compresso && cd compresso
mkvirtualenv -a $(pwd) bockwurst
pip install -r requirements.txt
```

## Compress Your Segmentation Stack

In order to test Compresso on your own data simply use

```
import compression as C

# With LZ78
C.LZ78.compress(C.BOCKWURST(<NUMPY-ARRAY>, compress=False))

# With included LZMA compression
C.BOCKWURST.compress(<NUMPY-ARRAY>)
```

## Experiments

```
experiments/run.py BOCKWURST LZ78 ac3 -r 1 -s 1 -d '/<PATH>/<TO>/<DATA>' -b 
```

Usage:

```
usage: run.py [-h] [--directory PATH] [--runs NUM] [--slices NUM]
              [--bockwurst] [--verbose]
              encoding compression dataset

positional arguments:
  encoding              name of encoding scheme
  compression           name of compression scheme
  dataset               name of data set

optional arguments:
  -h, --help            show this help message and exit
  --directory PATH, -d PATH
                        path to data directory
  --runs NUM, -r NUM    number of runs (default: 1)
  --slices NUM, -s NUM  number of slices per dataset (default: -1 (all))
  --bockwurst, -b       show me some bockwurst (default: False)
  --verbose, -v         print progress (default: False) 
```


Make sure the data sets are located in `experiments/data` or specify the location. specify the correct path to the data. The data itself can be found here:

- AC3: <http://www.openconnectomeproject.org/kasthuri11> _(Kasthuri et al. Saturated reconstruction of a volume of neocortex. Cell 2015.)_
- CREMI: <http://www.cremi.org>
- CYL: <http://www.openconnectomeproject.org/kasthuri11> _(Kasthuri et al. Saturated reconstruction of a volume of neocortex. Cell 2015.)_
- MRI: This is an unpublished dataset. Once it's available we will link it here.

### Results for Cylinder

**Compression Performance**

![Cylinder Compression Performance](/experiments/figures/cyl_performance.png?raw=true)

While run-length encoding provides the fastest compression and decompression speed, Comopresso in combination with LZ78 outperforms any other compression scheme.

**Encoding Performance**

![Cylinder Compressed Bytes](/experiments/figures/cyl_encoding_performance.png?raw=true)

The same is true for encoding only. Here we distinguish between compression and encoding to highlight system requirements for random access, which is only available before compression.
