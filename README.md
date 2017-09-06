# Compresso: Efficient Compression of Segmentation Data For Connectomics

[![Paper](https://img.shields.io/badge/paper-accepted-red.svg?colorB=f52ef0)](https://vcg.seas.harvard.edu/publications/compresso-efficient-compression-of-segmentation-data-for-connectomics)
[![MICCAI](https://img.shields.io/badge/presentation-MICCAI%202017-red.svg?colorB=135f89)](http://www.miccai2017.org/schedule)
[![doi](https://img.shields.io/badge/used%20by-rhoana-red.svg?colorB=2bf55b)](http://www.rhoana.org)

![Segmentations](/banner.png?raw=true)

> Recent advances in segmentation methods for connectomics and biomedical imaging produce very large datasets with labels that assign object classes to image pixels. The resulting label volumes are bigger than the raw image data and need compression for efficient storage and transfer. General-purpose compression methods are less effective because the label data consists of large low-frequency regions with structured boundaries unlike natural image data. We present Compresso, a new compression scheme for label data that outperforms existing approaches by using a sliding window to exploit redundancy across border regions in 2D and 3D. We compare our method to existing compression schemes and provide a detailed evaluation on eleven biomedical and image segmentation datasets. Our method provides a factor of 600-2200x compression for label volumes, with running times suitable for practice.

**Paper**: Matejek _et al._, "Compresso: Efficient Compression of Segmentation Data For Connectomics", Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2017, 10-14. \[[CITE](https://scholar.google.com/scholar?q=Compresso%3A+Efficient+Compression+of+Segmentation+Data+For+Connectomics) | [PDF](https://vcg.seas.harvard.edu/publications/compresso-efficient-compression-of-segmentation-data-for-connectomics/paper)\]

## Requirements

- Python 2.7
- PIP
- virtualenv with virtualenvwrapper

## Setup

```bash
git clone https://github.com/vcg/compresso && cd compresso
mkvirtualenv -a $(pwd) compresso
pip install -r requirements.txt
```

## Compress Your Segmentation Stack

In order to test Compresso on your own data simply use

```
import compression as C
```

# With LZMA
C.LZMA.compress(C.COMPRESSO(<NUMPY-ARRAY>, compress=False))

## Experiments

```
experiments/run.py COMPRESSO LZMA ac3 -r 1 -s 1 -d '/<PATH>/<TO>/<DATA>' -b 
```

Usage:

```
usage: run.py [-h] [--directory PATH] [--runs NUM] [--slices NUM]
              [--verbose]
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
  --verbose, -v         print progress (default: False) 
```


Make sure the data sets are located in `experiments/data` or specify the location. specify the correct path to the data. The data itself can be found here:

- AC3: <http://www.openconnectomeproject.org/kasthuri11> _(Kasthuri et al. Saturated reconstruction of a volume of neocortex. Cell 2015.)_
- CREMI: <http://www.cremi.org>
- CYL: <http://www.openconnectomeproject.org/kasthuri11> _(Kasthuri et al. Saturated reconstruction of a volume of neocortex. Cell 2015.)_
- MRI: This is an unpublished dataset. Once it's available we will link it here.

### Results From the Paper

**Compression Performance**

![Compression Performance of Connectomics Datasets](/experiments/figures/compression-performance.png?raw=true)

Compression ratios of general-purpose compression methods combined with Compresso and Neuroglancer. Compresso paired with LZMA yields the best compres- sion ratios for all connectomics datasets (left) and in average (four out of five) for the others (right).
