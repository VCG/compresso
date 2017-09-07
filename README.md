# Compresso: Efficient Compression of Segmentation Data For Connectomics

[![Paper](https://img.shields.io/badge/paper-accepted-red.svg?colorB=f52ef0)](https://vcg.seas.harvard.edu/publications/compresso-efficient-compression-of-segmentation-data-for-connectomics)
[![MICCAI](https://img.shields.io/badge/presentation-MICCAI%202017-red.svg?colorB=135f89)](http://www.miccai2017.org/schedule)
[![doi](https://img.shields.io/badge/used%20by-rhoana-red.svg?colorB=2bf55b)](http://www.rhoana.org)

![Segmentations](/banner.png?raw=true)

> Recent advances in segmentation methods for connectomics and biomedical imaging produce very large datasets with labels that assign object classes to image pixels. The resulting label volumes are bigger than the raw image data and need compression for efficient storage and transfer. General-purpose compression methods are less effective because the label data consists of large low-frequency regions with structured boundaries unlike natural image data. We present Compresso, a new compression scheme for label data that outperforms existing approaches by using a sliding window to exploit redundancy across border regions in 2D and 3D. We compare our method to existing compression schemes and provide a detailed evaluation on eleven biomedical and image segmentation datasets. Our method provides a factor of 600-2200x compression for label volumes, with running times suitable for practice.

**Paper**: Matejek _et al._, "Compresso: Efficient Compression of Segmentation Data For Connectomics", Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2017, 10-14. \[[CITE](https://scholar.google.com/scholar?q=Compresso%3A+Efficient+Compression+of+Segmentation+Data+For+Connectomics) | [PDF](https://vcg.seas.harvard.edu/publications/compresso-efficient-compression-of-segmentation-data-for-connectomics/paper)\]

## Requirements

- Python 2.7
- conda

## Setup

```bash
git clone https://github.com/vcg/compresso && cd compresso
conda create -n compresso_env --file requirements.txt
# for Compresso scheme as presented in MICCAI
cd experiments/compression/compresso; python setup.py build_ext --inplace
# to run the neuroglancer compression scheme
cd ../neuroglancer; python setup.py build_ext --inplace
# for Compresso v2 that is under development
cd ../../src/python; python setup.py build_ext --inplace
```

## Compress Segmentation Stacks

There are two versions of Compresso in this repository. Under the src folder there is an updated c++ and python version that extends on the Compresso scheme presented in MICCAI. This algorithm, among other things, implements bit-packing to further improve compression results.

The compression scheme in `experiments/compression/compresso` follows the MICCAI paper exactly. 

## Compress Your Segmentation Stack

In order to test Compresso on your own data simply use

```
import compression as C
```

# With LZMA
C.LZMA.compress(C.COMPRESSO.compress(<NUMPY-3D-ARRAY>))

## Experiments

```
# the dataset must be in hdf5 format.
experiments/run.py COMPRESSO LZMA ac3 -r 1 -s 1 -d '/<PATH>/<TO>/<DATA>'
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


Make sure the data sets are located in `~/compresso/data/` or specify the location. The data from the paper can be found here:

- AC3: <http://www.openconnectomeproject.org/kasthuri11> _(Kasthuri et al. Saturated reconstruction of a volume of neocortex. Cell 2015.)_
- CREMI: <http://www.cremi.org>
- CYL: <http://www.openconnectomeproject.org/kasthuri11> _(Kasthuri et al. Saturated reconstruction of a volume of neocortex. Cell 2015.)_
- SPL Brain Atlas: <http://www.spl.harvard.edu/publications/item/view/2037> _(Halle M., Talos I-F., Jakab M., Makris N., Meier D., Wald L., Fischl B., Kikinis R. Multi-modality MRI-based Atlas of the Brain. SPL 2017 Jan)_
- SPL Knee Atlas: <http://www.spl.harvard.edu/publications/item/view/2037> _(Richolt J.A., Jakab M., Kikinis R. SPL Knee Atlas. SPL 2015 Sep)_
- SPL Abdominal Atlas: <http://www.spl.harvard.edu/publications/item/view/1918> _(Talos I-F., Jakab M., Kikinis R. SPL Abdominal Atlas. SPL 2015 Sep)_
- BSD500: <https://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html> _(Contour Detection and Hierarchical Image Segmentation P. Arbelaez, M. Maire, C. Fowlkes and J. Malik. IEEE TPAMI, Vol. 33, No. 5, pp. 898-916, May 2011.)_
- VOC2012: <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/> _(Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A., The PASCAL Visual Object Classes Challenge 2012 (VOC2012) Results)_

### Results From the Paper

**Compression Performance**

![Compression Performance of Connectomics Datasets](/experiments/figures/compression-performance.png?raw=true)

Compression ratios of general-purpose compression methods combined with Compresso and Neuroglancer. Compresso paired with LZMA yields the best compression ratios for all connectomics datasets (left) and in average (four out of five) for the others (right).
