# Compresso: Efficient Compression of Segmentation Data For Connectomics (PyPI edition)

[![PyPI version](https://badge.fury.io/py/compresso.svg)](https://badge.fury.io/py/compresso)
[![Paper](https://img.shields.io/badge/paper-accepted-red.svg?colorB=f52ef0)](https://vcg.seas.harvard.edu/publications/compresso-efficient-compression-of-segmentation-data-for-connectomics)
[![MICCAI](https://img.shields.io/badge/presentation-MICCAI%202017-red.svg?colorB=135f89)](http://www.miccai2017.org/schedule)
[![doi](https://img.shields.io/badge/used%20by-rhoana-red.svg?colorB=2bf55b)](http://www.rhoana.org)

![Segmentations](/banner.png?raw=true)

```python
import compresso 
import numpy as np 

labels = np.array(...)
compressed_labels = compresso.compress(labels) # 3d numpy array -> compressed bytes
reconstituted_labels = compresso.decompress(compressed_labels) # compressed bytes -> 3d numpy array
```

*NOTE: This is an unofficial packaging of the work by Matejek et al. which can be found here: https://github.com/VCG/compresso*

> Recent advances in segmentation methods for connectomics and biomedical imaging produce very large datasets with labels that assign object classes to image pixels. The resulting label volumes are bigger than the raw image data and need compression for efficient storage and transfer. General-purpose compression methods are less effective because the label data consists of large low-frequency regions with structured boundaries unlike natural image data. We present Compresso, a new compression scheme for label data that outperforms existing approaches by using a sliding window to exploit redundancy across border regions in 2D and 3D. We compare our method to existing compression schemes and provide a detailed evaluation on eleven biomedical and image segmentation datasets. Our method provides a factor of 600-2200x compression for label volumes, with running times suitable for practice.

**Paper**: Matejek _et al._, "Compresso: Efficient Compression of Segmentation Data For Connectomics", Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2017, 10-14. \[[CITE](https://scholar.google.com/scholar?q=Compresso%3A+Efficient+Compression+of+Segmentation+Data+For+Connectomics) | [PDF](https://vcg.seas.harvard.edu/publications/compresso-efficient-compression-of-segmentation-data-for-connectomics/paper)\]

## Requirements

- Python 2.7, 3.5+

## Setup

```bash
pip install compresso
```

### Results From the Paper

**Compression Performance**

![Compression Performance of Connectomics Datasets](/experiments/figures/compression-performance.png?raw=true)

Compression ratios of general-purpose compression methods combined with Compresso and Neuroglancer. Compresso paired with LZMA yields the best compression ratios for all connectomics datasets (left) and in average (four out of five) for the others (right).
