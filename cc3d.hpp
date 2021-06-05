/*
 * Connected Components for 2D images. 

 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton University
 * Date: August 2018 - June 2019
 *
 * ----
 * LICENSE
 * 
 * This is a special reduced feature version of cc3d 
 * that includes only the logic needed for CCL 4-connected.
 * cc3d is ordinarily licensed as GPL v3. Get the full
 * version of cc3d here: 
 * 
 * https://github.com/seung-lab/connected-components-3d
 *
 *  The MIT License (MIT)
 *
 *  Copyright © 2020 William Silversmith
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a 
 *  copy of this software and associated documentation files (the “Software”), 
 *  to deal in the Software without restriction, including without limitation 
 *  the rights to use, copy, modify, merge, publish, distribute, sublicense, 
 *  and/or sell copies of the Software, and to permit persons to whom the 
 *  Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all 
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, 
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR 
 * THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * ----
 */

#ifndef CC3D_SPECIAL_4_HPP
#define CC3D_SPECIAL_4_HPP 

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <stdexcept>

namespace cc3d {

static size_t _dummy_N;

template <typename T>
class DisjointSet {
public:
  T *ids;
  size_t length;

  DisjointSet () {
    length = 65536; // 2^16, some "reasonable" starting size
    ids = new T[length]();
    if (!ids) { 
      throw std::runtime_error("Failed to allocate memory for the Union-Find datastructure for connected components.");
    }
  }

  DisjointSet (size_t len) {
    length = len;
    ids = new T[length]();
    if (!ids) { 
      throw std::runtime_error("Failed to allocate memory for the Union-Find datastructure for connected components.");
    }
  }

  DisjointSet (const DisjointSet &cpy) {
    length = cpy.length;
    ids = new T[length]();
    if (!ids) { 
      throw std::runtime_error("Failed to allocate memory for the Union-Find datastructure for connected components.");
    }

    for (int i = 0; i < length; i++) {
      ids[i] = cpy.ids[i];
    }
  }

  ~DisjointSet () {
    delete []ids;
  }

  T root (T n) {
    T i = ids[n];
    while (i != ids[i]) {
      ids[i] = ids[ids[i]]; // path compression
      i = ids[i];
    }

    return i;
  }

  bool find (T p, T q) {
    return root(p) == root(q);
  }

  void add(T p) {
    if (p >= length) {
      printf("Connected Components Error: Label %d cannot be mapped to union-find array of length %lu.\n", p, length);
      throw "maximum length exception";
    }

    if (ids[p] == 0) {
      ids[p] = p;
    }
  }

  void unify (T p, T q) {
    if (p == q) {
      return;
    }

    T i = root(p);
    T j = root(q);

    if (i == 0) {
      add(p);
      i = p;
    }

    if (j == 0) {
      add(q);
      j = q;
    }

    ids[i] = j;
  }

  void print() {
    for (int i = 0; i < 15; i++) {
      printf("%d, ", ids[i]);
    }
    printf("\n");
  }

  // would be easy to write remove. 
  // Will be O(n).
};

// This is the second raster pass of the two pass algorithm family.
// The input array (output_labels) has been assigned provisional 
// labels and this resolves them into their final labels. We
// modify this pass to also ensure that the output labels are
// numbered from 1 sequentially.
template <typename OUT = uint32_t>
OUT* relabel(
    OUT* out_labels, const int64_t voxels,
    const int64_t num_labels, DisjointSet<uint32_t> &equivalences,
    size_t &N = _dummy_N
  ) {

  if (num_labels <= 1) {
    N = num_labels;
    return out_labels;
  }

  OUT label;
  OUT* renumber = new OUT[num_labels + 1]();
  OUT next_label = 1;

  for (int64_t i = 1; i <= num_labels; i++) {
    label = equivalences.root(i);
    if (renumber[label] == 0) {
      renumber[label] = next_label;
      renumber[i] = next_label;
      next_label++;
    }
    else {
      renumber[i] = renumber[label];
    }
  }

  // Raster Scan 2: Write final labels based on equivalences
  N = next_label - 1;
  if (N < static_cast<size_t>(num_labels)) {
    for (int64_t loc = 0; loc < voxels; loc++) {
      out_labels[loc] = renumber[out_labels[loc]];
    }
  }

  delete[] renumber;

  return out_labels;
}

template <typename T, typename OUT = uint32_t>
OUT* connected_components2d_4(
    T* in_labels, 
    const int64_t sx, const int64_t sy, const int64_t sz,
    size_t max_labels, OUT *out_labels = NULL, 
    size_t &N = _dummy_N
  ) {

  const int64_t sxy = sx * sy;
  const int64_t voxels = sx * sy * sz;

  max_labels++;
  max_labels = std::min(max_labels, static_cast<size_t>(voxels) + 1); // + 1L for an array with no zeros
  max_labels = std::min(max_labels, static_cast<size_t>(std::numeric_limits<OUT>::max()));


  DisjointSet<uint32_t> equivalences(max_labels);

  if (out_labels == NULL) {
    out_labels = new OUT[voxels]();
  }
  if (!out_labels) { 
    throw std::runtime_error("Failed to allocate out_labels memory for connected components.");
  }
    
  /*
    Layout of forward pass mask. 
    A is the current location.
    D C 
    B A 
  */

  const int64_t A = 0;
  const int64_t B = -1;
  const int64_t C = -sx;
  const int64_t D = -1-sx;

  int64_t loc = 0;
  OUT next_label = 0;

  // Raster Scan 1: Set temporary labels and 
  // record equivalences in a disjoint set.

  T cur = 0;
  for (int64_t z = 0; z < sz; z++) {
    for (int64_t y = 0; y < sy; y++) {
      for (int64_t x = 0; x < sx; x++) {
        loc = x + sx * y + sxy * z;
        cur = in_labels[loc];

        if (cur == 0) {
          continue;
        }

        if (x > 0 && cur == in_labels[loc + B]) {
          out_labels[loc + A] = out_labels[loc + B];
          if (y > 0 && cur != in_labels[loc + D] && cur == in_labels[loc + C]) {
            equivalences.unify(out_labels[loc + A], out_labels[loc + C]);
          }
        }
        else if (y > 0 && cur == in_labels[loc + C]) {
          out_labels[loc + A] = out_labels[loc + C];
        }
        else {
          next_label++;
          out_labels[loc + A] = next_label;
          equivalences.add(out_labels[loc + A]);
        }
      }
    }
  }

  return relabel<OUT>(out_labels, voxels, next_label, equivalences, N);
}

template <typename OUT = uint64_t>
OUT* connected_components2d(
  bool* in_labels, 
  const int64_t sx, const int64_t sy, const int64_t sz,
  size_t &N = _dummy_N
) {

  const int64_t sxy = sx * sy;
  const int64_t voxels = sxy * sz;

  const size_t max_labels = static_cast<size_t>((sxy + 2) / 2 * (sz + 2));
  OUT* out_labels = new OUT[voxels]();

  for (int64_t z = 0; z < sz; z++) {
    size_t tmp_N = 0;
    connected_components2d_4<bool, OUT>(
      (in_labels + sxy * z), sx, sy, 1, 
      max_labels, (out_labels + sxy * z), tmp_N
    );
    N += tmp_N;
  }

  return out_labels;
}



};



#endif
