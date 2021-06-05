#ifndef __COMPRESSO_HXX__
#define __COMPRESSO_HXX__

#include <unordered_map>
#include <set>
#include <limits>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include "cc3d.hpp"

namespace compresso {

// little endian serialization of integers to chars
// returns bytes written
inline size_t itoc(uint8_t x, std::vector<unsigned char> &buf, size_t idx) {
	buf[idx] = x;
	return 1;
}

inline size_t itoc(uint16_t x, std::vector<unsigned char> &buf, size_t idx) {
	buf[idx + 0] = x & 0xFF;
	buf[idx + 1] = (x >> 8) & 0xFF;
	return 2;
}

inline size_t itoc(uint32_t x, std::vector<unsigned char> &buf, size_t idx) {
	buf[idx + 0] = x & 0xFF;
	buf[idx + 1] = (x >> 8) & 0xFF;
	buf[idx + 2] = (x >> 16) & 0xFF;
	buf[idx + 3] = (x >> 24) & 0xFF;
	return 4;
}

inline size_t itoc(uint64_t x, std::vector<unsigned char> &buf, size_t idx) {
	buf[idx + 0] = x & 0xFF;
	buf[idx + 1] = (x >> 8) & 0xFF;
	buf[idx + 2] = (x >> 16) & 0xFF;
	buf[idx + 3] = (x >> 24) & 0xFF;
	buf[idx + 4] = (x >> 32) & 0xFF;
	buf[idx + 5] = (x >> 40) & 0xFF;
	buf[idx + 6] = (x >> 48) & 0xFF;
	buf[idx + 7] = (x >> 56) & 0xFF;
	return 8;
}

/* Header: 
 *   'cpso'            : magic number (4 bytes)
 *   format version    : unsigned integer (1 byte) 
 *   data width        : unsigned integer (1 byte) (1: uint8, ... 8: uint64)
 *   sx, sy, sz        : size of each dimension (2 bytes x3)
 *   xstep,ystep,zstep : size of each grid (1 byte x 3) (typical values: 4, 8)
 *   id_size          : number of uniq labels (u64) (could be one per voxel)
 *   value_size       : number of values (u32)
 *   location_size    : number of locations (u64)
 */
struct CompressoHeader {
public:
	static constexpr size_t header_size{35};

	static constexpr char magic[]{ "cpso" }; 
	static constexpr uint8_t format_version{0};
	uint8_t data_width;
	uint16_t sx;
	uint16_t sy;
	uint16_t sz;
	uint8_t xstep; // 4 bits each to x and y (we only use 4 and 8 anyway)
	uint8_t ystep; // 4 bits each to x and y (we only use 4 and 8 anyway)
	uint8_t zstep; // 4 bits each to x and y (we only use 4 and 8 anyway)
	uint64_t id_size; // label per connected component 
	uint32_t value_size; // boundary encodings (less than size / 16 or size / 64)
	uint64_t location_size; // remapped labels

	CompressoHeader() :
		data_width(8), 
		sx(1), sy(1), sz(1), 
		xstep(8), ystep(8), zstep(1),
		id_size(0), value_size(0), location_size(0)
	{}

	CompressoHeader(
		const uint8_t _data_width,
		const uint16_t _sx, const uint16_t _sy, const uint16_t _sz,
		const uint8_t _xstep = 8, const uint8_t _ystep = 8, const uint8_t _zstep = 1,
		const uint64_t _id_size = 0, const uint32_t _value_size = 0, 
		const uint64_t _location_size = 0
	) : 
		data_width(_data_width), 
		sx(_sx), sy(_sy), sz(_sz), 
		xstep(_xstep), ystep(_ystep), zstep(_zstep),
		id_size(_id_size), value_size(_value_size), location_size(_location_size)
	{}

	size_t tochars(std::vector<unsigned char> &buf, size_t idx = 0) {
		if ((idx + CompressoHeader::header_size) >= buf.size()) {
			throw std::runtime_error("Unable to write past end of buffer.");
		}

		size_t i = idx;
		for (int j = 0; j < 4; j++, i++) {
			buf[i] = magic[j];
		}

		i += itoc(format_version, buf, i);
		i += itoc(data_width, buf, i);
		i += itoc(sx, buf, i);
		i += itoc(sy, buf, i);
		i += itoc(sz, buf, i);
		i += itoc(xstep, buf, i);
		i += itoc(ystep, buf, i);
		i += itoc(zstep, buf, i);
		i += itoc(id_size, buf, i);
		i += itoc(value_size, buf, i);
		i += itoc(location_size, buf, i);

		return i - idx;
	}
};


// false = boundary, true = not boundary
template <typename T>
bool* extract_boundaries(
	T *data, 
	const size_t sx, const size_t sy, const size_t sz
) {
	const size_t sxy = sx * sy;
	const size_t voxels = sxy * sz;
	bool *boundaries = new bool[voxels]();
	if (!boundaries) { 
		fprintf(stderr, "Failed to allocate memory for boundaries.\n"); 
		exit(-1); 
	}

	for (size_t z = 0; z < sz; z++) {
		for (size_t y = 0; y < sy; y++) {
			for (size_t x = 0; x < sx; x++) {
				size_t loc = x + sx * y + sxy * z;
				boundaries[loc] = true;

				// check the east neighbor
				if (x < sx - 1 && data[loc] != data[loc + 1]) { 
					boundaries[loc] = false;
				}
				// check the south neighbor
				else if (y < sy - 1 && data[loc] != data[loc + sx]) {
					boundaries[loc] = false;
				}
			}
		}
	}

	return boundaries;
}

template <typename T>
std::vector<T> component_map(
    uint32_t *components, T *labels, 
    const size_t sx, const size_t sy, const size_t sz,
    const size_t num_components = 100
) {
    const size_t sxy = sx * sy;
    const size_t voxels = sxy * sz;

    std::vector<T> ids;
    ids.reserve(num_components);

    if (voxels == 0) {
    	return ids;
    }

    size_t loc = 0;
    for (size_t z = 0; z < sz; z++) {
      std::set<T> hash_map;
      loc = z * sxy;
      T last_label = components[loc];
      hash_map.insert(components[loc]);
      ids.push_back(labels[loc] + 1);

      for (size_t y = 0; y < sy; y++) {
        for (size_t x = 0; x < sx; x++) {
          loc = x + sx * y + sxy * z;

          if (last_label == components[loc]) {
          	continue;
          }

          bool inserted = hash_map.insert(components[loc]).second;
          if (inserted) {
            ids.push_back(labels[loc] + 1);
          }

          last_label = components[loc];
        }
      }
    }

    return ids;
}

template <typename T>
std::vector<T> encode_boundaries(
    bool *boundaries, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t xstep, const size_t ystep, const size_t zstep
) {

    const size_t sxy = sx * sy;

    const size_t nz = (sz + (zstep / 2)) / zstep;
    const size_t ny = (sy + (ystep / 2)) / ystep;
    const size_t nx = (sx + (xstep / 2)) / xstep;
    const size_t nblocks = nz * ny * nx;

    std::vector<T> boundary_data(nblocks);
    
    size_t xblock, yblock, zblock;
    size_t xoffset, yoffset, zoffset;

    // all these divisions can be replaced by plus/minus
    for (size_t z = 0; z < sz; z++) {
      zblock = z / zstep;
      zoffset = z % zstep;
      for (size_t y = 0; y < sy; y++) {
        yblock = y / ystep;
        yoffset = y % ystep;
        for (size_t x = 0; x < sx; x++) {
          size_t loc = x + sx * y + sxy * z;

          // boundaries == false is the actual boundaries
          // b/c we used cc3d which treats black as bg instead 
          // of white
          if (boundaries[loc]) { 
            continue; 
          }

          xblock = x / xstep;
          xoffset = x % xstep;

          size_t block = xblock + nx * yblock + (ny * nx) * zblock;
          size_t offset = xoffset + xstep * yoffset + (ystep * xstep) * zoffset;

          boundary_data[block] += (1LU << offset);
        }
      }
    }

    return boundary_data;    
}

template <typename T>
std::vector<T> encode_indeterminate_locations(
    bool *boundaries, T *labels, 
    const size_t sx, const size_t sy, const size_t sz
) {
  const size_t sxy = sx * sy;
  std::vector<T> locations(sx * sy * sz);

  int64_t iv = 0;
  for (size_t z = 0; z < sz; z++) {
    for (size_t y = 0; y < sy; y++) {
      for (size_t x = 0; x < sx; x++, iv++) {
        size_t loc = x + sx * y + sxy * z;

        if (!boundaries[loc]) { 
          continue; 
        }
        else if (y > 0 && !boundaries[loc - sx]) {
          continue; // boundaries[iv] = 0;
        }
        else if (x > 0 && !boundaries[loc - 1]) {
          continue; // boundaries[iv] = 0;
        }
        
        size_t north = loc - 1; // IndicesToIndex(ix - 1, iy, iz);
        size_t south = loc + 1; // IndicesToIndex(ix + 1, iy, iz);
        size_t east = loc - sx;// IndicesToIndex(ix, iy - 1, iz);
        size_t west = loc + sx; // IndicesToIndex(ix, iy + 1, iz);
        size_t up = loc + sxy; // IndicesToIndex(ix, iy, iz + 1);
        size_t down = loc - sxy; // IndicesToIndex(ix, iy, iz - 1);

        // see if any of the immediate neighbors are candidates
        if (x > 0 && !boundaries[north] && (labels[north] == labels[iv])) {
          locations.push_back(0);
        }
        else if (x < sx - 1 && !boundaries[south] && (labels[south] == labels[iv])) {
          locations.push_back(1);
        }
        else if (y > 0 && !boundaries[east] && (labels[east] == labels[iv])) {
          locations.push_back(2);
        }
        else if (y < sy - 1 && !boundaries[west] && (labels[west] == labels[iv])) {
          locations.push_back(3);
        }
        else if (z > 0 && !boundaries[down] && (labels[down] == labels[iv])) {
          locations.push_back(4);
        }
        else if (z < sz - 1 && !boundaries[up] && (labels[up] == labels[iv])) {
          locations.push_back(5);
        }
        else {
        	if (labels[loc] > std::numeric_limits<T>::max() - 6) {
          	locations.push_back(labels[loc] + 6);
          }
          else {
          	printf("%lld %lld", labels[loc], std::numeric_limits<T>::max() - 6);
          	throw std::runtime_error("compresso: Cannot encode labels within 6 units of integer overflow.");
          }
        }
      }
    }
  }

  return locations;
}

template <typename T>
std::vector<T> unique(const std::vector<T> &data) {
	std::vector<T> values;
  std::set<T> hash_map;
  const size_t n_vals = data.size();
  for (size_t iv = 0; iv < n_vals; iv++) {
  	bool inserted = hash_map.insert(data[iv]).second;
    if (inserted) {
      values.push_back(data[iv]);
    }
  }
  sort(values.begin(), values.end());
  return values;
}

template <typename T>
void renumber_boundary_data(const std::vector<T>& values, std::vector<T> &boundary_data) {
  if (boundary_data.size() == 0) {
  	return;
  }

  std::unordered_map<T, T> mapping;
  const size_t n_vals = values.size();
  for (size_t iv = 0; iv < n_vals; iv++) {
    mapping[values[iv]] = iv;
  }

  const size_t n_data = boundary_data.size();
  T last = boundary_data[0];
  boundary_data[0] = mapping[boundary_data[0]];
  T last_remap = boundary_data[0];

  for (size_t iv = 1; iv < n_data; iv++) {
  	if (boundary_data[iv] == last) {
  		boundary_data[iv] = last_remap;
  		continue;
  	}

  	last_remap = mapping[boundary_data[iv]];
  	last = boundary_data[iv];
    boundary_data[iv] = last_remap;
  }
}

template <typename T>
std::vector<T> run_length_encode_windows(const std::vector<T> &windows) {
	std::vector<T> rle_windows;
	rle_windows.reserve(windows.size() / 4);

	bool zero_run = false;
	size_t prev_zero = 0;

	const size_t window_size = windows.size();
	for (size_t i = 0; i < window_size; i++) {
		if (windows[i] == 0) {
			if (!zero_run) {
				zero_run = true;
				prev_zero = i;
			}
		}
		else {
			if (zero_run) {
				rle_windows.push_back((i - prev_zero) * 2 + 1);
				zero_run = false;
			}
			rle_windows.push_back(windows[i] * 2);
		}
	}

	return rle_windows;
}

bool is_little_endian() {
	int n = 1;
	return (*(char *) & n == 1);
}

/* compress
 *
 * Convert 3D integer array data into a compresso encoded byte stream.
 * Array is expected to be in Fortran order.
 *
 * Parameters:
 *  data: pointer to 3D integer segmentation image 
 *  sx, sy, sz: axial dimension sizes
 *  xstep, ystep, zstep: (optional) picks the size of the 
 *      compresso grid. 4x4x1 or 8x8x1 are acceptable sizes.
 *
 * Returns: vector<char>
 */
template <typename T>
std::vector<unsigned char> compress(
	T *labels, 
	const size_t sx, const size_t sy, const size_t sz,
	const size_t xstep = 8, const size_t ystep = 8, const size_t zstep = 1
) {

	if (xstep * ystep * zstep > 64) {
		throw std::runtime_error("Unable to encode blocks larger than 64 voxels.");
	}

	const size_t sxy = sx * sy;
	const size_t voxels = sx * sy * sz;

	const size_t nx = (sz + (zstep / 2)) / zstep;
	const size_t ny = (sy + (ystep / 2)) / ystep;
	const size_t nz = (sx + (xstep / 2)) / xstep;

	const size_t nblocks = nx * ny * nz;

	bool *boundaries =  extract_boundaries<T>(labels, sx, sy, sz);
	size_t num_components = 0;
	uint32_t *components = cc3d::connected_components2d<uint32_t>(boundaries, sx, sy, sz, num_components);
	
	// Very similar to fastremap.component_map
	std::vector<T> ids = component_map<T>(components, labels, sx, sy, sz, num_components);
	delete[] components;

	// for 4,4,1 we could use uint16_t
	std::vector<uint64_t> windows = encode_boundaries<uint64_t>(boundaries, sx, sy, sz, xstep, ystep, zstep);
	std::vector<T> locations = encode_indeterminate_locations<T>(boundaries, labels, sx, sy, sz);
	delete[] boundaries;

	std::vector<uint64_t> window_values = unique<uint64_t>(windows);
	renumber_boundary_data(window_values, windows);
	windows = run_length_encode_windows<uint64_t>(windows);

	size_t num_out_bytes = (
		CompressoHeader::header_size 
		+ (ids.size() * sizeof(T))
		+ (window_values.size() * sizeof(uint64_t))
		+ (locations.size() * sizeof(T))
		+ (nblocks * sizeof(uint64_t))
	);
	std::vector<unsigned char> compressed_data(num_out_bytes);

	CompressoHeader header(
		/*data_width=*/sizeof(T), 
		/*sx=*/sx, /*sy=*/sy, /*sz=*/sz,
		/*xstep=*/xstep, /*ystep=*/ystep, /*zstep=*/zstep,
		/*id_size=*/ids.size(), 
		/*value_size=*/window_values.size(), 
		/*location_size=*/locations.size()
	);
	size_t idx = header.tochars(compressed_data, 0);
	for (size_t i = 0 ; i < ids.size(); i++) {
    idx += itoc(ids[i], compressed_data, idx);
	}
	for (size_t i = 0 ; i < window_values.size(); i++) {
    idx += itoc(window_values[i], compressed_data, idx);
	}
	for (size_t i = 0 ; i < locations.size(); i++) {
    idx += itoc(locations[i], compressed_data, idx);
	}
	for (size_t i = 0 ; i < nblocks; i++) {
    idx += itoc(windows[i], compressed_data, idx);
	}

	return compressed_data;
}

};

namespace pycompresso {

template <typename T>
std::vector<unsigned char> cpp_compress(
	T *labels, 
	const size_t sx, const size_t sy, const size_t sz,
	const size_t xstep = 8, const size_t ystep = 8, const size_t zstep = 1
) {

	return compresso::compress<T>(labels, sx, sy, sz, xstep, ystep, zstep);
}

};

#endif