#ifndef __COMPRESSO_H__
#define __COMPRESSO_H__

#include <unordered_map>
#include <set>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include "cc3d.hpp"


namespace compresso {

/////////////////////////////////////////
//// COMPRESSO COMPRESSION ALGORITHM ////
/////////////////////////////////////////

bool* ExtractBoundaries(
    uint64_t *data, 
    const size_t sx, const size_t sy, const size_t sz
) {
    const size_t sxy = sx * sy;
    const size_t voxels = sxy * sz;
    bool *boundaries = new bool[voxels]();
    if (!boundaries) { 
        fprintf(stderr, "Failed to allocate memory for boundaries...\n"); exit(-1); 
    }

    for (size_t z = 0; z < sz; z++) {
        for (size_t y = 0; y < sy; y++) {
            for (size_t x = 0; x < sx; x++) {
                size_t loc = x + sx * y + sxy * z;
                boundaries[loc] = false;

                // check the east neighbor
                if (x < sx - 1 && data[loc] != data[loc + 1]) { 
                    boundaries[loc] = true;
                }
                // check the south neighbor
                else if (y < sy - 1 && data[loc] != data[loc + sx]) {
                    boundaries[loc] = true;
                }
            }
        }
    }

    return boundaries;
}

std::vector<uint64_t>* IDMapping(
    uint64_t *components, uint64_t *data, 
    const size_t sx, const size_t sy, const size_t sz
) {
    const size_t sxy = sx * sy;

    std::vector<uint64_t> *ids = new std::vector<uint64_t>();

    size_t loc = 0;
    for (size_t z = 0; z < sz; ++z) {
        std::set<uint64_t> hash_map = std::set<uint64_t>();

        for (size_t y = 0; y < sy; ++y) {
            for (size_t x = 0; x < sx; ++x) {
                loc = x + sx * y + sxy * z;

                uint64_t component_id = components[loc];

                // if this component does not belong yet, add it
                if (!hash_map.count(component_id)) {
                    hash_map.insert(component_id);
                    uint64_t segment_id = data[loc] + 1;
                    ids->push_back(segment_id);
                }
            }
        }
    }

    return ids;
}

uint64_t* EncodeBoundaries(
    bool *boundaries, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t xstep, const size_t ystep, const size_t zstep
) {

    const size_t sxy = sx * sy;

    const size_t nzblocks = (sz + (zstep / 2)) / zstep;
    const size_t nyblocks = (sy + (ystep / 2)) / ystep;
    const size_t nxblocks = (sx + (xstep / 2)) / xstep;
    const size_t nblocks = nzblocks * nyblocks * nxblocks;

    uint64_t *boundary_data = new uint64_t[nblocks]();
    
    size_t xblock, yblock, zblock;
    size_t xoffset, yoffset, zoffset;

    for (size_t z = 0; z < sz; z++) {
        zblock = z / zstep;
        zoffset = z % zstep;
        for (size_t y = 0; y < sy; y++) {
            yblock = y / ystep;
            yoffset = y % ystep;
            for (size_t x = 0; x < sx; x++) {
                size_t loc = x + sx * y + sxy * z;

                if (!boundaries[loc]) { 
                    continue; 
                }

                xblock = x / xstep;
                xoffset = x % xstep;

                size_t block = xblock + nxblocks * yblock + (nyblocks * nxblocks) * zblock;
                size_t offset = xoffset + xstep * yoffset + (ystep * xstep) * zoffset;

                boundary_data[block] += (1LU << offset);
            }
        }
    }

    return boundary_data;    
}

std::vector<uint64_t>* ValueMapping(uint64_t *boundary_data, int nblocks) {
    std::vector<uint64_t> *values = new std::vector<uint64_t>();
    std::set<uint64_t> hash_map = std::set<uint64_t>();

    // go through all boundary data to create array of values
    for (int iv = 0; iv < nblocks; ++iv) {
        if (!hash_map.count(boundary_data[iv])) {
            hash_map.insert(boundary_data[iv]);
            values->push_back(boundary_data[iv]);
        }
    }

    sort(values->begin(), values->end());

    // create mapping from values to indices
    std::unordered_map<uint64_t, uint64_t> mapping = std::unordered_map<uint64_t, uint64_t>();
    for (unsigned int iv = 0; iv < values->size(); ++iv) {
        mapping[(*values)[iv]] = iv;
    }

    // update boundary data
    for (int iv = 0; iv < nblocks; ++iv) {
        boundary_data[iv] = mapping[boundary_data[iv]];
    }

    return values;
}

std::vector<uint64_t>* EncodeIndeterminateLocations(
    bool *boundaries, uint64_t *data, 
    const size_t sx, const size_t sy, const size_t sz
) {
    const size_t sxy = sx * sy;
    std::vector<uint64_t> *locations = new std::vector<uint64_t>();

    int iv = 0;
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
                if (x > 0 && !boundaries[north] && data[north] == data[iv]) {
                    locations->push_back(0);
                }
                else if (x < sx - 1 && !boundaries[south] && data[south] == data[iv]) {
                    locations->push_back(1);
                }
                else if (y > 0 && !boundaries[east] && data[east] == data[iv]){
                    locations->push_back(2);
                }
                else if (y < sy - 1 && !boundaries[west] && data[west] == data[iv]) {
                    locations->push_back(3);
                }
                else if (z > 0 && !boundaries[down] && data[down] == data[iv]) {
                    locations->push_back(4);
                }
                else if (z < sz - 1 && !boundaries[up] && data[up] == data[iv]) {
                    locations->push_back(5);
                }
                else {
                    locations->push_back(data[loc] + 6);
                }
            }
        }
    }

    return locations;
}

uint64_t* Compress(
    uint64_t *data, 
    const size_t sx, const size_t sy, const size_t sz,
    const size_t xstep, const size_t ystep, const size_t zstep
) {
    const size_t nzblocks = (sz + (zstep / 2)) / zstep;
    const size_t nyblocks = (sy + (ystep / 2)) / ystep;
    const size_t nxblocks = (sx + (xstep / 2)) / xstep;

    const size_t nblocks = nzblocks * nyblocks * nxblocks;

    // get boundary voxels
    bool *boundaries = ExtractBoundaries(data, sx, sy, sz);   

    uint64_t *components = cc3d::connected_components2d(boundaries, sx, sy, sz);
    std::vector<uint64_t> *ids = IDMapping(components, data, sx, sy, sz);
    delete[] components;
    
    uint64_t *boundary_data = EncodeBoundaries(boundaries, sx, sy, sz, xstep, ystep, zstep);

    std::vector<uint64_t> *values = ValueMapping(boundary_data, nblocks);

    std::vector<uint64_t> *locations = EncodeIndeterminateLocations(boundaries, data, sx, sy, sz);

    unsigned short header_size = 9;
    uint64_t *compressed_data = new uint64_t[header_size + ids->size() + values->size() + locations->size() + nblocks]();

    // add the resolution
    compressed_data[0] = sz;
    compressed_data[1] = sy;
    compressed_data[2] = sx;

    // add the sizes of the vectors
    compressed_data[3] = ids->size();
    compressed_data[4] = values->size();
    compressed_data[5] = locations->size();

    compressed_data[6] = zstep;
    compressed_data[7] = ystep;
    compressed_data[8] = xstep;

    size_t iv = header_size;
    for (size_t ix = 0 ; ix < ids->size(); ++ix, ++iv) {
        compressed_data[iv] = (*ids)[ix];
    }
    for (size_t ix = 0; ix < values->size(); ++ix, ++iv) {
        compressed_data[iv] = (*values)[ix];
    }
    for (size_t ix = 0; ix < locations->size(); ++ix, ++iv) {
        compressed_data[iv] = (*locations)[ix];
    }
    for (size_t ix = 0; ix < nblocks; ++ix, ++iv) {
        compressed_data[iv] = boundary_data[ix];
    }

    delete[] boundaries;
    delete ids;
    delete[] boundary_data;
    delete values;
    delete locations;

    return compressed_data;
}

///////////////////////////////////////////
//// COMPRESSO DECOMPRESSION ALGORITHM ////
///////////////////////////////////////////

bool* DecodeBoundaries(
    uint64_t *boundary_data, std::vector<uint64_t> *values, 
    const size_t sx, const size_t sy, const size_t sz,
    const size_t xstep, const size_t ystep, const size_t zstep
) {

    const size_t sxy = sx * sy;
    const size_t voxels = sx * sy * sz;

    const size_t nyblocks = (sy + (ystep / 2)) / ystep;
    const size_t nxblocks = (sx + (xstep / 2)) / xstep;

    bool *boundaries = new bool[voxels]();

    size_t xblock, yblock, zblock;
    size_t xoffset, yoffset, zoffset;

    for (size_t z = 0; z < sz; ++z) {
        zblock = z / zstep;
        zoffset = z % zstep;
        for (size_t y = 0; y < sy; ++y) {
            yblock = y / ystep;
            yoffset = y % ystep;
            for (size_t x = 0; x < sx; ++x) {
                size_t iv = x + sx * y + sxy * z;
                xblock = x / xstep;
                xoffset = x % xstep;

                size_t block = zblock * (nyblocks * nxblocks) + yblock * nxblocks + xblock;
                size_t offset = zoffset * (ystep * xstep) + yoffset * xstep + xoffset;

                uint64_t value = (*values)[boundary_data[block]];
                if ((value >> offset) & 0b1) { 
                    boundaries[iv] = true;
                }
            }
        }
    }

    return boundaries;
}

uint64_t* IDReverseMapping(
    uint64_t *components, std::vector<uint64_t> *ids, 
    const size_t sx, const size_t sy, const size_t sz
) {
    const size_t sxy = sx * sy;
    const size_t voxels = sxy * sz;

    uint64_t *decompressed_data = new uint64_t[voxels]();
    
    size_t ids_index = 0;
    for (size_t z = 0; z < sz; z++) {
        uint64_t *mapping = new uint64_t[ids->size()]();
        for (size_t y = 0; y < sy; ++y) {
            for (size_t x = 0; x < sz; ++x) {
                size_t iv = x + sx * y + sxy * z;

                if (!mapping[components[iv]]) {
                    mapping[components[iv]] = (*ids)[ids_index];
                    ids_index++;
                }

                decompressed_data[iv] = mapping[components[iv]] - 1;
            }
        }

        delete[] mapping;
    }

    return decompressed_data;
}

void DecodeIndeterminateLocations(
    bool *boundaries, uint64_t *decompressed_data, 
    std::vector<uint64_t> *locations, 
    const size_t sx, const size_t sy, const size_t sz
) {
    const size_t sxy = sx * sy;

    size_t iv = 0;
    size_t index = 0;

    // go through all coordinates
    for (size_t z = 0; z < sz; z++) {
        for (size_t y = 0; y < sy; y++) {
            for (size_t x = 0; x < sx; x++, iv++) {
                size_t loc = x + sx * y + sxy * z;
                size_t north = loc - 1;
                size_t west = loc - sx;

                if (!boundaries[iv]) {
                    continue;
                }
                else if (x > 0 && !boundaries[north]) {
                    decompressed_data[iv] = decompressed_data[north];
                    // boundaries[iv] = 0;
                }
                else if (y > 0 && !boundaries[west]) {
                    decompressed_data[iv] = decompressed_data[west];
                    // boundaries[iv] = 0;
                }
                else {
                    size_t offset = (*locations)[index];
                    if (offset == 0) {
                        decompressed_data[iv] = decompressed_data[loc - 1];
                    }
                    else if (offset == 1) {
                        decompressed_data[iv] = decompressed_data[loc + 1];
                    }
                    else if (offset == 2) {
                        decompressed_data[iv] = decompressed_data[loc - sx];
                    }
                    else if (offset == 3) {
                        decompressed_data[iv] = decompressed_data[loc + sx];
                    }
                    else if (offset == 4) {
                        decompressed_data[iv] = decompressed_data[loc - sxy];
                    }
                    else if (offset == 5) {
                        decompressed_data[iv] = decompressed_data[loc + sxy];
                    }
                    else {
                        decompressed_data[iv] = offset - 6;                        
                    }
                    index += 1;
                }
            }
        }
    }
}

uint64_t* Decompress(uint64_t *compressed_data) {
    // constants
    const int header_size = 9;

    // get the resolution
    const size_t sz = compressed_data[0];
    const size_t sy = compressed_data[1];
    const size_t sx = compressed_data[2];

    // get the size of the vectors
    const size_t ids_size = compressed_data[3];
    const size_t values_size = compressed_data[4];
    const size_t locations_size = compressed_data[5];

    const size_t zstep = compressed_data[6];
    const size_t ystep = compressed_data[7];
    const size_t xstep = compressed_data[8];
    
    const size_t nzblocks = (sz + (zstep / 2)) / zstep;
    const size_t nyblocks = (sy + (ystep / 2)) / ystep;
    const size_t nxblocks = (sx + (xstep / 2)) / xstep;

    // create an empty array for the encodings
    size_t nblocks = nzblocks * nyblocks * nxblocks;

    // allocate memory for all arrays
    std::vector<uint64_t> *ids = new std::vector<uint64_t>();
    std::vector<uint64_t> *values = new std::vector<uint64_t>();
    std::vector<uint64_t> *locations = new std::vector<uint64_t>();
    uint64_t *boundary_data = new uint64_t[nblocks]();

    size_t iv = header_size;
    for (size_t ix = 0; ix < ids_size; ++ix, ++iv) {
        ids->push_back(compressed_data[iv]);
    }
    for (size_t ix = 0; ix < values_size; ++ix, ++iv) {
        values->push_back(compressed_data[iv]);
    }
    for (size_t ix = 0; ix < locations_size; ++ix, ++iv) {
        locations->push_back(compressed_data[iv]);
    }
    for (size_t ix = 0; ix < nblocks; ++ix, ++iv) {
        boundary_data[ix] = compressed_data[iv];
    }

    bool *boundaries = DecodeBoundaries(boundary_data, values, sx, sy, sz, xstep, ystep, zstep);

    uint64_t *components = cc3d::connected_components2d(boundaries, sx, sy, sz);
    uint64_t *decompressed_data = IDReverseMapping(components, ids, sx, sy, sz);
    delete[] components;

    DecodeIndeterminateLocations(boundaries, decompressed_data, locations, sx, sy, sz);

    // free memory
    delete[] boundaries;
    delete[] boundary_data;
    delete ids;
    delete values;
    delete locations;

    return decompressed_data;
}

};

#endif