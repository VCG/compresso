#ifndef __COMPRESSO_H__
#define __COMPRESSO_H__

#include <unordered_map>
#include <set>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <cstdint>

namespace compresso {

// size of various dimensions

static int row_size = -1;
static int sheet_size = -1;
static int grid_size = -1;

///////////////////////////////////
//// INTERNAL HELPER FUNCTIONS ////
///////////////////////////////////

static int 
IndicesToIndex(int ix, int iy, int iz) {
    return iz * sheet_size + iy * row_size + ix;
}


/////////////////////////////////////////
//// UNION-FIND CLASS FOR COMPONENTS ////
/////////////////////////////////////////


template <typename T>
class DisjointSet {
public:
  T *ids;
  size_t length;

  DisjointSet () {
    length = 65536;
    ids = new T[length]();
  }

  DisjointSet (size_t len) {
    length = len;
    ids = new T[length]();
  }

  DisjointSet (const DisjointSet &cpy) {
    length = cpy.length;
    ids = new T[length]();

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

  void clear () {
    for (size_t i = 0; i < length; i++) {
        ids[i] = 0;
    }
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
/////////////////////////////////////////
//// COMPRESSO COMPRESSION ALGORITHM ////
/////////////////////////////////////////

static bool *
ExtractBoundaries(unsigned long *data, int zres, int yres, int xres)
{
    // create the boundaries array
    bool *boundaries = new bool[grid_size];
    if (!boundaries) { fprintf(stderr, "Failed to allocate memory for boundaries...\n"); exit(-1); }

    // determine which pixels differ from east or south neighbors
    for (int iz = 0; iz < zres; ++iz) {
        for (int iy = 0; iy < yres; ++iy) {
            for (int ix = 0; ix < xres; ++ix) {
                int iv = IndicesToIndex(ix, iy, iz);

                boundaries[iv] = false;

                // check the east neighbor
                if (ix < xres - 1) { 
                    if (data[iv] != data[IndicesToIndex(ix + 1, iy, iz)]) boundaries[iv] = true;
                }
                // check the south neighbor
                if (iy < yres - 1) {
                    if (data[iv] != data[IndicesToIndex(ix, iy + 1, iz)]) boundaries[iv] = true;
                }
            }
        }
    }

    // return the boundary array
    return boundaries;
}

static unsigned long *
ConnectedComponents(bool *boundaries, int zres, int yres, int xres)
{
    // create the connected components
    unsigned long *components = new unsigned long[grid_size]();
    if (!components) { 
        fprintf(stderr, "Failed to allocate memory for connected components...\n"); exit(-1); 
    }

    DisjointSet<unsigned int> equivalences(xres * yres);

    // run connected components for each slice
    for (int iz = 0; iz < zres; ++iz) {
        equivalences.clear();

        // current label in connected component
        int curlab = 1;
        for (int iy = 0; iy < yres; ++iy) {
            for (int ix = 0; ix < xres; ++ix) {
                int iv = IndicesToIndex(ix, iy, iz);

                // continue if boundary
                if (boundaries[iv]) {
                    continue;
                }

                // only consider the pixel directly to the north and west
                int north = IndicesToIndex(ix - 1, iy, iz);
                int west = IndicesToIndex(ix, iy - 1, iz);

                int neighbor_labels[2] = { 0, 0 };

                // get the labels for the relevant neighbor
                if (ix > 0) {
                    neighbor_labels[0] = components[north];
                }
                
                if (iy > 0) {
                    neighbor_labels[1] = components[west];
                }

                // if the neighbors are boundary, create new label
                if (!neighbor_labels[0] && !neighbor_labels[1]) {
                    components[iv] = curlab;
                    equivalences.add(curlab);
                    curlab++;
                }
                // the two pixels have equal non-trivial values
                else if (neighbor_labels[0] == neighbor_labels[1]) {
                    components[iv] = neighbor_labels[0];
                }
                // neighbors have differing values
                else if (!neighbor_labels[0]) {
                    components[iv] = neighbor_labels[1];
                }
                else if (!neighbor_labels[1]) {
                    components[iv] = neighbor_labels[0];
                }
                else {
                    components[iv] = std::min(neighbor_labels[0], neighbor_labels[1]);
                    equivalences.unify(neighbor_labels[0], neighbor_labels[1]);
                }
            }
        }

        size_t num_labels = curlab;
        unsigned int* renumber = new unsigned int[num_labels + 1]();
        // reset the current label to 1
        curlab = 1;
        int label = 0;

        // create connected components (ordered)
        for (int iy = 0; iy < yres; ++iy) {
            for (int ix = 0; ix < xres; ++ix) {
                int iv = IndicesToIndex(ix, iy, iz);

                if (boundaries[iv]) {
                   continue;
                }

                label = equivalences.root(components[iv]);

                if (renumber[label]) {
                  components[iv] = renumber[label];
                }
                else {
                  renumber[label] = curlab;
                  components[iv] = curlab;
                  curlab++;
                }
            }
        }

        delete[] renumber;
    }

    return components;
}

static std::vector<unsigned long> *
IDMapping(unsigned long *components, unsigned long *data, int zres, int yres, int xres)
{
    // create a vector of the ids
    std::vector<unsigned long> *ids = new std::vector<unsigned long>();

    for (int iz = 0; iz < zres; ++iz) {
        // create a set for this individual slice
        std::set<unsigned long> hash_map = std::set<unsigned long>();

        // iterate over the entire slice
        for (int iy = 0; iy < yres; ++iy) {
            for (int ix = 0; ix < xres; ++ix) {
                int iv = IndicesToIndex(ix, iy, iz);

                // get the segment id
                unsigned long component_id = components[iv];

                // if this component does not belong yet, add it
                if (!hash_map.count(component_id)) {
                    hash_map.insert(component_id);

                    // add the segment id
                    unsigned long segment_id = data[iv] + 1;
                    ids->push_back(segment_id);
                }
            }
        }

    }

    // return the mapping
    return ids;
}

static unsigned long *
EncodeBoundaries(bool *boundaries, int zres, int yres, int xres, int zstep, int ystep, int xstep) 
{
    // determine the number of blocks in the z, y, and x dimensions
    int nzblocks = (int) (ceil((double)zres / zstep) + 0.5);
    int nyblocks = (int) (ceil((double)yres / ystep) + 0.5);
    int nxblocks = (int) (ceil((double)xres / xstep) + 0.5);

    // create an empty array for the encodings
    int nblocks = nzblocks * nyblocks * nxblocks;
    unsigned long *boundary_data = new unsigned long[nblocks];
    for (int iv = 0; iv < nblocks; ++iv)
        boundary_data[iv] = 0;
    
    for (int iz = 0; iz < zres; ++iz) {
        for (int iy = 0; iy < yres; ++iy) {
            for (int ix = 0; ix < xres; ++ix) {
                int iv = IndicesToIndex(ix, iy, iz);

                // no encoding for non-boundaries
                if (!boundaries[iv]) continue;

                // find the block from the index
                int zblock = iz / zstep;
                int yblock = iy / ystep;
                int xblock = ix / xstep;

                // find the offset within the block
                int zoffset = iz % zstep;
                int yoffset = iy % ystep;
                int xoffset = ix % xstep;

                int block = zblock * (nyblocks * nxblocks) + yblock * nxblocks + xblock;
                int offset = zoffset * (ystep * xstep) + yoffset * xstep + xoffset;

                boundary_data[block] += (1LU << offset);
            }
        }
    }

    return boundary_data;    
}

static std::vector<unsigned long> *
ValueMapping(unsigned long *boundary_data, int nblocks)
{
    // get a list of values
    std::vector<unsigned long> *values = new std::vector<unsigned long>();
    std::set<unsigned long> hash_map = std::set<unsigned long>();

    // go through all boundary data to create array of values
    for (int iv = 0; iv < nblocks; ++iv) {
        if (!hash_map.count(boundary_data[iv])) {
            hash_map.insert(boundary_data[iv]);
            values->push_back(boundary_data[iv]);
        }
    }

    // sort the values
    sort(values->begin(), values->end());

    // create mapping from values to indices
    std::unordered_map<unsigned long, unsigned long> mapping = std::unordered_map<unsigned long, unsigned long>();
    for (unsigned int iv = 0; iv < values->size(); ++iv) {
        mapping[(*values)[iv]] = iv;
    }

    // update boundary data
    for (int iv = 0; iv < nblocks; ++iv) {
        boundary_data[iv] = mapping[boundary_data[iv]];
    }

    // return values
    return values;
}

std::vector<unsigned long> *
EncodeIndeterminateLocations(bool *boundaries, unsigned long *data, int zres, int yres, int xres)
{
    // update global size variables
    row_size = xres;
    sheet_size = yres * xres;
    grid_size = zres * yres * xres;

    std::vector<unsigned long> *locations = new std::vector<unsigned long>();

    int iv = 0;
    for (int iz = 0; iz < zres; ++iz) {
        for (int iy = 0; iy < yres; ++iy) {
            for (int ix = 0; ix < xres; ++ix, ++iv) {

                if (!boundaries[iv]) continue;
                else if (iy > 0 && !boundaries[IndicesToIndex(ix, iy - 1, iz)]) continue; //boundaries[iv] = 0;
                else if (ix > 0 && !boundaries[IndicesToIndex(ix - 1, iy, iz)]) continue; //boundaries[iv] = 0;
                else {
                    int north = IndicesToIndex(ix - 1, iy, iz);
                    int south = IndicesToIndex(ix + 1, iy, iz);
                    int east = IndicesToIndex(ix, iy - 1, iz);
                    int west = IndicesToIndex(ix, iy + 1, iz);
                    int up = IndicesToIndex(ix, iy, iz + 1);
                    int down = IndicesToIndex(ix, iy, iz - 1);

                    // see if any of the immediate neighbors are candidates
                    if (ix > 0 && !boundaries[north] && data[north] == data[iv])
                        locations->push_back(0);
                    else if (ix < xres - 1 && !boundaries[south] && data[south] == data[iv])
                        locations->push_back(1);
                    else if (iy > 0 && !boundaries[east] && data[east] == data[iv])
                        locations->push_back(2);
                    else if (iy < yres - 1 && !boundaries[west] && data[west] == data[iv])
                        locations->push_back(3);
                    else if (iz > 0 && !boundaries[down] && data[down] == data[iv])
                        locations->push_back(4);
                    else if (iz < zres - 1 && !boundaries[up] && data[up] == data[iv])
                        locations->push_back(5);
                    else 
                        locations->push_back(data[IndicesToIndex(ix, iy, iz)] + 6);
                }
            }
        }
    }

    return locations;
}


unsigned long *
Compress(unsigned long *data, int zres, int yres, int xres, int zstep, int ystep, int xstep)
{
    // set global variables
    row_size = xres;
    sheet_size = yres * xres;
    grid_size = zres * yres * xres;

    // determine the number of blocks in the z, y, and x dimensions
    int nzblocks = (int) (ceil((double)zres / zstep) + 0.5);
    int nyblocks = (int) (ceil((double)yres / ystep) + 0.5);
    int nxblocks = (int) (ceil((double)xres / xstep) + 0.5);

    // create an empty array for the encodings
    int nblocks = nzblocks * nyblocks * nxblocks;

    // get boundary voxels
    bool *boundaries = ExtractBoundaries(data, zres, yres, xres);   

    // get the connected components
    unsigned long *components = ConnectedComponents(boundaries, zres, yres, xres);

    std::vector<unsigned long> *ids = IDMapping(components, data, zres, yres, xres);

    unsigned long *boundary_data = EncodeBoundaries(boundaries, zres, yres, xres, zstep, ystep, xstep);

    std::vector<unsigned long> *values = ValueMapping(boundary_data, nblocks);

    std::vector<unsigned long> *locations = EncodeIndeterminateLocations(boundaries, data, zres, yres, xres);

    unsigned short header_size = 9;
    unsigned long *compressed_data = new unsigned long[header_size + ids->size() + values->size() + locations->size() + nblocks];

    // add the resolution
    compressed_data[0] = zres;
    compressed_data[1] = yres;
    compressed_data[2] = xres;

    // add the sizes of the vectors
    compressed_data[3] = ids->size();
    compressed_data[4] = values->size();
    compressed_data[5] = locations->size();

    compressed_data[6] = zstep;
    compressed_data[7] = ystep;
    compressed_data[8] = xstep;

    int iv = header_size;
    for (unsigned int ix = 0 ; ix < ids->size(); ++ix, ++iv)
        compressed_data[iv] = (*ids)[ix];
    for (unsigned int ix = 0; ix < values->size(); ++ix, ++iv)
        compressed_data[iv] = (*values)[ix];
    for (unsigned int ix = 0; ix < locations->size(); ++ix, ++iv)
        compressed_data[iv] = (*locations)[ix];
    for (int ix = 0; ix < nblocks; ++ix, ++iv)
        compressed_data[iv] = boundary_data[ix];

    // free memory
    delete[] boundaries;
    delete[] components;
    delete ids;
    delete[] boundary_data;
    delete values;
    delete locations;

    return compressed_data;
}



///////////////////////////////////////////
//// COMPRESSO DECOMPRESSION ALGORITHM ////
///////////////////////////////////////////

static bool *
DecodeBoundaries(unsigned long *boundary_data, std::vector<unsigned long> *values, int zres, int yres, int xres, int zstep, int ystep, int xstep)
{
    int nyblocks = (int)(ceil((double)yres / ystep) + 0.5);
    int nxblocks = (int)(ceil((double)xres / xstep) + 0.5);

    bool *boundaries = new bool[grid_size];
    for (int iv = 0; iv < grid_size; ++iv)
        boundaries[iv] = false;

    for (int iz = 0; iz < zres; ++iz) {
        for (int iy = 0; iy < yres; ++iy) {
            for (int ix = 0; ix < xres; ++ix) {
                int iv = IndicesToIndex(ix, iy, iz);

                int zblock = iz / zstep;
                int yblock = iy / ystep;
                int xblock = ix / xstep;

                int zoffset = iz % zstep;
                int yoffset = iy % ystep;
                int xoffset = ix % xstep;

                int block = zblock * (nyblocks * nxblocks) + yblock * nxblocks + xblock;
                int offset = zoffset * (ystep * xstep) + yoffset * xstep + xoffset;

                unsigned long value = (*values)[boundary_data[block]];
                if ((value >> offset) % 2) boundaries[iv] = true;
            }
        }
    }

    return boundaries;
}

static unsigned long *
IDReverseMapping(unsigned long *components, std::vector<unsigned long> *ids, int zres, int yres, int xres)
{
    unsigned long *decompressed_data = new unsigned long[grid_size];
    for (int iv = 0; iv < grid_size; ++iv)
        decompressed_data[iv] = 0;

    int ids_index = 0;
    for (int iz = 0; iz < zres; ++iz) {

        // create mapping (not memory efficient but FAST!!)
        // number of components is guaranteed to be less than ids->size()
        unsigned long *mapping = new unsigned long[ids->size()];
        for (unsigned int iv = 0; iv < ids->size(); ++iv) {
            mapping[iv] = 0;
        }

        for (int iy = 0; iy < yres; ++iy) {
            for (int ix = 0; ix < xres; ++ix) {
                int iv = IndicesToIndex(ix, iy, iz);

                if (!mapping[components[iv]]) {
                    mapping[components[iv]] = (*ids)[ids_index];
                    ids_index++;
                }

                decompressed_data[iv] = mapping[components[iv]] - 1;
            }
        }
    }

    return decompressed_data;
}

static void 
DecodeIndeterminateLocations(bool *boundaries, unsigned long *decompressed_data, std::vector<unsigned long> *locations, int zres, int yres, int xres)
{
    int iv = 0;
    int index = 0;

    // go through all coordinates
    for (int iz = 0; iz < zres; ++iz) {
        for (int iy = 0; iy < yres; ++iy) {
            for (int ix = 0; ix < xres; ++ix, ++iv) {
                int north = IndicesToIndex(ix - 1, iy, iz);
                int west = IndicesToIndex(ix, iy - 1, iz);

                if (!boundaries[iv]) continue;
                else if (ix > 0 && !boundaries[north]) {
                    decompressed_data[iv] = decompressed_data[north];
                    //boundaries[iv] = 0;
                }
                else if (iy > 0 && !boundaries[west]) {
                    decompressed_data[iv] = decompressed_data[west];
                    //boundaries[iv] = 0;
                }
                else {
                    int offset = (*locations)[index];
                    if (offset == 0) decompressed_data[iv] = decompressed_data[IndicesToIndex(ix - 1, iy, iz)];
                    else if (offset == 1) decompressed_data[iv] = decompressed_data[IndicesToIndex(ix + 1, iy, iz)];
                    else if (offset == 2) decompressed_data[iv] = decompressed_data[IndicesToIndex(ix, iy - 1, iz)];
                    else if (offset == 3) decompressed_data[iv] = decompressed_data[IndicesToIndex(ix, iy + 1, iz)];
                    else if (offset == 4) decompressed_data[iv] = decompressed_data[IndicesToIndex(ix, iy, iz - 1)];
                    else if (offset == 5) decompressed_data[iv] = decompressed_data[IndicesToIndex(ix, iy, iz + 1)];
                    else {
                        decompressed_data[iv] = offset - 6;                        
                    }
                    index += 1;
                }
            }
        }
    }
}

unsigned long* 
Decompress(unsigned long *compressed_data)
{
    // constants
    int header_size = 9;

    // get the resolution
    int zres = compressed_data[0];
    int yres = compressed_data[1];
    int xres = compressed_data[2];

    // set global variables
    row_size = xres;
    sheet_size = yres * xres;
    grid_size = zres * yres * xres;

    // get the size of the vectors
    int ids_size = compressed_data[3];
    int values_size = compressed_data[4];
    int locations_size = compressed_data[5];

    // get the step size
    int zstep = compressed_data[6];
    int ystep = compressed_data[7];
    int xstep = compressed_data[8];

    // determine the number of blocks in the z, y, and x dimensions
    int nzblocks = (int) (ceil((double)zres / zstep) + 0.5);
    int nyblocks = (int) (ceil((double)yres / ystep) + 0.5);
    int nxblocks = (int) (ceil((double)xres / xstep) + 0.5);

    // create an empty array for the encodings
    int nblocks = nzblocks * nyblocks * nxblocks;

    // allocate memory for all arrays
    std::vector<unsigned long> *ids = new std::vector<unsigned long>();
    std::vector<unsigned long> *values = new std::vector<unsigned long>();
    std::vector<unsigned long> *locations = new std::vector<unsigned long>();
    unsigned long *boundary_data = new unsigned long[nblocks];

    int iv = header_size;
    for (int ix = 0; ix < ids_size; ++ix, ++iv)
        ids->push_back(compressed_data[iv]);
    for (int ix = 0; ix < values_size; ++ix, ++iv)
        values->push_back(compressed_data[iv]);
    for (int ix = 0; ix < locations_size; ++ix, ++iv)
        locations->push_back(compressed_data[iv]);
    for (int ix = 0; ix < nblocks; ++ix, ++iv)
        boundary_data[ix] = compressed_data[iv];

    bool *boundaries = DecodeBoundaries(boundary_data, values, zres, yres, xres, zstep, ystep, xstep);

    unsigned long *components = ConnectedComponents(boundaries, zres, yres, xres);

    unsigned long *decompressed_data = IDReverseMapping(components, ids, zres, yres, xres);

    DecodeIndeterminateLocations(boundaries, decompressed_data, locations, zres, yres, xres);

    // free memory
    delete[] boundaries;
    delete[] components;
    delete[] boundary_data;
    delete ids;
    delete values;
    delete locations;

    return decompressed_data;
}

};

#endif