#include "cpp-neuroglancer.h"
#include <cmath>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>



// size of various dimensions

static int row_size = -1;
static int sheet_size = -1;
static int grid_size = -1;



///////////////////////////////////
//// INTERNAL HELPER FUNCTIONS ////
///////////////////////////////////

static int 
IndicesToIndex(int ix, int iy, int iz)
{
    return iz * sheet_size + iy * row_size + ix;
}


static const int header_size = 7;


////////////////////////////////////////////
//// NEUROGLANCER COMPRESSION ALGORITHM ////
////////////////////////////////////////////

unsigned long *
neuroglancer::Compress(unsigned long *data, int zres, int yres, int xres, int bz, int by, int bx, int origz, int origy, int origx)
{
    // set global variables
    row_size = xres;
    sheet_size = yres * xres;
    grid_size = zres * yres * xres;

    // the number of blocks
    unsigned long gz = (unsigned long)(ceil((double)zres / bz) + 0.5);
    unsigned long gy = (unsigned long)(ceil((double)yres / by) + 0.5);
    unsigned long gx = (unsigned long)(ceil((double)xres / bx) + 0.5);

    // the number of elements and the block size
    unsigned long nelements = gz * gy * gx;
    unsigned int block_size = bz * by * bx;

    // get the end of the header
    unsigned int header_offset = nelements + header_size;

    // create arrays that store the table offset, number of bits, and the value offsets
    unsigned int *table_offsets = new unsigned int[nelements];
    unsigned char *nbits = new unsigned char[nelements];
    unsigned int *values_offsets = new unsigned int[nelements];
    for (unsigned int iv = 0; iv < nelements; ++iv) {
        table_offsets[iv] = 0;
        nbits[iv] = 0;
        values_offsets[iv] = 0;
    }

    // create the arrays for the encoded values and the look up table
    unsigned int **encoded_values = new unsigned int *[nelements];
    std::vector<unsigned long> *lookup_table = new std::vector<unsigned long>[nelements];
    for (unsigned int iv = 0; iv < nelements; ++iv) {
        lookup_table[iv] = std::vector<unsigned long>();
        encoded_values[iv] = new unsigned int[block_size];
        for (unsigned int ie = 0; ie < block_size; ++ie)
            encoded_values[iv][ie] = 0;
    }

    // get the number of blocks for each dimension
    int nyblocks = yres / by;
    int nxblocks = xres / bx;

    unsigned int offset = header_offset;
    // iterate over every block
    for (unsigned int index = 0; index < nelements; ++index) {
        // get the block in terms if x, y, z
        int iz = index / (nyblocks * nxblocks);
        int iy = (index - iz * nyblocks * nxblocks) / nxblocks;
        int ix = index % nxblocks;

        // get the block 
        unsigned long *block = new unsigned long[block_size];

        // populate the temporary block array
        int iv = 0;
        for (int ik = iz * bz; ik < (iz + 1) * bz; ++ik) {
            for (int ij = iy * by; ij < (iy + 1) * by; ++ij) {
                for (int ii = ix * bx; ii < (ix + 1) * bx; ++ii, ++iv) {
                    block[iv] = data[IndicesToIndex(ii, ij, ik)];
                }
            }
        }

        // get an ordered list of unique elements
        std::vector<unsigned long> unique_elements = std::vector<unsigned long>();
        std::unordered_set<unsigned long> hash_set = std::unordered_set<unsigned long>();

        for (unsigned int iv = 0; iv < block_size; ++iv) {
            if (!hash_set.count(block[iv])) {
                unique_elements.push_back(block[iv]);
                hash_set.insert(block[iv]);
            }
        }

        std::sort(unique_elements.begin(), unique_elements.end());

        // create a mapping for the look up table and populate the lookup table
        unsigned int nunique = unique_elements.size();
        std::unordered_map<unsigned long, unsigned int> mapping = std::unordered_map<unsigned long, unsigned int>();
        for (unsigned int iv = 0; iv < nunique; ++iv) {
            mapping[unique_elements[iv]] = iv;
            lookup_table[index].push_back(unique_elements[iv]);
        }

        // populate the encoded values array
        for (unsigned int iv = 0; iv < block_size; ++iv) {
            encoded_values[index][iv] = mapping[block[iv]];
        }

        // determine the number of bits
        if (nunique <= 1) nbits[index] = 0;
        else if (nunique <= 1<<1) nbits[index] = 1;
        else if (nunique <= 1<<2) nbits[index] = 2;
        else if (nunique <= 1<<4) nbits[index] = 4;
        else if (nunique <= 1<<8) nbits[index] = 8;
        else if (nunique <= 1<<16) nbits[index] = 16;
        else nbits[index] = 32;

        values_offsets[index] = offset;
        offset += nbits[index] * block_size / 64;
        table_offsets[index] = offset;
        offset += nunique;

        // free memory
        delete[] block;
    }

    unsigned long *compressed_data = new unsigned long[offset + 1];
    for (unsigned int iv = 0; iv < offset + 1; ++iv) {
        compressed_data[iv] = 0;
    }

    // add the header information
    compressed_data[0] = offset + 1;
    compressed_data[1] = zres;
    compressed_data[2] = yres;
    compressed_data[3] = xres;
    compressed_data[4] = origz;
    compressed_data[5] = origy;
    compressed_data[6] = origx;

    int data_entry = header_size;
    for (unsigned int iv = 0; iv < nelements; ++iv, ++data_entry) {
        compressed_data[data_entry] = ((unsigned long)table_offsets[iv] << 40) + ((unsigned long)nbits[iv] << 32) + values_offsets[iv];
    }

    // add the encoded values
    for (unsigned int index = 0; index < nelements; ++index) {
        // encode all of the values
        if (nbits[index] > 0) {
            // get the number of values per 8 bytes
            unsigned int nvalues_per_entry = 64 / nbits[index];
            // get the number of entries for this block
            unsigned int nentries = block_size * nbits[index] / 64;

            // for every entry, for every value
            int ii = 0;
            for (unsigned int ie = 0; ie < nentries; ++ie, ++data_entry) {
                unsigned long value = 0;
                for (unsigned int iv = 0; iv < nvalues_per_entry; ++iv, ++ii) {
                    // get the encoded value for this location
                    unsigned long encoded_value = (unsigned long)encoded_values[index][ii];

                    // the amount to shift the encoded value
                    unsigned int shift = (nvalues_per_entry - 1 - iv) * nbits[index];
                    value += (encoded_value << shift);
                }
                compressed_data[data_entry] = value;
            }
        }

        // add the lookup table
        for (unsigned int iv = 0; iv < lookup_table[index].size(); ++iv, ++data_entry) {
            compressed_data[data_entry] = lookup_table[index][iv];
        }
    }

    // free memory
    delete[] table_offsets;
    delete[] nbits;
    delete[] values_offsets;
    for (unsigned int iv = 0; iv < nelements; ++iv)
        delete[] encoded_values[iv];
    delete[] encoded_values;
    delete[] lookup_table;

    return compressed_data;
}



//////////////////////////////////////////////
//// NEUROGLANCER DECOMPRESSION ALGORITHM ////
//////////////////////////////////////////////

unsigned long *
neuroglancer::Decompress(unsigned long *compressed_data, int bz, int by, int bx)
{
    int zres = (int)compressed_data[1];
    int yres = (int)compressed_data[2];
    int xres = (int)compressed_data[3];

    // set global variables
    row_size = xres;
    sheet_size = yres * xres;
    grid_size = zres * yres * xres;

    // the number of blocks
    unsigned long gz = (unsigned long)(ceil((double)zres / bz) + 0.5);
    unsigned long gy = (unsigned long)(ceil((double)yres / by) + 0.5);
    unsigned long gx = (unsigned long)(ceil((double)xres / bx) + 0.5);

    // get the number of elements
    unsigned long nelements = gz * gy * gx;
    unsigned int block_size = bz * by * bx;

    // create arrays that store the table offset, number of bits, and the value offsets
    unsigned int *table_offsets = new unsigned int[nelements];
    unsigned char *nbits = new unsigned char[nelements];
    unsigned int *values_offsets = new unsigned int[nelements];
    for (unsigned int iv = 0; iv < nelements; ++iv) {
        table_offsets[iv] = 0;
        nbits[iv] = 0;
        values_offsets[iv] = 0;
    }

    // decompress header values
    unsigned long data_entry = header_size;
    for (unsigned int index = 0; index < nelements; ++index, ++data_entry) {
        unsigned long header = compressed_data[data_entry];

        table_offsets[index] = header >> 40;
        nbits[index] = (header << 24) >> 56;
        values_offsets[index] = (header << 32) >> 32;
    }

    // create decompressed data array
    unsigned long *decompressed_data = new unsigned long[zres * yres * xres];
    for (int iv = 0; iv < zres * yres * xres; ++iv)
        decompressed_data[iv] = 0;

    // get the number of blocks for each dimension
    int nyblocks = yres / by;
    int nxblocks = xres / bx;

    // decode each block
    for (unsigned int index = 0; index < nelements; ++index) {
        // get the number of encoded blocks
        int nblocks = nbits[index] * block_size / 64;

        // get the encoded values
        unsigned long *encoded_values = new unsigned long[nblocks];
        for (int iv = 0; iv < nblocks; ++iv) 
            encoded_values[iv] = compressed_data[values_offsets[index] + iv];

        // create empty block array
        unsigned long *block = new unsigned long[block_size];
        for (unsigned int iv = 0; iv < block_size; ++iv)
            block[iv] = 0;

        // get the number of values per 8 bytes
        if (nbits[index]) {
            unsigned long nvalues_per_long = 64 / nbits[index];

            // for every long value
            int ib = 0;
            for (int iv = 0; iv < nblocks; ++iv) {
                unsigned long value = encoded_values[iv];
                // for every entry per 8 bytes
                for (unsigned int ie = 0; ie < nvalues_per_long; ++ie, ++ib) {
                    unsigned int lower_bits_to_remove = (nvalues_per_long - ie - 1) * nbits[index];

                    block[ib] = (value >> lower_bits_to_remove) % (int)(pow(2, nbits[index]) + 0.5);
                }
            }
        }

        // get an ordered list of unique elements
        std::unordered_set<unsigned long> hash_set = std::unordered_set<unsigned long>();

        for (unsigned int iv = 0; iv < block_size; ++iv) {
            if (!hash_set.count(block[iv])) {
                hash_set.insert(block[iv]);
            }
        }

        // get the lookup table
        unsigned int nunique = hash_set.size();
        unsigned long *lookup_table = new unsigned long[nunique];
        for (unsigned int iv = 0; iv < nunique; ++iv) {
            lookup_table[iv] = compressed_data[table_offsets[index] + iv];
        }

        // update the block values
        for (unsigned int iv = 0; iv < block_size; ++iv) {
            block[iv] = lookup_table[block[iv]];
        }

        // get the block in terms if x, y, z
        int iz = index / (nyblocks * nxblocks);
        int iy = (index - iz * nyblocks * nxblocks) / nxblocks;
        int ix = index % nxblocks;

        int iv = 0;
        for (int ik = iz * bz; ik < (iz + 1) * bz; ++ik) {
            for (int ij = iy * by; ij < (iy + 1) * by; ++ij) {
                for (int ii = ix * bx; ii < (ix + 1) * bx; ++ii, ++iv) {
                    decompressed_data[IndicesToIndex(ii, ij, ik)] = block[iv];
                }
            }
        }

        // free memory
        delete[] encoded_values;
        delete[] block;
        delete[] lookup_table;
    }


    // free memory
    delete[] table_offsets;
    delete[] nbits;
    delete[] values_offsets;

    return decompressed_data;
}
