from numba import jit
from skimage import measure
import backports.lzma as lzma
import numpy as np
import itertools
import math









# global parameters

fill_background = False



###########################
### BOUNDARY EXTRACTION ###
###########################

def ExtractBoundaries(segmentation, fill_background):
    # determine the resolution
    (zres, yres, xres) = segmentation.shape

    # create the output array
    boundaries = np.zeros((zres, yres, xres), dtype=np.uint8)

    # create a boundary map
    for iz in range(0, zres):
        zslice = segmentation[iz,:,:]
        
        xdiff = np.array(zslice[:,:-1] != zslice[:,1:], dtype=np.uint8)
        ydiff = np.array(zslice[:-1,:] != zslice[1:,:], dtype=np.uint8)

        xbound = np.zeros((yres, xres), dtype=np.uint8)
        ybound = np.zeros((yres, xres), dtype=np.uint8)

        xbound[:,:-1] = xdiff
        ybound[:-1,:] = ydiff

        diff = xbound | ybound

        boundaries[iz,:,:] = diff

    # fill in the background pixels as boundary
    if (fill_background):
        boundaries[segmentation == 0] = 1

    # return generated boundaries
    return segmentation, boundaries



########################
### LABEL EXTRACTION ###
########################

def ConnectedComponentsSlice(boundaries):
    # invert the boundaries
    boundaries = np.copy(1 - boundaries)

    # run scimage algorithm
    components = measure.label(boundaries, neighbors=4)

    # return the components
    return components



def ConnectedComponents(boundaries):
    # determine the resolution
    (zres, yres, xres) = boundaries.shape

    # create the components array
    components = np.zeros((zres, yres, xres), dtype=np.uint64)
    for iz in range(0, zres):
        components[iz,:,:] = ConnectedComponentsSlice(boundaries[iz,:,:])

    # return the connected components
    return components



def IDMapping(components, segmentation):
    # determine the resolution
    (zres, yres, xres) = components.shape


    # go through every slice
    ids = list()
    for iz in range(0, zres):
        # get the number of unique components
        unique_components, indices = np.unique(components[iz,:,:], return_index=True)
        segmentation_ = np.reshape(segmentation[iz,:,:], yres * xres)

        # add the segmentation id to the mapping
        for seg in segmentation_[indices]:
            ids.append(seg)

    # return the ids
    return np.array(ids, dtype=np.uint64)


##################################
### BOUNDARY ENCODING/DECODING ###
##################################

def ValueMapping(boundary_data):
    # map the boundary data values
    values, noccurrences = np.unique(boundary_data, return_counts=True) 
    mapping = {}

    for iv, value in enumerate(values):
        mapping[value] = iv

    for iv in range(0, len(boundary_data)):
        boundary_data[iv] = mapping[boundary_data[iv]]

    return values, boundary_data



@jit(nopython=True)
def EncodeBoundaries(boundaries, steps):
    # determine the resolution
    (zres, yres, xres) = boundaries.shape

    # get the step values in easier format
    (zstep, ystep, xstep) = steps
    assert (zstep * ystep * xstep <= 64)

    # the number of blocks in the z, y, and x dimension
    nzblocks = int(math.ceil(float(zres) / zstep) + 0.5)
    nyblocks = int(math.ceil(float(yres) / ystep) + 0.5)
    nxblocks = int(math.ceil(float(xres) / xstep) + 0.5)

    # create an empty boundary data structure
    nblocks = nzblocks * nyblocks * nxblocks
    boundary_data = np.zeros(nblocks, dtype=np.uint64)

    index = 0
    for iz in range(0, zres, zstep):
        for iy in range(0, yres, ystep):
            for ix in range(0, xres, xstep):

                # encoded value
                encoding = 0

                # go through the entire block
                ii = 0
                for iw in range(iz, iz + zstep):
                    if (iw > zres - 1): continue
                    for iv in range(iy, iy + ystep):
                        if (iv > yres - 1): continue
                        for iu in range(ix, ix + xstep):
                            if (iu > xres - 1): continue

                            # does this location contribute to the boundary
                            if (boundaries[iw,iv,iu]): encoding += 2**ii

                            # increment the index
                            ii += 1

                # add to the bit array
                boundary_data[index] = encoding
                index += 1

    return boundary_data



@jit(nopython=True)
def DecodeBoundaries(boundary_data, values, zres, yres, xres, zstep, ystep, xstep):
    # the number of blocks in the z, y, and x dimension
    zblocks = int(math.ceil(float(zres) / zstep) + 0.5)
    yblocks = int(math.ceil(float(yres) / ystep) + 0.5)
    xblocks = int(math.ceil(float(xres) / xstep) + 0.5)

    # create an empty bit data structure
    nblocks = zblocks * yblocks * xblocks
    
    # just checking..
    assert(nblocks == boundary_data.shape[0])

    # start reconstructing the boundary
    boundaries = np.zeros((zres, yres, xres), dtype=np.uint8)

    index = 0
    for iz in range(0, zres, zstep):
        for iy in range(0, yres, ystep):
            for ix in range(0, xres, xstep):
                # get the encoded value
                encoding = boundary_data[index]
                # get the true value
                value = int(values[encoding])

                assert(value == values[encoding])

                # decode the value
                for iw in range(iz, iz + zstep):
                    if (iw > zres - 1): continue
                    for iv in range(iy, iy + ystep):
                        if (iv > yres - 1): continue
                        for iu in range(ix, ix + xstep):
                            if (ix > xres - 1): continue
                            
                            # get the current bit
                            bit = (value % 2 == 1)

                            # determine if there is a boundary
                            if (bit): boundaries[iw,iv,iu] = 1

                            # shift over the value
                            value = value >> 1


                # update the current index
                index += 1

    # return the boundaries
    return boundaries



#######################
### REVERSE MAP IDS ###
#######################

@jit(nopython=True)
def IDUpdate(decompressed_data, mapping, components):
    (yres, xres) = components.shape

    for iy in range(0, yres):
        for ix in range(0, xres):
            decompressed_data[iy,ix] = mapping[components[iy,ix]]

    return decompressed_data



def IDReverseMapping(components, ids):
    # determine the resolution
    (zres, yres, xres) = components.shape

    decompressed_data = np.zeros((zres, yres, xres), dtype=np.int64) - 1

    # get the number of components
    ids_index = 0
    for iz in range(0, zres):
        # get the number of components
        unique_components = np.unique(components[iz,:,:])

        # create a reverse mapping
        mapping = [-1] * (len(unique_components) + 1)

        # go through every component
        for component in unique_components:
            mapping[component] = ids[ids_index]
            ids_index += 1

        decompressed_data[iz,:,:] = IDUpdate(decompressed_data[iz,:,:], np.array(mapping), components[iz,:,:])

    # return the decompressed data
    return decompressed_data




###############################
### INDETERMINATE LOCATIONS ###
###############################

@jit(nopython=True)
def GetOffset(iu, iv):
    if (iu == -1 and iv == -1): return 0
    if (iu == -1 and iv == 0): return 1
    if (iu == -1 and iv == 1): return 2
    if (iu == 0 and iv == -1): return 3
    if (iu == 0 and iv == 1): return 4
    if (iu == 1 and iv == -1): return 5
    if (iu == 1 and iv == 0): return 6
    if (iu == 1 and iv == 1): return 7



@jit(nopython=True)
def GetIndices(offset):
    if offset == 0: return (0, 0, -1)
    if offset == 1: return (0, 0, 1)
    if offset == 2: return (0, -1, 0)
    if offset == 3: return (0, 1, 0)
    if offset == 4: return (-1, 0, 0)
    if offset == 5: return (1, 0, 0)



def FindIndeterminateLocations(boundaries, segmentation):
    (zres, yres, xres) = boundaries.shape

    # create an array of ambiguous locations
    locations = list()

    # find all the ambiguous values
    for iz in range(0, zres):
        diff = boundaries[iz,:,:]

        # find the second derivative of the difference
        xdiff = np.array(diff[:,:-1] == diff[:,1:], dtype=np.uint8)
        ydiff = np.array(diff[:-1,:] == diff[1:,:], dtype=np.uint8)

        xbound = np.ones((yres, xres), dtype=np.uint8)
        ybound = np.ones((yres, xres), dtype=np.uint8)

        xbound[:,1:] = xdiff
        ybound[1:,:] = ydiff

        second_derivative = xbound + ybound
        
        diff[np.where(second_derivative != 2)] = 0

        indeterminate_locations = np.where(diff == 1)

        for (iy, ix) in itertools.izip(indeterminate_locations[0], indeterminate_locations[1]):
            # go through neighbors to find one similar location

            if (ix > 0 and segmentation[iz,iy,ix] == segmentation[iz,iy,ix-1] and not boundaries[iz,iy,ix-1]):
                locations.append(0)
            elif (ix < xres - 1 and segmentation[iz,iy,ix] == segmentation[iz,iy,ix+1] and not boundaries[iz,iy,ix+1]):
                locations.append(1)
            elif (iy > 0 and segmentation[iz,iy,ix] == segmentation[iz,iy-1,ix] and not boundaries[iz,iy-1,ix]):
                locations.append(2)
            elif (iy < yres - 1 and segmentation[iz,iy,ix] == segmentation[iz,iy+1,ix] and not boundaries[iz,iy+1,ix]):
                locations.append(3)
            elif (iz > 0 and segmentation[iz,iy,ix] == segmentation[iz-1,iy,ix] and not boundaries[iz-1,iy,ix]):
                locations.append(4)
            elif (iz < zres - 1 and segmentation[iz,iy,ix] == segmentation[iz+1,iy,ix] and not boundaries[iz+1,iy,ix]):
                locations.append(5)
            else:
                locations.append(segmentation[iz,iy,ix] + 6)

            # update the boundary matrix
            boundaries[iz,iy,ix] = 0

    # return the list of locations
    return np.array(locations, dtype=np.uint64)



def DecodeBoundaryLocations(decompressed_data, boundaries, locations):
    (zres, yres, xres) = decompressed_data.shape

    # add in the boundaries
    locations_index = 0

    # take care of the easy locations
    for iz in range(0, zres):
        for iy in range(0, yres):
            for ix in range(0, xres):
                if not boundaries[iz,iy,ix]: continue

                if (ix > 0 and not boundaries[iz,iy,ix-1]):
                    decompressed_data[iz,iy,ix] = decompressed_data[iz,iy,ix-1]
                elif (iy > 0 and not boundaries[iz,iy-1,ix]):
                    decompressed_data[iz,iy,ix] = decompressed_data[iz,iy-1,ix]

    for iz in range(0, zres):
        diff = boundaries[iz,:,:]

        # find the second derivative of the difference
        xdiff = np.array(diff[:,:-1] == diff[:,1:], dtype=np.uint8)
        ydiff = np.array(diff[:-1,:] == diff[1:,:], dtype=np.uint8)

        xbound = np.ones((yres, xres), dtype=np.uint8)
        ybound = np.ones((yres, xres), dtype=np.uint8)

        xbound[:,1:] = xdiff
        ybound[1:,:] = ydiff

        second_derivative = xbound + ybound
        
        diff[np.where(second_derivative != 2)] = 0

        indeterminate_locations = np.where(diff == 1)
        for (iy, ix) in itertools.izip(indeterminate_locations[0], indeterminate_locations[1]):
            offset = locations[locations_index]
            if (offset < 6):
                (iw, iv, iu) = GetIndices(offset)
                decompressed_data[iz,iy,ix] = decompressed_data[iz+iw,iy+iv,ix+iu]
            else:
                decompressed_data[iz,iy,ix] = offset - 6
            
            locations_index += 1

    # return the final dataset
    return decompressed_data



#######################################
### ENCODE/DECODE CLASS DEFINITIONS ###
#######################################

class boundary_encoding(object):

    @staticmethod
    def name():
        return 'Boundary Encoding'


    @staticmethod
    def compress(data, compress=False):
        '''Boundary Encoding compression
        '''

        # size of window for boundary encoding
        steps = (1, 8, 8)

        # get the boundary image
        segmentation, boundaries = ExtractBoundaries(data, fill_background)
        components = ConnectedComponents(boundaries)
        ids = IDMapping(components, segmentation)
        boundary_data = EncodeBoundaries(boundaries, steps)
        values, boundary_data = ValueMapping(boundary_data)
        locations = FindIndeterminateLocations(boundaries, segmentation)

        # get additional header information
        (zres, yres, xres) = boundaries.shape
        (zstep, ystep, xstep) = steps

        # construct the header
        header = np.zeros(9, dtype=np.uint64)
        header[0] = zres
        header[1] = yres
        header[2] = xres

        header[6] = zstep
        header[7] = ystep
        header[8] = xstep

        if (not compress):

            header[3] = ids.size
            header[4] = values.size
            header[5] = locations.size

            return np.concatenate((header, ids, values, locations, boundary_data))

        # update the size of the boundary data
        if (len(values) <= 2**8):
            boundary_data = boundary_data.astype(np.uint8)
        elif (len(values) <= 2**16):
            boundary_data = boundary_data.astype(np.uint16)
        elif (len(values) <= 2**32):
            boundary_data = boundary_data.astype(np.uint32)

        # compress values and boundary data
        compressed_ids = lzma.compress(ids.tobytes())
        compressed_values = lzma.compress(values.tobytes())
        compressed_boundaries = lzma.compress(boundary_data.tobytes())
        compressed_locations = lzma.compress(locations.tobytes())

        header[3] = len(compressed_ids)
        header[4] = len(compressed_values)
        header[5] = len(compressed_locations)

        # send the header to bytes
        compressed_header = header.tobytes()

        # concatenate all of the comrpessed information
        compressed_data = compressed_header + compressed_ids + compressed_values + compressed_locations + compressed_boundaries

        # return the compressed data
        return compressed_data

    @staticmethod
    def decompress(data, compress=False):
        '''Boundary Decoding decompression
        '''

        # constants
        byte_size = 8
        header_size = 9

        if not compress:
            header = data[0:header_size]
            data = data[header_size:]

            # get information regarding locations
            zres, yres, xres = (int(header[0]), int(header[1]), int(header[2]))
            ids_size, values_size, locations_size = (int(header[3]), int(header[4]), int(header[5]))
            zstep, ystep, xstep = (int(header[6]), int(header[7]), int(header[8]))

            ids = data[0:ids_size]
            data = data[ids_size:]
            values = data[0:values_size]
            data = data[values_size:]
            locations = data[0:locations_size]
            data = data[locations_size:]
            boundary_data = data[0:]

        else:
            # extract the header
            header_string = data[0:header_size*byte_size]
            header = np.fromstring(header_string, dtype=np.uint64)

            # get information regarding locations
            zres, yres, xres = (int(header[0]), int(header[1]), int(header[2]))
            ids_length, values_length, locations_length = (int(header[3]), int(header[4]), int(header[5]))
            zstep, ystep, xstep = (int(header[6]), int(header[7]), int(header[8]))

            # remove the header
            data = data[header_size*byte_size:]

            # retrieve all of the ids
            ids_string = data[0:ids_length]
            ids_string = lzma.decompress(ids_string)#, lzma.FORMAT_RAW, filters=lzma_filters)
            ids = np.fromstring(ids_string, dtype=np.uint64)

            # remove the ids
            data = data[ids_length:]

            # retrieve all of the values
            values_string = data[0:values_length]
            values_string = lzma.decompress(values_string)#, lzma.FORMAT_RAW, filters=lzma_filters)
            values = np.fromstring(values_string, dtype=np.uint64)

            # remove the values
            data = data[values_length:]

            # retrieve all the locations
            locations_string = data[0:locations_length]
            locations_string = lzma.decompress(locations_string)#, lzma.FORMAT_RAW, filters=lzma_filters)
            locations = np.fromstring(locations_string, dtype=np.uint64)

            # remove the locations
            data = data[locations_length:]
     
            # retrieve all of the boundary encodings
            boundary_string = data[:]
            boundary_string = lzma.decompress(boundary_string)#, lzma.FORMAT_RAW, filters=lzma_filters)
            if (len(values) < 2**8):
                boundary_data = np.fromstring(boundary_string, dtype=np.uint8)
            elif (len(values) < 2**16):
                boundary_data = np.fromstring(boundary_string, dtype=np.uint16)
            elif (len(values) < 2**32):
                boundary_data = np.fromstring(boundary_string, dtype=np.uint32)
            else:
                boundary_data = np.fromstring(boundary_string, dtype=np.uint64)

        boundaries = DecodeBoundaries(boundary_data, values, zres, yres, xres, zstep, ystep, xstep)
        components = ConnectedComponents(boundaries)
        decompressed_data = IDReverseMapping(components, ids)
        decompressed_data = DecodeBoundaryLocations(decompressed_data, boundaries, locations)

        return decompressed_data