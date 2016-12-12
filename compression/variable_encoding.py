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

def EncodeBoundaries(boundaries, steps):
    # determine the resolution
    (zres, yres, xres) = boundaries.shape

    # get the step values in easier format
    (zstep, ystep, xstep) = steps
  
    assert (zres % zstep == 0)
    assert (yres % ystep == 0)
    assert (xres % xstep == 0)
    #assert ((zstep * ystep * xstep) % 8 == 0)

    window_values = list()
    values = set()

    # go through the entire segment
    for iz in range(0, zres, zstep):
        for iy in range(0, yres, ystep):
            for ix in range(0, xres, xstep):
                # retrieve the window
                window = boundaries[iz:iz+zstep,iy:iy+ystep,ix:ix+xstep]

                # flatten the window
                window = window.flatten(order='C')

                # generate the value for this window
                value = 0
                for ib, bit in enumerate(window):
                    if (bit): value += (2**ib)

                window_values.append(value)
                values.add(value)

    # calculate the mapping for all values
    values = list(sorted(values))
    mapping = {}
    for iv, value in enumerate(values):
        mapping[value] = iv

    # get the maximum value and the number of values
    N = len(values)
    max_value = values[N - 1]

    # encode the boundaries with the new values
    if (N < 2**8):
        dtype=np.uint8
    elif (N < 2**16):
        dtype=np.uint16
    elif (N < 2**32):
        dtype=np.uint32
    elif (N < 2**64):
        dtype=np.uint64
    else:
        assert (False)

    # create the empty boundary data
    nwindows = len(window_values)
    boundary_data = np.zeros(nwindows, dtype=dtype)

    # populate the boundary data array
    for iw, value in enumerate(window_values):
        boundary_data[iw] = mapping[value]

    # create the value encoding
    nbytes_per_value = int(math.ceil(math.log(max_value, 2) / 8) + 0.5)

    # encode the values in a byte array
    np_values = np.zeros(nbytes_per_value * N, dtype=np.uint8)

    index = 0
    for value in values:
        for byte in range(0, nbytes_per_value):
            last_byte = value % 256
            value = value >> 8

            np_values[index] = last_byte
            index += 1

    return boundary_data, np_values, nbytes_per_value




def DecodeBoundaries(boundary_data, values, zres, yres, xres, zstep, ystep, xstep, nbytes_per_value):
    # get the number of values
    N = values.size / nbytes_per_value
    
    np_values = np.split(values, N)

    mapping = {}
    for iv, value in enumerate(np_values):
        encoding = 0
        for ib, byte in enumerate(value):
            encoding += (long(byte) * (2**(8*ib)))
        mapping[iv] = encoding

    # get the window size
    window_size = zstep * ystep * xstep

    # create empty boundaries array
    boundaries = np.zeros((zres, yres, xres), dtype=np.uint8)

    # go through all of the values in boundary data
    index = 0
    for iz in range(0, zres, zstep):
        for iy in range(0, yres, ystep):
            for ix in range(0, xres, xstep):
                value = mapping[boundary_data[index]]

                # create an empty window
                window = np.zeros(window_size, dtype=np.uint8)

                # decode the value
                for ib in range(0, window_size):
                    window[ib] = (value % 2)
                    value = value >> 1

                window = window.reshape((zstep, ystep, xstep), order='C')

                boundaries[iz:iz+zstep,iy:iy+ystep,ix:ix+xstep] = window

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

class variable_encoding(object):

    @staticmethod
    def name():
        return 'Variable Encoding'


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
        boundary_data, values, nbytes_per_value = EncodeBoundaries(boundaries, steps)
        locations = FindIndeterminateLocations(boundaries, segmentation)

        # get additional header information
        (zres, yres, xres) = boundaries.shape
        (zstep, ystep, xstep) = steps

        # construct the header
        header = np.zeros(10, dtype=np.uint64)
        header[0] = zres
        header[1] = yres
        header[2] = xres
        header[3] = ids.size
        header[4] = values.size
        header[5] = locations.size
        header[6] = zstep
        header[7] = ystep
        header[8] = xstep
        header[9] = nbytes_per_value

        return np.concatenate((header, ids, values, locations, boundary_data))


    @staticmethod
    def decompress(data):
        '''Boundary Decoding decompression
        '''

        # constants
        byte_size = 8
        header_size = 10

        header = data[0:header_size]
        data = data[header_size:]

        # get information regarding locations
        zres, yres, xres = (int(header[0]), int(header[1]), int(header[2]))
        ids_size, values_size, locations_size = (int(header[3]), int(header[4]), int(header[5]))
        zstep, ystep, xstep = (int(header[6]), int(header[7]), int(header[8]))
        nbytes_per_value = int(header[9])

        ids = data[0:ids_size]
        data = data[ids_size:]
        values = data[0:values_size]
        data = data[values_size:]
        locations = data[0:locations_size]
        data = data[locations_size:]
        boundary_data = data[0:]

        boundaries = DecodeBoundaries(boundary_data, values, zres, yres, xres, zstep, ystep, xstep, nbytes_per_value)
        components = ConnectedComponents(boundaries)
        decompressed_data = IDReverseMapping(components, ids)
        decompressed_data = DecodeBoundaryLocations(decompressed_data, boundaries, locations)

        return decompressed_data