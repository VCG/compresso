import glob
import h5py
import mahotas as mh
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from methods import (
    LZMA,
    NC,
    ZLIB
)
from ne import NE
from rle import RLE


class Util(object):

    DIM_AC3 = (75, 1024, 1024)
    DIM_CYL = (300, 2048, 2048)
    DIM_MRI = (250, 250, 250)
    DIM_CREMI = (125, 1250, 1250)

    @staticmethod
    def adj_fig_size(width=10, height=10):
        '''Adjust figsize of plot
        '''

        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = width
        fig_size[1] = height
        plt.rcParams["figure.figsize"] = fig_size

    @staticmethod
    def load_data(name='ac3', N=-1, prefix=None, gold=False):
        '''Load data
        '''

        dataset = []

        if gold:

            if name == 'ac3':
                d_path = prefix if prefix else '~/data/ac3/gold/'
                dataset = [
                    Util.DIM_AC3,
                    os.path.expanduser(d_path + 'ac3_labels_*.tif')
                ]
            if name == 'cylinder':
                d_path = prefix if prefix else '~/data/cylinderNEW/gold/'
                dataset = [
                    Util.DIM_CYL, os.path.expanduser(d_path + '*.png')
                ]

            if name == 'ac4':
                # open the h5 file
                d_path = prefix if prefix else '~/data/ac4/gold/'
                full_path = os.path.expanduser(d_path)
                hf = h5py.File(full_path + 'human_labels.h5', 'r')

                # create numpy dataset
                dataset = np.array(hf['stack'])

                # close the file
                hf.close()

                # crop dataset
                if (N == -1):
                    return dataset
                else:
                    return dataset[0:N, :, :]

        else:

            if name == 'ac3':
                d_path = prefix if prefix else '~/data/ac3/rhoana/'
                dataset = [
                    Util.DIM_AC3, os.path.expanduser(d_path + 'z=*.tif')
                ]
            if name == 'cylinder':
                d_path = prefix if prefix else '~/data/cylinderNEW/rhoana/'
                dataset = [
                    Util.DIM_CYL, os.path.expanduser(d_path + '*.png')
                ]
            if name == 'mri':
                d_path = prefix if prefix else '~/data/'
                dataset = [
                    Util.DIM_MRI, os.path.expanduser(d_path + 'mri.npy')
                ]

                return np.load(dataset[-1])

            if name == 'cremi':
                # open the h5 file
                d_path = prefix if prefix else '~/data/CREMI/A/rhoana/'
                dataset = [
                    Util.DIM_CREMI, os.path.expanduser(d_path + '*.png')
                ]

        if len(dataset) == 0:
            print 'Could not find dataset'
            return None

        if N == -1:
            N = dataset[0][0]

        Y = dataset[0][1]
        X = dataset[0][2]
        path = dataset[1]

        output = np.zeros((N, Y, X), dtype=np.uint64)

        all_imgs = sorted(glob.glob(path))

        for z in range(N):

            img = mh.imread(all_imgs[z])

            if name == 'cylinder' or name == 'cremi':
                img = (
                    img[:, :, 0] * 256 * 256 +
                    img[:, :, 1] * 256 +
                    img[:, :, 2]
                )

            output[z] = img

        if name == 'cremi':
            # fix for now to make multiple of 8
            output = output[:, :1024, :1024]
            return output.copy(order='C')

        return output

    @staticmethod
    def colorize(slice):
        colors = np.zeros(slice.shape + (3,), dtype=np.uint8)
        colors[:, :, 0] = np.mod(107 * slice[:, :], 700).astype(np.uint8)
        colors[:, :, 1] = np.mod(509 * slice[:, :], 900).astype(np.uint8)
        colors[:, :, 2] = np.mod(200 * slice[:, :], 777).astype(np.uint8)
        return colors

    @staticmethod
    def convert_to_rgb(img):

        colorized = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        colorized[:, :, 0] = img % (2**8)
        img = img >> 8
        colorized[:, :, 1] = img % (2**8)
        img = img >> 8
        colorized[:, :, 2] = img % (2**8)

        return colorized

    @staticmethod
    def convert_from_rgb(frame):

        img_ = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint64)
        img_[:] = (
            frame[:, :, 0] +
            frame[:, :, 1] * 256 +
            frame[:, :, 2] * 256 * 256
        )

        return img_

    @staticmethod
    def get_size(variable):
        '''Get bytes of variable
        '''
        if type(variable).__module__ == np.__name__:
            variable = variable.tobytes()

        # checking the length of a bytestring is more accurate
        return len(variable)

    @staticmethod
    def to_best_type(array):
        '''Convert array to lowest possible bitrate.
        '''
        ui8 = np.iinfo(np.uint8(10))
        ui8 = ui8.max
        ui16 = np.iinfo(np.uint16(10))
        ui16 = ui16.max
        ui32 = np.iinfo(np.uint32(10))
        ui32 = ui32.max
        ui64 = np.iinfo(np.uint64(10))
        ui64 = ui64.max

        if array.max() <= ui64:
            new_type = np.uint64
        if array.max() <= ui32:
            new_type = np.uint32
        if array.max() <= ui16:
            new_type = np.uint16
        if array.max() <= ui8:
            new_type = np.uint8

        return array.astype(new_type)

    @staticmethod
    def compress(method, data, *args, **kwargs):
        '''Compress data
        '''

        t0 = time.time()

        compressed_data = method.compress(data, *args, **kwargs)

        return compressed_data, time.time() - t0

    @staticmethod
    def decompress(method, compressed_data, *args, **kwargs):
        '''Decompress data
        '''

        t0 = time.time()

        data = method.decompress(compressed_data, *args, **kwargs)

        return data, time.time() - t0

    @staticmethod
    def encode(method, data):
        '''Encode data
        '''

        t0 = time.time()

        enc_data = method.encode(data)

        return enc_data, time.time() - t0

    @staticmethod
    def decode(method, enc_data):
        '''Decode data
        '''

        t0 = time.time()

        data = method.decode(enc_data)

        return data, time.time() - t0

    @staticmethod
    def benchmark(com, enc, data, N=100, verbose=True, *args, **kwargs):
        '''Benchmark one compression method
        com = [BROTLI, BZ2, LZMA, LZO, ZLIB, ZSTD]
        data = data set
        N = number of runs
        *args / **kwargs = settings for the compression method, e.g. `9`
        '''

        enc_speed = []
        denc_speed = []

        com_speed = []
        dcom_speed = []

        total_com_speed = []
        total_dcom_speed = []

        original_data = data.copy()

        for n in range(N):

            data, t1 = Util.encode(enc, data)

            compressed_data, t2 = Util.compress(com, data, *args)
            decompressed_data, t3 = Util.decompress(com, compressed_data)

            if isinstance(data, (np.ndarray, np.generic)):
                # convert back to numpy array
                decompressed_data = np.fromstring(
                    decompressed_data, dtype=data.dtype
                )

            decompressed_data = enc.decode(decompressed_data)
            data, t4 = Util.decode(enc, data)

            enc_speed.append(t1)
            denc_speed.append(t4)

            com_speed.append(t2)
            dcom_speed.append(t3)

            total_com_speed.append(t1 + t2)
            total_dcom_speed.append(t3 + t4)

        enc_MB = Util.get_size(compressed_data) / float(1000**2)
        dec_MB = Util.get_size(data) / float(1000**2)

        # Higher is better
        ratio = dec_MB / enc_MB

        if np.mean(com_speed):
            com_speed = dec_MB / np.mean(com_speed)
        else:
            com_speed = np.inf

        if np.mean(dcom_speed):
            dcom_speed = dec_MB / np.mean(dcom_speed)
        else:
            dcom_speed = np.inf

        if np.mean(enc_speed):
            enc_speed = dec_MB / np.mean(enc_speed)
        else:
            enc_speed = np.inf

        if np.mean(denc_speed):
            denc_speed = dec_MB / np.mean(denc_speed)
        else:
            denc_speed = np.inf

        total_com_speed = dec_MB / np.mean(total_com_speed)
        total_dcom_speed = dec_MB / np.mean(total_dcom_speed)

        if verbose:
            print '> %s + %s' % (enc.name(), com.name())
            print 'Compression Method:', com.name()
            print 'Encoding Method:', enc.name()
            print 'Input Size:', dec_MB, 'MB'
            print 'Output Size:', enc_MB, 'MB'
            print 'Ratio:', ratio
            print 'Total Compression Speed [MB/s]:', total_com_speed
            print 'Total Decompression Speed [MB/s]:', total_dcom_speed
            print 'Compression (Only) Speed [MB/s]:', com_speed
            print 'Decompression (Only) Speed [MB/s]:', dcom_speed
            print 'Encoding Speed [MB/s]:', enc_speed
            print 'Decoding Speed [MB/s]:', denc_speed
            print ''

        assert np.array_equal(
            original_data.flatten(), decompressed_data.flatten()
        )

        return (
            compressed_data,
            ratio,
            com_speed,
            dcom_speed,
            enc_speed,
            denc_speed,
            total_com_speed,
            total_dcom_speed
        )

    @staticmethod
    def run_experiments(
        data, N=100, com_alg=None, enc_alg=None, plot=False, verbose=False
    ):
        data_bytes = []
        ratios = []
        # Compression times
        com_speed = []
        com_speed_stderr = []
        # Decompression times
        dcom_speed = []
        dcom_speed_stderr = []
        # Encoding times
        enc_speed = []
        enc_speed_stderr = []
        # Decoding times
        dec_speed = []
        dec_speed_stderr = []
        # Total compression times
        total_com_speed = []
        total_com_speed_stderr = []
        # Total decompression times
        total_dec_speed = []
        total_dec_speed_stderr = []

        com_alg = com_alg if com_alg else [
            LZMA, NC, ZLIB
        ]
        enc_alg = enc_alg if enc_alg else [
            NE, RLE
        ]
        method_labels = []

        for com in com_alg:
            for enc in enc_alg:
                if enc == NE:
                    method_labels.append(com.name())
                else:
                    # TODO FIX THIS
                    if (
                        com.name() == 'Bockwurst' or
                        com.name() == 'Neuroglancer' or
                        com.name() == 'JPEG2000' or
                        com.name() == 'PNG' or
                        com.name() == 'Boundary Encoding' or
                        com.name() == 'Variable Encoding' or
                        com.name() == 'X.264'
                    ):
                        if enc.name() is not 'NE':
                            continue

                    if com.name() == 'NC':
                        method_labels.append(enc.name())
                    else:
                        method_labels.append(
                            com.name() + ' with ' + enc.name()
                        )

        for com in com_alg:
            for enc in enc_alg:
                # TODO FIX THIS
                if (
                    com.name() == 'Bockwurst' or
                    com.name() == 'Neuroglancer' or
                    com.name() == 'JPEG2000' or
                    com.name() == 'PNG' or
                    com.name() == 'Boundary Encoding' or
                    com.name() == 'Variable Encoding' or
                    com.name() == 'X.264'
                ):
                    if enc.name() is not 'NE':
                        continue

                if com.name() == 'Zlib' and data.shape[0] == Util.DIM_CYL[0]:
                    print '=================================================='
                    print 'WARNING! ONLY 50 SLICES ARE GIVEN TO ZLIB'
                    print '=================================================='
                    _data = data[:50]
                else:
                    _data = data

                b, r, ct, dct, et, det, tct, tdct = Util.benchmark(
                    com=com,
                    enc=enc,
                    data=_data,
                    N=N,
                    verbose=verbose
                )

                data_bytes.append(Util.get_size(b))
                ratios.append(r)
                com_speed.append(np.mean(ct))
                com_speed_stderr.append(np.std(ct))
                dcom_speed.append(np.mean(dct))
                dcom_speed_stderr.append(np.std(dct))
                enc_speed.append(np.mean(ct))
                enc_speed_stderr.append(np.std(ct))
                dec_speed.append(np.mean(dct))
                dec_speed_stderr.append(np.std(dct))

                total_com_speed.append(np.mean(tct))
                total_com_speed_stderr.append(np.std(tct))
                total_dec_speed.append(np.mean(tdct))
                total_dec_speed_stderr.append(np.std(tdct))

        if plot:
            Util.plot(
                method_labels,
                data_bytes,
                ratios,
                total_com_speed,
                total_com_speed_stderr,
                total_dec_speed,
                total_dec_speed_stderr
            )

        return {
            'methods': method_labels,
            'orig_bytes': Util.get_size(data),
            'comp_bytes': data_bytes,
            'ratios': ratios,
            'comp_speed': com_speed,
            'comp_speed_std': com_speed_stderr,
            'decomp_speed': dcom_speed,
            'decomp_speed_std': dcom_speed_stderr,
            'enc_speed': enc_speed,
            'enc_speed_std': enc_speed_stderr,
            'dec_speed': dec_speed,
            'dec_speed_std': dec_speed_stderr,
            'total_comp_speed': total_com_speed,
            'total_comp_speed_std': total_com_speed_stderr,
            'total_decomp_speed': total_dec_speed,
            'total_decomp_speed_std': total_dec_speed_stderr
        }

    @staticmethod
    def plot(
        method_labels,
        data_bytes,
        ratios,
        com_speed,
        com_speed_stderr,
        dcom_speed,
        dcom_speed_stderr,
        save='',
        dpi=300,
        bw=False
    ):
        '''Plot results
        '''

        # Plots
        index = np.arange(len(method_labels))
        bar_width = 0.35
        opacity = 1
        # colors = ['#333333'] * len(method_labels)

        # FIGURE SIZE
        Util.adj_fig_size(20, 9)

        # FONT
        font = {
            'family': 'sans-serif',
            'size': 18
        }
        plt.rc('font', **font)

        fig = plt.figure()

        fig = plt.figure()
        fig, ax = plt.subplots()
        ax.tick_params(color='#333333', labelcolor='#333333')
        for spine in ax.spines.values():
            spine.set_edgecolor('#999999')
        plt.bar(
            left=index,
            height=ratios,
            width=bar_width,
            alpha=opacity,
            color='k' if bw else 'r',
            label='Compression Rate\n(Output Bytes / Input Bytes)',
            align='center'
        )
        plt.ylabel('Compression Rate')
        plt.xticks(index, method_labels, rotation=75)
        plt.xlim([index[0] - 1, len(index)])
        plt.legend(
            loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True
        )
        plt.tight_layout()
        if len(save) > 0:
            plt.savefig('%s_comp_rate.eps' % save, format='eps', dpi=dpi)

        fig, ax = plt.subplots()
        plt.bar(
            left=index,
            height=com_speed,
            width=bar_width,
            alpha=opacity,
            color='k' if bw else 'g',
            yerr=com_speed_stderr,
            label='Compression Speed\n(Encoding + Compression)',
            align='center'
        )
        plt.ylabel('Speed [MB/s]')
        plt.xticks(index, method_labels, rotation=75)
        plt.xlim([index[0] - 1, len(index)])
        plt.legend(
            loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True
        )
        plt.tight_layout()
        if len(save) > 0:
            plt.savefig('%s_comp_speed.eps' % save, format='eps', dpi=dpi)

        fig = plt.figure()
        fig, ax = plt.subplots()
        plt.bar(
            left=index,
            height=dcom_speed,
            width=bar_width,
            alpha=opacity,
            color='k' if bw else 'b',
            yerr=dcom_speed_stderr,
            label='Decompression Speed\n(Decoding + Decompression)',
            align='center'
        )
        plt.ylabel('Speed [MB/s]')
        plt.xticks(index, method_labels, rotation=75)
        plt.xlim([index[0] - 1, len(index)])
        plt.legend(
            loc='upper center', bbox_to_anchor=(0.5, 1.2), fancybox=True
        )
        plt.xticks(rotation=75)
        plt.tight_layout()
        if len(save) > 0:
            plt.savefig('%s_decomp_speed.eps' % save, format='eps', dpi=dpi)

        fig = plt.figure()
        fig, ax = plt.subplots()
        plt.bar(
            left=np.arange(len(data_bytes)),
            height=[Util.get_size(wurst) for wurst in data_bytes],
            width=bar_width,
            alpha=opacity,
            color='k' if bw else 'y',
            label='Data Size',
            align='center'
        )
        plt.ylabel('Bytes')
        ax.set_yscale('log')
        plt.xticks(
            np.arange(len(data_bytes)),
            ['Uncompressed'] + method_labels,
            rotation=75
        )
        plt.xlim([index[0] - 1, len(data_bytes)])
        plt.legend(
            loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True
        )
        plt.tight_layout()
        if len(save) > 0:
            plt.savefig('%s_data_size.eps' % save, format='eps', dpi=dpi)
