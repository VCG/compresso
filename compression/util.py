import glob
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import sys

from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap


class Util(object):

    @staticmethod
    def adj_fig_size(width=10, height=10):
        '''Adjust figsize of plot
        '''

        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = width
        fig_size[1] = height
        plt.rcParams["figure.figsize"] = fig_size

    @staticmethod
    def colorize(slice):
        colorized = np.zeros(slice.shape + (3,), dtype=np.uint8)

        colorized[:, :, 0] = np.mod(107 * slice[:, :], 700).astype(np.uint8)
        colorized[:, :, 1] = np.mod(509 * slice[:, :], 900).astype(np.uint8)
        colorized[:, :, 2] = np.mod(200 * slice[:, :], 777).astype(np.uint8)

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
    def convert_to_rgba(img):
        colorized = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)

        colorized[:, :, 0] = img % (2**8)
        img = img >> 8
        colorized[:, :, 1] = img % (2**8)
        img = img >> 8
        colorized[:, :, 2] = img % (2**8)
        img = img >> 8
        colorized[:, :, 3] = img % (2**8)

        return colorized

    @staticmethod
    def convert_from_rgb(frame):

        img = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint64)
        img[:] = (np.uint64(frame[:, :, 0]) + np.uint64(frame[:, :, 1]) * 256 + np.uint64(frame[:, :, 2]) * 256 * 256)

        return img

    @staticmethod
    def convert_from_rgba(frame):
        img = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint64)
        img[:] = (frame[:, :, 0] + frame[:, :, 1] * 256 + frame[:, :, 2] * 256 * 256 + frame[:, :, 3] * 256 * 256 * 256)

        return img

    @staticmethod
    def get_size(variable):
        '''Get bytes of variable
        '''
        if type(variable).__module__ == np.__name__:
            variable = variable.tobytes()
        elif type(variable) is str:
            assert (all(ord(c) < 256) for c in variable)
        else:
            raise ValueError('Data type not supported')

        # checking the length of a bytestring is more accurate
        return len(variable)

    @staticmethod
    def to_best_type(array):
        '''Convert array to lowest possible bitrate.
        '''
        ui8 = np.iinfo(np.uint8)
        ui8 = ui8.max
        ui16 = np.iinfo(np.uint16)
        ui16 = ui16.max
        ui32 = np.iinfo(np.uint32)
        ui32 = ui32.max
        ui64 = np.iinfo(np.uint64)
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
    def load_data(name='ac3', N=-1, prefix=None, gold=False):
        '''Load data
        '''

        if not 'mri' in name:
            if gold: filename = '~/compresso/data/' + name + '/gold/' + name + '_gold.h5'
            else: filename = '~/compresso/data/' + name + '/rhoana/' + name + '_rhoana.h5'

            with h5py.File(os.path.expanduser(filename), 'r') as hf:
                output = np.array(hf['main'], dtype=np.uint64)
        else:
            filename = '~/compresso/data/MRI/' + name + '.h5'

            with h5py.File(os.path.expanduser(filename), 'r') as hf:
                output = np.array(hf['main'], dtype=np.uint64)

        if (not N == -1):
            output = output[0:N,:,:]

        return output

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
    def encode(method, data, *args, **kwargs):
        '''Encode data
        '''
        t0 = time.time()

        enc_data = method.compress(data, *args, **kwargs)

        return enc_data, time.time() - t0

    @staticmethod
    def decode(method, enc_data, *args, **kwargs):
        '''Decode data
        '''
        t0 = time.time()

        data = method.decompress(enc_data, *args, **kwargs)

        return data, time.time() - t0

    @staticmethod
    def run_experiment(com, enc, data, N=100, verbose=True, *args, **kwargs):
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

        # run N iterations
        for n in range(N):
            start_time = time.time()

            # run encoding and compression
            encoded_data, t1 = Util.encode(enc, data)
            compressed_data, t2 = Util.compress(com, encoded_data, *args)

            # decompress data
            if (com.name() == 'LZF'):
                # LZF requires the original output size
                decompressed_data, t3 = Util.decompress(com, compressed_data, 8 * long(data.size))
            else:
                decompressed_data, t3 = Util.decompress(com, compressed_data)

            # make sure the data is returned as an array
            if not isinstance(decompressed_data, (np.ndarray, np.generic)) and isinstance(encoded_data, (np.ndarray, np.generic)):
                # convert back to numpy array
                decompressed_data = np.fromstring(decompressed_data, dtype=encoded_data.dtype)

            # decode the data
            decoded_data, t4 = Util.decode(enc, decompressed_data)

            # update the speed lists
            enc_speed.append(t1)
            denc_speed.append(t4)

            com_speed.append(t2)
            dcom_speed.append(t3)

            total_com_speed.append(t1 + t2)
            total_dcom_speed.append(t3 + t4)

            # guarantee lossless behavior
            assert np.array_equal(np.ndarray.flatten(data), np.ndarray.flatten(decoded_data))

            print 'Ran iteration ' + str(n + 1) + ' of ' + str(N) + ' on ' + enc.name() + ' + ' + com.name() + ' in %0.2f seconds' % (time.time() - start_time)
            sys.stdout.flush()

        com_MB = Util.get_size(compressed_data) / float(1000**2)
        dec_MB = Util.get_size(data) / float(1000**2)

        # Higher is better
        ratio = dec_MB / com_MB

        # turn the speeds in MB / s
        for n in range(N):
            if enc_speed[n] == 0:
                enc_speed[n] = 0.01
            if denc_speed[n] == 0:
                denc_speed[n] = 0.01
            if com_speed[n] == 0:
                com_speed[n] = 0.01
            if dcom_speed[n] == 0:
                dcom_speed[n] = 0.01
            if total_com_speed[n] == 0:
                total_com_speed[n] = 0.01
            if total_dcom_speed[n] == 0:
                total_dcom_speed[n] = 0.01

            enc_speed[n] = dec_MB / enc_speed[n]
            denc_speed[n] = dec_MB / denc_speed[n]

            com_speed[n] = dec_MB / com_speed[n]
            dcom_speed[n] = dec_MB / dcom_speed[n]

            total_com_speed[n] = dec_MB / total_com_speed[n]
            total_dcom_speed[n] = dec_MB / total_dcom_speed[n]

        # get stddev for speeds
        com_speed_std = np.std(com_speed)
        dcom_speed_std = np.std(dcom_speed)

        enc_speed_std = np.std(enc_speed)
        denc_speed_std = np.std(denc_speed)

        total_com_speed_std = np.std(total_com_speed)
        total_dcom_speed_std = np.std(total_dcom_speed)

        # get means for speeds
        enc_speed = np.mean(enc_speed)
        denc_speed = np.mean(denc_speed)

        com_speed = np.mean(com_speed)
        dcom_speed = np.mean(dcom_speed)

        total_com_speed = np.mean(total_com_speed)
        total_dcom_speed = np.mean(total_dcom_speed)

        if verbose:
            print '>>>> %s + %s <<<<' % (enc.name(), com.name())
            print 'Compression Method:', com.name()
            print 'Encoding Method:', enc.name()
            print 'Input Size:', dec_MB, 'MB'
            print 'Output Size:', com_MB, 'MB'
            print 'Ratio:', ratio
            print 'Total Compression Speed [MB/s]:', total_com_speed
            print 'Total Decompression Speed [MB/s]:', total_dcom_speed
            print 'Compression (Only) Speed [MB/s]:', com_speed
            print 'Decompression (Only) Speed [MB/s]:', dcom_speed
            print 'Encoding Speed [MB/s]:', enc_speed
            print 'Decoding Speed [MB/s]:', denc_speed
            print ''

        return {
            'encoding': enc.name(),
            'compression': com.name(),
            'orig_bytes': dec_MB,
            'comp_bytes': com_MB,
            'ratio': ratio,
            'comp_speed': com_speed,
            'comp_speed_stddev': com_speed_std,
            'decomp_speed': dcom_speed,
            'decomp_speed_stddev': dcom_speed_std,
            'enc_speed': enc_speed,
            'enc_speed_stddev': enc_speed_std,
            'denc_speed': denc_speed,
            'denc_speed_stddev': denc_speed_std,
            'total_comp_speed': total_com_speed,
            'total_comp_speed_stddev': total_com_speed_std,
            'total_decomp_speed': total_dcom_speed,
            'total_decomp_speed_stddev': total_dcom_speed_std
        }

    @staticmethod
    def run_variable_experiment(com, enc, data, steps, N=100, verbose=True, *args, **kwargs):
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

        # run N iterations
        for n in range(N):
            start_time = time.time()

            # run encoding and compression
            encoded_data, t1 = Util.encode(enc, data, steps)
            compressed_data, t2 = Util.compress(com, encoded_data, *args)

            # decompress data
            if (com.name() == 'LZF'):
                # LZF requires the original output size
                decompressed_data, t3 = Util.decompress(com, compressed_data, 8 * data.size)
            else:
                decompressed_data, t3 = Util.decompress(com, compressed_data)

            # make sure the data is returned as an array
            if not isinstance(decompressed_data, (np.ndarray, np.generic)) and isinstance(encoded_data, (np.ndarray, np.generic)):
                # convert back to numpy array
                decompressed_data = np.fromstring(decompressed_data, dtype=encoded_data.dtype)

            # decode the data
            decoded_data, t4 = Util.decode(enc, decompressed_data, steps)

            # update the speed lists
            enc_speed.append(t1)
            denc_speed.append(t4)

            com_speed.append(t2)
            dcom_speed.append(t3)

            total_com_speed.append(t1 + t2)
            total_dcom_speed.append(t3 + t4)

            # guarantee lossless behavior
            assert np.array_equal(np.ndarray.flatten(data), np.ndarray.flatten(decoded_data))

            print 'Ran iteration ' + str(n + 1) + ' of ' + str(N) + ' on ' + enc.name() + ' + ' + com.name() + ' in %0.2f seconds' % (time.time() - start_time)
            sys.stdout.flush()

        com_MB = Util.get_size(compressed_data) / float(1000**2)
        dec_MB = Util.get_size(data) / float(1000**2)

        # Higher is better
        ratio = dec_MB / com_MB

        # turn the speeds in MB / s
        for n in range(N):
            if enc_speed[n] == 0:
                enc_speed[n] = 0.01
            if denc_speed[n] == 0:
                denc_speed[n] = 0.01
            if com_speed[n] == 0:
                com_speed[n] = 0.01
            if dcom_speed[n] == 0:
                dcom_speed[n] = 0.01
            if total_com_speed[n] == 0:
                total_com_speed[n] = 0.01
            if total_dcom_speed[n] == 0:
                total_dcom_speed[n] = 0.01

            enc_speed[n] = dec_MB / enc_speed[n]
            denc_speed[n] = dec_MB / denc_speed[n]

            com_speed[n] = dec_MB / com_speed[n]
            dcom_speed[n] = dec_MB / dcom_speed[n]

            total_com_speed[n] = dec_MB / total_com_speed[n]
            total_dcom_speed[n] = dec_MB / total_dcom_speed[n]

        # get stddev for speeds
        com_speed_std = np.std(com_speed)
        dcom_speed_std = np.std(dcom_speed)

        enc_speed_std = np.std(enc_speed)
        denc_speed_std = np.std(denc_speed)

        total_com_speed_std = np.std(total_com_speed)
        total_dcom_speed_std = np.std(total_dcom_speed)

        # get means for speeds
        enc_speed = np.mean(enc_speed)
        denc_speed = np.mean(denc_speed)

        com_speed = np.mean(com_speed)
        dcom_speed = np.mean(dcom_speed)

        total_com_speed = np.mean(total_com_speed)
        total_dcom_speed = np.mean(total_dcom_speed)

        if verbose:
            print '>>>> %s + %s <<<<' % (enc.name(), com.name())
            print 'Compression Method:', com.name()
            print 'Encoding Method:', enc.name()
            print 'Input Size:', dec_MB, 'MB'
            print 'Output Size:', com_MB, 'MB'
            print 'Ratio:', ratio
            print 'Total Compression Speed [MB/s]:', total_com_speed
            print 'Total Decompression Speed [MB/s]:', total_dcom_speed
            print 'Compression (Only) Speed [MB/s]:', com_speed
            print 'Decompression (Only) Speed [MB/s]:', dcom_speed
            print 'Encoding Speed [MB/s]:', enc_speed
            print 'Decoding Speed [MB/s]:', denc_speed
            print ''

        return {
            'encoding': enc.name(),
            'compression': com.name(),
            'orig_bytes': dec_MB,
            'comp_bytes': com_MB,
            'ratio': ratio,
            'comp_speed': com_speed,
            'comp_speed_stddev': com_speed_std,
            'decomp_speed': dcom_speed,
            'decomp_speed_stddev': dcom_speed_std,
            'enc_speed': enc_speed,
            'enc_speed_stddev': enc_speed_std,
            'denc_speed': denc_speed,
            'denc_speed_stddev': denc_speed_std,
            'total_comp_speed': total_com_speed,
            'total_comp_speed_stddev': total_com_speed_std,
            'total_decomp_speed': total_dcom_speed,
            'total_decomp_speed_stddev': total_dcom_speed_std
        }

    @staticmethod
    def plot_all(
        results,
        what,
        x_range=None,
        y_range=None,
        name=None,
        leg=True,
        leg_loc='right',
        no_bw=True,
        input_bytes=-1,
        output='',
        emphasis=-1,
        bar_label=-1,
        no_leg_bars=False,
        log=False,
        digital=True,
        no_errorbars=False,
        title=None
    ):

        if name is None:
            raise ValueError(
                'Holy Moly you haven\'t specified a `name`! Shame on you.'
            )

        if what not in results:
            raise ValueError('Wrong `what` parameter. Not found in `results`.')

        labels = []

        for i, method in enumerate(results['methods']):
            labels.append(method.split()[0])

        labels = sorted(list(set(labels)), key=lambda s: s.lower())

        font_base = FontProperties()

        font_bold = font_base.copy()
        font_bold.set_weight('bold')

        none = [0] * len(labels)
        none_std = [0] * len(labels)
        rle = [0] * len(labels)
        rle_std = [0] * len(labels)
        neuroglancer = [0] * len(labels)
        neuroglancer_std = [0] * len(labels)
        compresso = [0] * len(labels)
        compresso_std = [0] * len(labels)

        for i, label in enumerate(labels):
            none_index = label + ' None'

            if none_index in results['methods']:
                none_index = results['methods'].index(none_index)
            else:
                none_index = -1

            rle_index = label + ' RLE'

            if rle_index in results['methods']:
                rle_index = results['methods'].index(rle_index)
            else:
                rle_index = -1

            neuroglancer_index = label + ' Neuroglancer'

            if neuroglancer_index in results['methods']:
                neuroglancer_index = results['methods'].index(
                    neuroglancer_index
                )
            else:
                neuroglancer_index = -1

            compresso_index = label + ' Compresso'

            if compresso_index in results['methods']:
                compresso_index = results['methods'].index(compresso_index)
            else:
                compresso_index = -1

            if none_index != -1:
                none[i] = results[what][none_index]
                if what + '_std' in results:
                    none_std[i] = results[what + '_std'][none_index]
                else:
                    none_std[i] = 0

            if rle_index != -1:
                rle[i] = results[what][rle_index]
                if what + '_std' in results:
                    rle_std[i] = results[what + '_std'][rle_index]
                else:
                    rle_std[i] = 0

            if neuroglancer_index != -1:
                neuroglancer[i] = results[what][neuroglancer_index]
                if what + '_std' in results:
                    neuroglancer_std[i] = results[what + '_std'][neuroglancer_index]
                else:
                    neuroglancer_std[i] = 0

            if compresso_index != -1:
                compresso[i] = results[what][compresso_index]
                if what + '_std' in results:
                    compresso_std[i] = results[what + '_std'][compresso_index]
                else:
                    compresso_std[i] = 0

        plt.figure(figsize=(10, 10))
        N = len(labels)
        ind = np.arange(N)  # the x locations for the groups
        width = 0.25       # the width of the bars

        font = {
            'family': 'sans-serif',
            'size': 13.5
        }
        plt.rc('font', **font)

        if no_bw:
            ind = ind[1:]
            none = none[1:]
            none_std = none_std[1:]
            rle = rle[1:]
            rle_std = rle_std[1:]
            neuroglancer = neuroglancer[1:]
            neuroglancer_std = neuroglancer_std[1:]
            compresso = compresso[1:]
            compresso_std = compresso_std[1:]
            labels = labels[1:]

        lab_none = 'No first stage encoding'
        # lab_rle = 'RLE'
        lab_neuroglancer = 'Neuroglancer'
        lab_compresso = 'Compresso'

        if what == 'bytes':
            lab_none = None
            # lab_rle = None
            lab_neuroglancer = None
            lab_compresso = None

        if what.endswith('speed') and log:
            none = np.log10([max(x, 1) for x in none])
            neuroglancer = np.log10([max(x, 1) for x in neuroglancer])
            compresso = np.log10([max(x, 1) for x in compresso])

        capthick = 0 if no_errorbars else 2

        fig, ax = plt.subplots()
        ne = ax.bar(
            ind,
            none,
            width,
            color='#bbbbbb',
            label=lab_none,
            edgecolor='#ffffff',
            linewidth=0,
            yerr=none_std,
            ecolor=(0, 0, 0, 0.2) if digital else (0, 0, 0, 1),
            error_kw=dict(lw=2, capsize=2, capthick=capthick)
        )

        ng = ax.bar(
            ind + width,
            neuroglancer,
            width,
            color='#999999' if digital else '#808080',
            label=lab_neuroglancer,
            edgecolor='#ffffff',
            linewidth=0,
            yerr=neuroglancer_std,
            ecolor=(0, 0, 0, 0.2) if digital else (0, 0, 0, 1),
            error_kw=dict(lw=2, capsize=2, capthick=capthick)
        )
        cp = ax.bar(
            ind + width * 2,
            compresso,
            width,
            color='#dc133b',
            label=lab_compresso,
            edgecolor='#ffffff',
            linewidth=0,
            yerr=compresso_std,
            ecolor=(0, 0, 0, 0.2) if digital else (0, 0, 0, 1),
            error_kw=dict(lw=2, capsize=2, capthick=capthick)
        )

        if what == 'bytes_size':
            ax.axhline(
                y=input_bytes,
                color='gray',
                label='Input',
                linewidth=2,
                linestyle='--'
            )

        ax.tick_params(
            axis='y',
            color='#cccccc' if digital else '#888888',
            labelcolor='#999999' if digital else '#333333'
        )

        ax.tick_params(
            axis='x',
            color='#cccccc'
        )

        xticks_colors = ['#666666' if digital else '#333333'] * len(compresso)
        wurst_font = font_base
        wurst_font.set_size(15)
        xticks_fonts = [wurst_font] * len(compresso)

        if emphasis >= 0:
            xticks_colors[emphasis] = '#333333' if digital else '#000000'
            xticks_fonts[emphasis] = font_bold
            xticks_fonts[emphasis].set_size(16)

        for xtick, color, fp in zip(
            ax.get_xticklabels(), xticks_colors, xticks_fonts
        ):
            xtick.set_color(color)
            xtick.set_font_properties(fp)

        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc' if digital else '#888888')

        ylabel = 'Compression Ratio\n(Original / Compressed)'

        if what.endswith('comp_speed'):
            ylabel = 'Compression Speed\n(MB/s)'

        if what.endswith('dcom_speed'):
            ylabel = 'Decompression Speed\n(MB/s)'

        if what.endswith('bytes_size'):
            ylabel = 'Size\n(MB)'
            ax.set_yscale('log', nonposy='clip')

        plt.ylabel(
            ylabel,
            color='#333333',
            labelpad=10,
            fontsize=16
        )

        if leg:
            leg = plt.legend(
                loc='upper %s' % leg_loc,
                prop={
                    'size': 15
                }
            )
            if what != 'bytes':
                if no_leg_bars:
                    if type(leg.legendHandles[0]) == Rectangle:
                        leg.legendHandles[0].set_width(1)

                    if type(leg.legendHandles[1]) == Rectangle:
                        leg.legendHandles[1].set_width(1)

                    if type(leg.legendHandles[2]) == Rectangle:
                        leg.legendHandles[2].set_width(1)

                # leg.legendHandles[0].set_width(2)
                leg.legendHandles[0].set_color('#bbbbbb')
                # leg.legendHandles[1].set_color('#8c8c8c')
                leg.legendHandles[1].set_color(
                    '#999999' if digital else '#777777'
                )
                leg.legendHandles[2].set_color('#dc133b')
            frm = leg.get_frame()
            frm.set_edgecolor('#ffffff')
            frm.set_facecolor('#ffffff')
            leg_texts = leg.get_texts()
            leg_texts[0].set_color('#999999' if digital else '#808080')
            leg_texts[1].set_color('#666666')
            leg_texts[2].set_color('#dc133b')

        ax.set_xticks(ind + width * 1.5)
        ax.set_xticklabels(labels, rotation='vertical')

        if y_range is not None:
            plt.ylim(0, y_range)

        if no_bw:
            plt.xlim(0.5, len(none) + 1)
        else:
            plt.xlim(-0.5, len(none) + 0.5)

        plt.tick_params(
            axis='x',
            which='both',
            bottom='off',
            top='off'
        )

        if bar_label >= 0:
            # Plot label above bar to clarify relatiobship
            height = ne[bar_label].get_height()
            plt.text(
                ne[bar_label].get_x() + ne[bar_label].get_width() / 2.0,
                height,
                '1',
                ha='center',
                va='bottom',
                color='#bbbbbb',
                fontproperties=font_bold
            )

            height = ng[bar_label].get_height()
            plt.text(
                ng[bar_label].get_x() + ng[bar_label].get_width() / 2.0,
                height,
                '2',
                ha='center',
                va='bottom',
                color='#999999',
                fontproperties=font_bold
            )

            height = cp[bar_label].get_height()
            plt.text(
                cp[bar_label].get_x() + cp[bar_label].get_width() / 2.0,
                height,
                '3',
                ha='center',
                va='bottom',
                color='#dc133b',
                fontproperties=font_bold
            )

        if title is not None:
            plt.title(title, fontsize=18)

        ttl = ax.title
        ttl.set_position([.5, 1.05])
        ttl.set_font_properties

        plt.savefig(
            os.path.join(output, '%s_compression_%s.pdf' % (name, what)),
            bbox_inches='tight'
        )
