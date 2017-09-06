import itertools
import numpy as np
import os
import png
import tempfile


class _png(object):

    @staticmethod
    def name():
        '''No Encoding
        '''

        return 'PNG'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''PNG compression
        '''
        TMPFOLDER = tempfile.mkdtemp()

        compressed_data = ''

        sizes = []


        for iz in range(0, data.shape[0]):
            img = data[iz, :, :]

            colorized = np.zeros(
                (3, img.shape[0], img.shape[1]), dtype=np.uint16
            )

            # for every value split into three 16 bit samples
            colorized[0, :, :] = img % (2**16)
            img = img >> 16
            colorized[1, :, :] = img % (2**16)
            img = img >> 16
            colorized[2, :, :] = img % (2**16)

            colorized = colorized.swapaxes(0, 1).swapaxes(1, 2)

            row_count, column_count, plane_count = colorized.shape

            pngfile = open(TMPFOLDER+'/tmp_' + str(iz) + '.png', 'wb')
            pngWriter = png.Writer(
                column_count,
                row_count,
                greyscale=False,
                alpha=False,
                bitdepth=16
            )
            pngWriter.write(
                pngfile,
                np.reshape(colorized, (-1, column_count * plane_count))
            )
            pngfile.close()

            with open(TMPFOLDER+'/tmp_' + str(iz) + '.png', 'rb') as fd:
                c_data = fd.read()
                compressed_data += c_data
                sizes.append(len(c_data))


        frames = np.zeros((len(sizes)), dtype=np.uint64)

        for i,s in enumerate(sizes):

            frames[i] = s

        #
        #
        # no of frames
        output = np.uint64(len(sizes)).tobytes()

        # frame sizes
        output += frames.tobytes()

        output += compressed_data

        # print sizes

        return output

    @staticmethod
    def decompress(data, *args, **kwargs):
        '''PNG decompression
        '''


        TMPFOLDER = tempfile.mkdtemp()

        # grab no of frames
        no_frames = np.fromstring(data[0:8], dtype=np.uint64)
        # print no_frames, len(data), data[8:8*no_frames]
        no_frames = no_frames[0]

        frame_sizes = data[8:8+int(8*no_frames)]

        # print no_frames, frame_sizes

        # grab frame sizes
        sizes = np.fromstring(frame_sizes, dtype=np.uint64)

        # store each frame to TMP FOLDER
        data_start_byte = 8 + 8*no_frames

        current_byte_pointer = data_start_byte
        for i in range(sizes.shape[0]):

            # print 'writing',i,current_byte_pointer,current_byte_pointer+sizes[i]

            current_bytes = data[int(current_byte_pointer):int(current_byte_pointer+sizes[i])]
            with open(TMPFOLDER+'/tmp_'+str(i)+'.jp2', 'wb') as f:
                f.write(current_bytes)

            

            current_byte_pointer = current_byte_pointer+sizes[i]



        nfiles = len(os.listdir(TMPFOLDER))
        for ie, filename in enumerate(os.listdir(TMPFOLDER)):
            input_filename = TMPFOLDER + '/' + filename

            index = int(filename.split('_')[1].split('.')[0])

            pngReader = png.Reader(filename=input_filename)
            row_count, column_count, png_data, meta = pngReader.asDirect()
            plane_count = meta['planes']

            # make sure rgb files
            assert plane_count == 3

            img = np.vstack(itertools.imap(np.uint16, png_data))
            colorized = np.reshape(img, (row_count, column_count, plane_count))

            colorized = colorized.swapaxes(1, 2).swapaxes(0, 1)

            if (ie == 0):
                decompressed_data = np.zeros(
                    (nfiles, colorized.shape[1], colorized.shape[2]),
                    dtype=np.uint64
                )

            decompressed_data[index, :, :] = (
                colorized[0, :, :] +
                colorized[1, :, :] * (2 ** 16) +
                colorized[2, :, :] * (2 ** 16)
            )


        return decompressed_data
