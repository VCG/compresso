import glymur
import os
import numpy as np
import tempfile


class jpeg(object):

    @staticmethod
    def name():
        '''No Encoding
        '''

        return 'JPEG2000'

    @staticmethod
    def compress(data, *args, **kwargs):
        '''JPEG2000 compression
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
            img = img >> 32
            colorized[2, :, :] = img % (2**16)

            #print colorized.shape

            glymur.Jp2k(TMPFOLDER+'/tmp_' + str(iz) + '.jp2', colorized)
            #glymur.Jp2k('JPEG_TMP/tmp_' + str(iz) + '.jp2', img.astype(np.uint16))
            with open(TMPFOLDER+'/tmp_' + str(iz) + '.jp2', 'rb') as fd:
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
        '''JPEG2000 decompression
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
            colorized = glymur.Jp2k(input_filename)

            index = int(filename.split('_')[1].split('.')[0])

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
