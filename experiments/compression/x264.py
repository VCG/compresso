import os
import numpy as np
from PIL import Image
from subprocess import Popen, PIPE
import tempfile

class x264(object):

    @staticmethod
    def name():
        '''X.264 compression
        '''

        return 'X.264'

    @staticmethod
    def compress(data):
        '''X.264 compression
        '''

        from util import Util
        outlog = tempfile.mktemp()
        outvideo = tempfile.mktemp(suffix='.mp4')

        process_output = open(outlog,'w')
        p = Popen(['ffmpeg', 
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-y', 
                   '-r', str(data.shape[0]),
                   '-video_size', str(data.shape[1])+'x'+str(data.shape[2]),
                   '-pixel_format', 'yuv444p',
                   '-i', '-',
                   '-c:v', 'libx264',
                   '-pix_fmt', 'yuv444p',
                   '-profile:v', 'high444',
                   '-crf', '0',
                   '-preset:v', 'slow',
                   outvideo], stdin=PIPE, stdout=process_output, stderr=process_output)
        
        for z in range(data.shape[0]):
          Util.convert_to_rgb(data[z]).tofile(p.stdin)

        process_output.close()
        p.stdin.close()
        p.wait()


        outdata = None

        with open(outvideo, 'rb') as f:
            outdata = f.read()

        # we also need to pass the X,Y,Z dimensions
        dims = np.zeros((3), dtype=np.uint64)
        dims[0] = data.shape[0]# Z
        dims[1] = data.shape[1]# Y
        dims[2] = data.shape[2]# X

        return dims.tobytes() + outdata


    @staticmethod
    def decompress(data):
        '''X.264 decompression
        '''
        from util import Util
        errlog = tempfile.mktemp()
        outvideo = tempfile.mktemp(suffix='.mp4')

        dims = data[0:3*8] # 3 * 64bit
        dims = np.fromstring(dims, dtype=np.uint64)

        videodata = data[3*8:]

        with open(outvideo, 'wb') as f:
            f.write(videodata)

        process_output = open(errlog,'w')

        p = Popen(['ffmpeg', 
                   '-i', outvideo, 
                   '-vcodec', 'rawvideo',        
                   '-f', 'image2pipe',
                   '-video_size', str(dims[1])+'x'+str(dims[2]),
                   '-pix_fmt', 'yuv444p',
                   '-'
                  ], stdout=PIPE, stderr=process_output)

        framesize = dims[1]*dims[2]*3

        frames = p.stdout.read(int(framesize*dims[0]))

        output_data = np.fromstring(frames, dtype=np.uint8)
        output_data_rgb = output_data.reshape((dims[0], dims[1], dims[2], 3))
        output_data_64 = np.zeros((dims[0], dims[1], dims[2]), dtype=np.uint64)
        for z in range(output_data_64.shape[0]):

          slice64 = Util.convert_from_rgb(output_data_rgb[z])

          output_data_64[z] = slice64

        p.stdout.close()
        p.wait()
        process_output.close()     

        return output_data_64
