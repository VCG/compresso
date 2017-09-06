namespace compresso {
    unsigned long *Compress(unsigned long *data, int zres, int yres, int xres, int zstep, int ystep, int xstep);

    unsigned long *Decompress(unsigned long *compressed_data);
}
