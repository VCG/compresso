namespace neuroglancer {
    unsigned long *Compress(unsigned long *data, int zres, int yres, int xres, int bz, int by, int bx, int origz, int origy, int origx);

    unsigned long *Decompress(unsigned long *compressed_data, int bz, int by, int bx);
}