__kernel void xor_cipher(__global unsigned char* data, __global const unsigned char* key, const unsigned int key_size) {
    int id = get_global_id(0);
    // Each thread processes 16 bytes for high throughput
    int offset = id * 16;
    
    // Use vectorized loads/stores if possible
    uchar16 d = vload16(0, data + offset);
    
    // Since key is small (32 bytes), we cycle it
    // Note: This assumes 16-byte alignment of the data buffer
    uchar16 k;
    if ((offset % key_size) == 0) {
        k = vload16(0, key);
    } else {
        k = vload16(0, key + 16);
    }
    
    d ^= k;
    vstore16(d, 0, data + offset);
}
