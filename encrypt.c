#include <stdint.h>
#include <stddef.h>
#include <omp.h>

// We use 'extern "C"' logic implicitly by compiling with gcc 
// to ensure symbol names are clean for Python.

// to compile:
//  gcc -O3 -fopenmp -shared -fPIC encrypt.c -o libbeagle_crypt.so

void beagle_crypt(uint64_t *data, size_t len_bytes, uint64_t key) {
    size_t total_64bit_blocks = len_bytes / 8;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total_64bit_blocks; i++) {
        uint64_t *current_ptr = &data[i];
        asm volatile (
            "ld   t0, 0(%0)\n\t"
            "xor  t0, t0, %1\n\t"
            "sd   t0, 0(%0)\n\t"
            : 
            : "r" (current_ptr), "r" (key)
            : "t0", "memory"
        );
    }
}
