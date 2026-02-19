#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

#define KEY_SIZE 32

/**
 * RISC-V Optimized XOR Cipher
 * Uses inline assembly to avoid the modulo operator and division.
 * Manually manages pointers to reset the key-stream at the end of the key buffer.
 */
void xor_buffer_asm(unsigned char *data, size_t data_len, unsigned char *key, size_t key_len) {
    if (data_len == 0 || key_len == 0) return;

    unsigned char *data_end = data + data_len;
    unsigned char *key_ptr = key;
    unsigned char *key_end = key + key_len;

    __asm__ __volatile__ (
        "1: \n\t"
        "beq %[curr_data], %[data_end], 3f \n\t" // If data_ptr == data_end, exit
        "lbu t0, 0(%[curr_data]) \n\t"           // t0 = *data
        "lbu t1, 0(%[curr_key]) \n\t"            // t1 = *key
        "xor t0, t0, t1 \n\t"                    // t0 = t0 ^ t1
        "sb t0, 0(%[curr_data]) \n\t"            // *data = t0
        
        "addi %[curr_data], %[curr_data], 1 \n\t" // data++
        "addi %[curr_key], %[curr_key], 1 \n\t"   // key++
        
        "blt %[curr_key], %[key_end], 2f \n\t"    // If key_ptr < key_end, skip reset
        "mv %[curr_key], %[key_start] \n\t"       // Else, key_ptr = key_start (reset key)
        
        "2: \n\t"
        "j 1b \n\t"                               // Loop
        "3: \n\t"
        : [curr_data] "+r" (data), [curr_key] "+r" (key_ptr)
        : [data_end] "r" (data_end), [key_end] "r" (key_end), [key_start] "r" (key)
        : "t0", "t1", "memory"
    );
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s [encrypt|decrypt] [filename]\n", argv[0]);
        return 1;
    }

    char *mode = argv[1];
    char *filename = argv[2];

    if (strcmp(mode, "encrypt") == 0) {
        FILE *f = fopen(filename, "rb");
        if (!f) { perror("Source file error"); return 1; }

        fseek(f, 0, SEEK_END);
        size_t fsize = ftell(f);
        rewind(f);

        unsigned char *buffer = malloc(fsize);
        if (!buffer) { printf("Memory allocation failed\n"); fclose(f); return 1; }
        fread(buffer, 1, fsize, f);
        fclose(f);

        // Generate 32-byte key
        unsigned char key[KEY_SIZE];
        srand(time(NULL));
        for (int i = 0; i < KEY_SIZE; i++) key[i] = rand() % 256;

        // Optimized ASM Encryption
        xor_buffer_asm(buffer, fsize, key, KEY_SIZE);

        // Save source.jpg.enc
        char enc_name[256];
        snprintf(enc_name, sizeof(enc_name), "%s.enc", filename);
        FILE *fout = fopen(enc_name, "wb");
        fwrite(buffer, 1, fsize, fout);
        fclose(fout);

        // Save source.jpg.ky
        char key_name[256];
        snprintf(key_name, sizeof(key_name), "%s.ky", filename);
        FILE *fkey = fopen(key_name, "wb");
        fwrite(key, 1, KEY_SIZE, fkey);
        fclose(fkey);

        printf("Successfully encrypted %s\nFiles created: %s, %s\n", filename, enc_name, key_name);
        free(buffer);

    } else if (strcmp(mode, "decrypt") == 0) {
        char enc_name[256], key_name[256];
        snprintf(enc_name, sizeof(enc_name), "%s.enc", filename);
        snprintf(key_name, sizeof(key_name), "%s.ky", filename);

        // Load Key
        unsigned char key[KEY_SIZE];
        FILE *fk = fopen(key_name, "rb");
        if (!fk) { printf("Key file %s not found\n", key_name); return 1; }
        fread(key, 1, KEY_SIZE, fk);
        fclose(fk);

        // Load Encrypted Data
        FILE *fe = fopen(enc_name, "rb");
        if (!fe) { printf("Encrypted file %s not found\n", enc_name); return 1; }
        fseek(fe, 0, SEEK_END);
        size_t fsize = ftell(fe);
        rewind(fe);

        unsigned char *buffer = malloc(fsize);
        if (!buffer) { printf("Memory allocation failed\n"); fclose(fe); return 1; }
        fread(buffer, 1, fsize, fe);
        fclose(fe);

        // Optimized ASM Decryption (Symmetric)
        xor_buffer_asm(buffer, fsize, key, KEY_SIZE);

        // Restore original filename
        FILE *fout = fopen(filename, "wb");
        fwrite(buffer, 1, fsize, fout);
        fclose(fout);

        printf("Successfully decrypted %s from %s\n", filename, enc_name);
        free(buffer);
    } else {
        printf("Invalid mode. Use 'encrypt' or 'decrypt'.\n");
    }

    return 0;
}
