# experiments/gpu

GPU compute experiments using CUDA and Numba.

---

## `encrypt.cu` — GPU File Encryptor

An industrial-strength file encryption tool using the GPU to accelerate the ChaCha20 keystream generation, with Argon2id key derivation and Poly1305 authentication.

### Security model

| Component | Primitive | Standard |
|---|---|---|
| Key derivation | Argon2id | RFC 9106 |
| Encryption | ChaCha20 stream cipher (GPU-parallelised) | RFC 8439 |
| Authentication | Poly1305 MAC (CPU-side) | RFC 8439 |
| AAD | Full 41-byte header authenticated | RFC 8439 §2.8 |
| Nonce | 96-bit, OS CSPRNG | — |
| Salt | 128-bit, OS CSPRNG | — |

- Passphrase read from `/dev/tty` (echo off) — never via `argv`
- Key material zeroed after use via `volatile` loop (`secure_zero`)
- Decrypted output written to `.tmp` file; renamed to final destination only after MAC verification passes — partial plaintext never persists on failure
- Constant-time MAC comparison prevents timing side-channels
- `orig_size` serialized as explicit little-endian — portable across architectures
- OS CSPRNG used directly: `getrandom()` (Linux), `BCryptGenRandom` (Windows)

### File format

```
[4 bytes magic 'VBCR'] [1 byte version=0x02] [16 bytes salt] [12 bytes nonce]
[8 bytes orig_size LE] [ciphertext chunks...]  [16 bytes Poly1305 tag]
```

The full 41-byte header is fed to Poly1305 as AAD, binding all header fields to the tag.

### Dependencies

- CUDA Toolkit >= 10.0
- C++17 (for `std::filesystem`)
- [Argon2 reference implementation](https://github.com/P-H-C/phc-winner-argon2) (`-largon2`)

### Build

**Linux (sm_75+):**
```bash
nvcc -O3 -use_fast_math -arch=sm_75 -std=c++17 \
     --ptxas-options=-v encrypt.cu -o vibecoder -largon2
```

**Windows (MSVC + vcpkg argon2):**
```bat
nvcc -allow-unsupported-compiler -O3 -use_fast_math -std=c++17 ^
  -I "C:\vcpkg\installed\x64-windows\include" ^
  -L "C:\vcpkg\installed\x64-windows\lib" ^
  encrypt.cu -o vibecoder.exe -largon2
```

### Usage

```
vibecoder encrypt  [-r] [--chunk <MB>] <file|dir>
vibecoder decrypt  [-r] [--chunk <MB>] <file.enc|dir>
vibecoder verify        [--chunk <MB>] <file.enc>
```

| Option | Description |
|---|---|
| `-r` | Recurse into directory |
| `--chunk <MB>` | Streaming window size (default: 256 MB, min: 16 MB). Auto-clamped to 40% of free VRAM. |

**Passphrase:** read interactively from `/dev/tty` (echo disabled), or set `VIBECODER_PASSPHRASE` env var for scripted/CI use.

### Examples

```bash
# Encrypt a single file
vibecoder encrypt secret.tar.gz

# Decrypt
vibecoder decrypt secret.tar.gz.enc

# Verify MAC without writing plaintext
vibecoder verify secret.tar.gz.enc

# Recursively encrypt a directory
vibecoder encrypt -r ./documents

# Use a smaller chunk on a GPU with limited VRAM
vibecoder encrypt --chunk 64 large_file.iso
```

---

## TRON 3d

Implemented in CUDA and Python.
