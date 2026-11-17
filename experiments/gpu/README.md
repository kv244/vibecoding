# experiments/gpu

GPU compute experiments using CUDA and Numba.

---

## `encrypt.cu` — GPU File Encryptor

An experimental file encryption tool using the GPU to accelerate the ChaCha20 keystream generation, with Argon2id key derivation and Poly1305 authentication.

> **Experimental, not security-audited.** The corrected Poly1305 primitive is
> tested against RFC vectors, but the file-encryption application's complete
> AEAD framing and file-safety issues remain unresolved. Passing the primitive
> tests does not establish RFC 8439 AEAD compliance or production readiness.
> Tags from the previous, incorrect implementation are not compatible with
> this fix. Preserve existing encrypted files and their original tool; this
> change does not provide a legacy-file migration path.

### CPU-only Poly1305 regression tests

These tests use the same `poly1305.hpp` included by `encrypt.cu`; they do not
need CUDA, an NVIDIA GPU or Argon2. Run from the repository root:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pedantic \
    experiments/gpu/tests/poly1305_test.cpp -o /tmp/poly1305-test
/tmp/poly1305-test
python3 experiments/gpu/tests/poly1305_reference_test.py
```

The C++ suite checks [RFC 8439](https://www.rfc-editor.org/rfc/rfc8439)
section 2.5.2 and Appendix A.3 vectors 1 and 5–11, all two-update splits,
byte-at-a-time input, empty input and explicit AAD zero padding. The Python
suite (Python 3.9+) compares random and boundary-length inputs with a separate
arbitrary-precision implementation of the RFC definition.

For memory and undefined-behaviour checks:

```bash
g++ -std=c++17 -O1 -g -fsanitize=address,undefined \
    -fno-omit-frame-pointer experiments/gpu/tests/poly1305_test.cpp \
    -o /tmp/poly1305-sanitized
/tmp/poly1305-sanitized
```

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

## TRON 3d (main.cu, mainpy)

Implemented in CUDA and Python.
