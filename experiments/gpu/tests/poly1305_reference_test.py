#!/usr/bin/env python3
"""Differential tests against the direct RFC 8439 integer definition.

No external Python packages, CUDA or GPU are required. The reference uses
arbitrary-precision arithmetic, not the implementation's radix-2^26 limbs.
"""

import ctypes
import os
from pathlib import Path
import random
import shlex
import subprocess
import tempfile

try:
    from cryptography.hazmat.primitives.poly1305 import Poly1305 as IndependentMAC
except ImportError:
    IndependentMAC = None


def reference(key, message):
    r = int.from_bytes(key[:16], "little") & 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF
    s = int.from_bytes(key[16:], "little")
    accumulator = 0
    for offset in range(0, len(message), 16):
        block = message[offset:offset + 16]
        number = int.from_bytes(block + b"\x01", "little")
        accumulator = ((accumulator + number) * r) % ((1 << 130) - 5)
    return ((accumulator + s) % (1 << 128)).to_bytes(16, "little")


def main():
    here = Path(__file__).resolve().parent
    rng = random.Random(8439)
    # Every length through 257 plus longer random messages and large boundaries.
    lengths = list(range(258)) + [rng.randrange(258, 8193) for _ in range(250)]
    lengths += [16383, 16384, 16385, 65535, 65536, 65537]
    checks = 0
    with tempfile.TemporaryDirectory(prefix="vault-poly1305-") as directory:
        library = str(Path(directory) / "poly1305.so")
        subprocess.run(
            shlex.split(os.environ.get("CXX", "g++")) +
            ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
             "-shared", "-fPIC", str(here / "poly1305_bridge.cpp"),
             "-o", library],
            check=True,
        )
        lib = ctypes.CDLL(library)
        mac = lib.poly1305_test_tag
        mac.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t,
                        ctypes.c_size_t, ctypes.c_void_p]
        mac.restype = None
        for length in lengths:
            key = rng.randbytes(32)
            message = rng.randbytes(length)
            expected = reference(key, message)
            if IndependentMAC is not None:
                if IndependentMAC.generate_tag(key, message) != expected:
                    raise AssertionError(f"Independent reference mismatch: {length}")
            for stride in sorted({1, 7, 15, 16, 17, 31, 64, max(1, length)}):
                tag = ctypes.create_string_buffer(16)
                mac(key, message, length, stride, tag)
                if tag.raw != expected:
                    raise AssertionError(
                        f"length={length}, stride={stride}: "
                        f"{tag.raw.hex()} != {expected.hex()}"
                    )
                checks += 1
    print(f"PASS: {checks} differential checks across {len(lengths)} messages")
    if IndependentMAC is not None:
        print(f"PASS: {len(lengths)} reference tags cross-checked with cryptography")
    else:
        print("SKIP: optional cryptography cross-check (package not installed)")


if __name__ == "__main__":
    main()
