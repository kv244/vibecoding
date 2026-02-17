import ctypes
import os

# 1. Load the shared library
lib_path = os.path.abspath("./libbeagle_crypt.so")
crypt_lib = ctypes.CDLL(lib_path)

# 2. Define the argument types for the C function
# void beagle_crypt(uint64_t *data, size_t len_bytes, uint64_t key)
crypt_lib.beagle_crypt.argtypes = [
    ctypes.POINTER(ctypes.c_uint64), 
    ctypes.c_size_t, 
    ctypes.c_uint64
]

def vibe_crypt(message, alpha_key):
    # Ensure key is 8 bytes
    key_bytes = alpha_key.encode('utf-8')[:8].ljust(8, b'\0')
    key_uint64 = int.from_bytes(key_bytes, byteorder='little')

    # Convert message to bytes and pad to 8-byte boundary
    msg_bytes = message.encode('utf-8')
    padded_len = ((len(msg_bytes) + 7) // 8) * 8
    
    # Create an aligned buffer
    buffer = (ctypes.c_uint64 * (padded_len // 8))()
    ctypes.memmove(buffer, msg_bytes, len(msg_bytes))

    # Call the C/ASM routine (Pass 1: Encrypt)
    crypt_lib.beagle_crypt(buffer, padded_len, key_uint64)
    
    # Get the encrypted bytes
    encrypted_data = bytes(buffer)
    
    # Call again (Pass 2: Decrypt)
    crypt_lib.beagle_crypt(buffer, padded_len, key_uint64)
    decrypted_msg = bytes(buffer).decode('utf-8').strip('\0')

    return encrypted_data, decrypted_msg

# --- Example Usage ---
msg = "Vibecoding on BeagleV-Fire!"
key = "VIBES_26"

enc, dec = vibe_crypt(msg, key)

print(f"Original:  {msg}")
print(f"Encrypted: {enc.hex()}")
print(f"Decrypted: {dec}")
