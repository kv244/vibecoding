import os
import sys
import secrets

def xor_cipher(data, key):
    """Applies a bitwise XOR between data and a repeating key."""
    return bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])

def process_file(file_path, mode='encrypt'):
    if mode == 'encrypt':
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found.")
            return

        # Read source data
        with open(file_path, 'rb') as f:
            data = f.read()

        # Generate a high-entropy random key (32 bytes)
        key = secrets.token_bytes(32)
        
        # Encrypt
        encrypted_data = xor_cipher(data, key)

        # Save outputs
        enc_filename = file_path + ".enc"
        key_filename = file_path + ".ky"

        with open(enc_filename, 'wb') as f:
            f.write(encrypted_data)
        with open(key_filename, 'wb') as f:
            f.write(key)

        print(f"File encrypted to: {enc_filename}")
        print(f"Key saved to: {key_filename}")

    elif mode == 'decrypt':
        # Expects base name, e.g., 'source.jpg' to find '.enc' and '.ky'
        enc_filename = file_path + ".enc"
        key_filename = file_path + ".ky"
        out_filename = file_path  # Restore to original name

        if not os.path.exists(enc_filename) or not os.path.exists(key_filename):
            print(f"Error: Missing {enc_filename} or {key_filename}")
            return

        # Read encrypted data and key
        with open(enc_filename, 'rb') as f:
            enc_data = f.read()
        with open(key_filename, 'rb') as f:
            key = f.read()

        # Decrypt (XOR is symmetric)
        decrypted_data = xor_cipher(enc_data, key)

        with open(out_filename, 'wb') as f:
            f.write(decrypted_data)

        print(f"File decrypted to: {out_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python vibecrypt_file.py [encrypt|decrypt] [filename]")
    else:
        cmd = sys.argv[1].lower()
        target = sys.argv[2]
        process_file(target, cmd)
