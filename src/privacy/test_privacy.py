from src.privacy.encryption import (
    generate_key,
    encrypt_data,
    decrypt_data
)

from src.privacy.hashing import (
    generate_hash
)

# -----------------------------

sample_data = """
Prediction : REAL
Confidence : 93.22%
Verified   : True
Trust Score: 92.86
"""

# -----------------------------

key = generate_key()

encrypted = encrypt_data(
    sample_data,
    key
)

decrypted = decrypt_data(
    encrypted,
    key
)

hash_value = generate_hash(
    sample_data
)

print("\n========== PRIVACY TEST ==========\n")

print("Original Data\n")

print(sample_data)

print("\nEncryption Key\n")

print(key.decode())

print("\nEncrypted Data\n")

print(encrypted.decode())

print("\nDecrypted Data\n")

print(decrypted)

print("\nSHA-256 Hash\n")

print(hash_value)