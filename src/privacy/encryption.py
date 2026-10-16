"""
encryption.py

Symmetric encryption using Fernet.

Note: Fernet is AES-128-CBC with HMAC-SHA256 authentication, NOT AES-256.
Describe it that way in reports/papers.
"""

from cryptography.fernet import Fernet


def generate_key():
    """
    Generate a Fernet key (AES-128-CBC + HMAC-SHA256).
    """
    return Fernet.generate_key()


def encrypt_data(data, key):
    """
    Encrypt string data.
    """

    cipher = Fernet(key)

    encrypted = cipher.encrypt(
        data.encode()
    )

    return encrypted


def decrypt_data(encrypted_data, key):
    """
    Decrypt encrypted data.
    """

    cipher = Fernet(key)

    decrypted = cipher.decrypt(
        encrypted_data
    )

    return decrypted.decode()