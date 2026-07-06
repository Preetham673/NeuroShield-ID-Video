"""
encryption.py

AES-256 Encryption using Fernet
"""

from cryptography.fernet import Fernet


def generate_key():
    """
    Generate AES encryption key.
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