"""
hashing.py

SHA-256 Hashing
"""

import hashlib


def generate_hash(data):
    """
    Generate SHA-256 hash.
    """

    return hashlib.sha256(
        data.encode()
    ).hexdigest()