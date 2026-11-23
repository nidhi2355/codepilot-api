import os
from cryptography.fernet import Fernet


def _get_fernet() -> Fernet:
    key = os.getenv("TOKEN_ENCRYPTION_KEY")
    if not key:
        raise RuntimeError("TOKEN_ENCRYPTION_KEY environment variable is not set")
    return Fernet(key.encode("utf-8"))


def encrypt_token(plain_token: str) -> str:
    """Encrypts a GitHub OAuth access token for safe storage in PostgreSQL."""
    return _get_fernet().encrypt(plain_token.encode("utf-8")).decode("utf-8")


def decrypt_token(encrypted_token: str) -> str:
    """Decrypts a stored GitHub OAuth access token in memory."""
    return _get_fernet().decrypt(encrypted_token.encode("utf-8")).decode("utf-8")