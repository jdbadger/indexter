import hashlib


def compute_hash(content: str) -> str:
    """Compute SHA256 hash of the provided content."""
    return hashlib.sha256(content.encode()).hexdigest()
