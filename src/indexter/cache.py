from pathlib import Path
from typing import TYPE_CHECKING

from indexter.config import settings

if TYPE_CHECKING:
    from indexter.models import Repo


class CacheManager:
    """Manages cached data for a repository."""

    def __init__(self, repo: "Repo") -> None:
        self.repo = repo
        self.cache_dir = settings.cache_dir / self.repo.name
        self.cache_key_prefix = f"{self.repo.name}"

    def _key_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{self.cache_key_prefix}_{key}.json"

    def get(self, key: str) -> str | None:
        """Get cached data by key."""
        path = self._key_path(key)
        if not path.exists():
            return None
        return path.read_text()

    def set(self, key: str, data: str) -> None:
        """Set cached data by key."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._key_path(key).write_text(data)

    def delete(self, key: str) -> bool:
        """Delete cached data by key. Returns True if deleted."""
        path = self._key_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> None:
        """Clear all cached data for this repository."""
        if self.cache_dir.exists():
            for file in self.cache_dir.iterdir():
                file.unlink()
            self.cache_dir.rmdir()
