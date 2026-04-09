"""Generic persistent cache with TTL for long-term data storage.

This module provides a reusable caching mechanism for data that should persist
across pipeline runs and has a long TTL (typically 30 days).

Used by:
- URL metadata extraction (titles, images, descriptions)
- AI-generated model descriptions
- Any other expensive-to-compute data that rarely changes
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

T = TypeVar('T')


class PersistentCache(Generic[T]):
    """Generic persistent disk cache with configurable TTL.

    Features:
    - SHA256-based file naming (safe for all key types)
    - Configurable TTL with automatic expiration
    - Thread-safe file operations
    - JSON serialization with custom serializers/deserializers

    Example:
        # Simple string cache
        cache = PersistentCache[str](
            cache_subdir="model_descriptions",
            ttl_days=30
        )
        cache.set("model-id", "description text")
        description = cache.get("model-id")

        # Complex object cache with custom serialization
        cache = PersistentCache[URLMetadata](
            cache_subdir="url_metadata",
            ttl_days=30,
            serializer=lambda obj: obj.to_dict(),
            deserializer=URLMetadata.from_dict
        )
    """

    def __init__(
        self,
        cache_subdir: str,
        ttl_days: int = 30,
        base_dir: str = "data/cache",
        serializer: Optional[Callable[[T], Any]] = None,
        deserializer: Optional[Callable[[Any], T]] = None,
        auto_cleanup: bool = True,
    ):
        """Initialize persistent cache.

        Args:
            cache_subdir: Subdirectory name under base_dir (e.g., "url_metadata")
            ttl_days: Time-to-live in days (default: 30)
            base_dir: Base cache directory (default: "data/cache")
            serializer: Optional function to convert T to JSON-serializable data
            deserializer: Optional function to convert JSON data back to T
            auto_cleanup: Automatically clean expired entries on init (default: True)
        """
        self.cache_dir = Path(base_dir) / cache_subdir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)
        self.serializer = serializer or (lambda x: x)
        self.deserializer = deserializer or (lambda x: x)

        # Auto-cleanup expired entries on initialization
        if auto_cleanup:
            self._auto_cleanup()

    def _auto_cleanup(self):
        """Silently clean expired entries on initialization.

        This runs automatically to prevent cache buildup without requiring
        explicit cleanup calls. Failures are silently ignored to avoid
        breaking cache initialization.
        """
        try:
            self.clear_expired()
        except Exception:
            # Silently ignore cleanup failures - cache will still work
            pass

    def _get_cache_path(self, key: str) -> Path:
        """Convert cache key to file path using SHA256 hash.

        Args:
            key: Cache key (e.g., URL, model_id, etc.)

        Returns:
            Path to cache file
        """
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str) -> Optional[T]:
        """Get cached value if it exists and hasn't expired.

        Args:
            key: Cache key to look up

        Returns:
            Cached value if found and valid, None otherwise
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)

            # Check expiration
            cached_at = datetime.fromisoformat(cache_data['cached_at'])
            if datetime.now(timezone.utc) - cached_at > self.ttl:
                # Expired - delete cache file
                cache_path.unlink()
                return None

            # Valid cache - deserialize and return
            return self.deserializer(cache_data['data'])

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Corrupted cache file - delete it
            cache_path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: T, metadata: Optional[Dict[str, Any]] = None):
        """Store value in cache with current timestamp.

        Args:
            key: Cache key
            value: Value to cache
            metadata: Optional additional metadata to store (e.g., original key)
        """
        cache_path = self._get_cache_path(key)

        cache_data = {
            'cached_at': datetime.now(timezone.utc).isoformat(),
            'data': self.serializer(value),
        }

        # Merge in optional metadata
        if metadata:
            cache_data.update(metadata)

        with open(cache_path, 'w') as f:
            json.dump(cache_data, f, indent=2)

    def clear_expired(self) -> int:
        """Remove all expired cache entries.

        Returns:
            Number of expired entries removed
        """
        removed_count = 0

        for cache_file in self.cache_dir.glob('*.json'):
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)

                cached_at = datetime.fromisoformat(cache_data['cached_at'])
                if datetime.now(timezone.utc) - cached_at > self.ttl:
                    cache_file.unlink()
                    removed_count += 1

            except (json.JSONDecodeError, KeyError, ValueError):
                # Corrupted file - remove it
                cache_file.unlink(missing_ok=True)
                removed_count += 1

        return removed_count

    def clear_all(self) -> int:
        """Remove all cache entries regardless of expiration.

        Returns:
            Number of entries removed
        """
        removed_count = 0

        for cache_file in self.cache_dir.glob('*.json'):
            cache_file.unlink()
            removed_count += 1

        return removed_count

    def size(self) -> int:
        """Get current number of cached entries.

        Returns:
            Number of cache files
        """
        return sum(1 for _ in self.cache_dir.glob('*.json'))
