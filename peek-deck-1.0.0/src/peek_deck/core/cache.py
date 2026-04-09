"""Cache system for tracking widget update times and data."""

import json
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from .output_manager import OutputManager


class Cache:
    """Manages widget update timestamps and determines when widgets need refreshing.

    Thread-safe for parallel widget fetching.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "widget_timestamps.json"
        self.timestamps: Dict[str, str] = {}
        self._lock = Lock()
        self.load()

    def load(self):
        """Load timestamps from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.timestamps = json.load(f)
                OutputManager.log(f"✅ Loaded {len(self.timestamps)} cached timestamps")
            except Exception as e:
                OutputManager.log(f"⚠️  Failed to load cache: {e}")
                self.timestamps = {}
        else:
            self.timestamps = {}

    def save(self):
        """Save timestamps to disk (thread-safe)."""
        with self._lock:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.timestamps, f, indent=2)
                OutputManager.log(f"💾 Saved {len(self.timestamps)} timestamps to cache")
            except Exception as e:
                OutputManager.log(f"❌ Failed to save cache: {e}")

    def get_cache_key(self, category: str, page_id: str, widget_type: str, widget_params: Dict[str, Any]) -> str:
        """Generate unique cache key for a widget instance.

        Includes category to prevent collisions when page IDs are reused across categories.
        Args:
            category: Page category (e.g., "crypto", "tech") - formerly series_id
            page_id: Page identifier
            widget_type: Widget type
            widget_params: Widget parameters
        """
        base = f"{category}_{page_id}_{widget_type}"

        if not widget_params:
            return base

        # For long/complex params, use hash to keep filename short
        param_str = "_".join(f"{k}={v}" for k, v in sorted(widget_params.items()))

        # Sanitize filename - replace characters invalid on NTFS and other filesystems
        # Invalid chars: " : < > | * ? \r \n
        invalid_chars = [':', '"', '<', '>', '|', '*', '?', '\r', '\n']
        for char in invalid_chars:
            param_str = param_str.replace(char, '-')

        # If param string is too long (> 100 chars), use hash instead
        if len(param_str) > 100:
            param_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
            return f"{base}_{param_hash}"
        else:
            return f"{base}_{param_str}"

    def needs_update(self, cache_key: str, update_minutes: Optional[int]) -> bool:
        """Check if widget needs updating based on last update time (thread-safe)."""
        if update_minutes is None:
            # No update frequency specified, always update
            return True

        with self._lock:
            if cache_key not in self.timestamps:
                # Never updated before
                return True

            try:
                last_update = datetime.fromisoformat(self.timestamps[cache_key])
                time_since_update = datetime.now(timezone.utc) - last_update
                threshold = timedelta(minutes=update_minutes)
                needs_update = time_since_update >= threshold

                if needs_update:
                    OutputManager.log(f"🔄 {cache_key}: Last updated {time_since_update.total_seconds() / 60:.1f}m ago (threshold: {update_minutes}m)")
                else:
                    remaining = (threshold - time_since_update).total_seconds() / 60
                    OutputManager.log(f"⏭️  {cache_key}: Updated {time_since_update.total_seconds() / 60:.1f}m ago, skipping ({remaining:.1f}m remaining)")

                return needs_update
            except Exception as e:
                OutputManager.log(f"⚠️  Error checking cache for {cache_key}: {e}")
                return True  # Update on error

    def mark_updated(self, cache_key: str):
        """Mark widget as updated at current time (thread-safe)."""
        with self._lock:
            self.timestamps[cache_key] = datetime.now(timezone.utc).isoformat()

    def get_last_update(self, cache_key: str) -> Optional[datetime]:
        """Get the last update time for a widget (thread-safe)."""
        with self._lock:
            if cache_key in self.timestamps:
                try:
                    return datetime.fromisoformat(self.timestamps[cache_key])
                except Exception:
                    return None
            return None
