"""Thread-safe output manager for parallel widget fetching.

Supports both direct printing (default) and output capture (for parallel execution).
Uses thread-local storage so each thread can independently capture or print output.
"""

import threading
from typing import List, Optional


class OutputManager:
    """Thread-safe output manager using thread-local storage.

    Each thread can independently:
    - Capture output to a list (for atomic printing later)
    - Print directly (default behavior)

    Usage:
        # In parallel worker thread:
        OutputManager.set_capture(True)
        widget.fetch_data()  # All logs captured
        lines = OutputManager.get_output()

        # In widgets/url_manager:
        OutputManager.log("✅ Fetched data")  # Automatically captured or printed
    """

    _local = threading.local()

    @classmethod
    def set_capture(cls, enabled: bool = True):
        """Enable or disable output capture for current thread.

        Args:
            enabled: If True, capture output to list. If False, print directly.
        """
        if enabled:
            cls._local.output = []
        else:
            cls._local.output = None

    @classmethod
    def log(cls, message: str, indent: int = 0):
        """Log a message - either capture to list or print directly.

        Thread-safe: Uses thread-local storage, so each thread has its own output.

        Args:
            message: Message to log
            indent: Number of spaces to indent (default: 0)
        """
        if indent > 0:
            message = " " * indent + message

        # Check if current thread has capture enabled
        if hasattr(cls._local, 'output') and cls._local.output is not None:
            cls._local.output.append(message)
        else:
            print(message)

    @classmethod
    def get_output(cls) -> List[str]:
        """Get captured output for current thread.

        Returns:
            List of captured log messages. Empty if capture not enabled.
        """
        if hasattr(cls._local, 'output') and cls._local.output is not None:
            return cls._local.output
        return []

    @classmethod
    def clear(cls):
        """Clear captured output for current thread."""
        if hasattr(cls._local, 'output') and cls._local.output is not None:
            cls._local.output = []
