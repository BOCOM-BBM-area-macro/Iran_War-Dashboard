"""PeekDeck - Configurable widget-based monitoring system.

A glance is all you need.
"""

# Import from central config (single source of truth)
import sys
from pathlib import Path

# Add project root to path to import project_config
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

from project_config import PROJECT_NAME, PROJECT_TAGLINE, __version__

__all__ = ["PROJECT_NAME", "PROJECT_TAGLINE", "__version__"]
