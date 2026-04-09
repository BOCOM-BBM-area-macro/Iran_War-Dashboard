"""Core framework components."""

from .base_widget import BaseWidget, WidgetData
from .cache import Cache
from .config import PageConfig, WidgetConfig
from .url_fetch_manager import URLFetchManager, get_url_fetch_manager
from .output_manager import OutputManager
from .url_metadata import (
    URLMetadata,
    URLMetadataExtractor,
    extract_url_metadata,
    get_url_metadata_extractor,
)

__all__ = [
    "BaseWidget",
    "WidgetData",
    "Cache",
    "PageConfig",
    "WidgetConfig",
    "URLFetchManager",
    "get_url_fetch_manager",
    "OutputManager",
    "URLMetadata",
    "URLMetadataExtractor",
    "extract_url_metadata",
    "get_url_metadata_extractor",
]
