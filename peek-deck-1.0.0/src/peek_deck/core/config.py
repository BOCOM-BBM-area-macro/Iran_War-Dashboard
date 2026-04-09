"""Configuration models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WidgetConfig(BaseModel):
    """Configuration for a single widget instance."""

    type: str  # e.g., "crypto-price", "news"
    params: Dict[str, Any] = Field(default_factory=dict)
    update_minutes: Optional[int] = Field(None, gt=0)
    max_cache_age: Optional[int] = Field(None, gt=0)  # Minutes. None = never expire

    class Config:
        extra = 'forbid'  # Reject unknown fields


class PageConfig(BaseModel):
    """Configuration for a single page (dashboard)."""

    category: str = 'general'  # e.g., "crypto", "tech" - used for index grouping
    id: str = Field(pattern=r'^[a-z0-9-]+$')
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    enabled: bool = True
    theme: Optional[Dict[str, str]] = None  # Optional page-specific theme
    params: Dict[str, Any] = Field(default_factory=dict)
    widgets: List[WidgetConfig] = Field(min_length=1)

    class Config:
        extra = 'forbid'


