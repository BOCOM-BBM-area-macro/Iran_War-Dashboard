"""URL metadata extraction for rich link previews.

Extracts metadata from web pages including:
- Open Graph tags (og:image, og:description, og:title, og:site_name)
- Twitter Card tags (twitter:image, twitter:description, twitter:title)
- Standard meta tags (description, keywords)
- Favicons (various formats and sizes)
- Page title

Uses persistent disk cache with long TTL to minimize fetches (most metadata doesn't change).
Follows design patterns from DESIGN.md (graceful degradation, retry logic, emoji logging).
"""

from typing import Dict, Any, Optional
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

from .url_fetch_manager import get_url_fetch_manager
from .output_manager import OutputManager
from .persistent_cache import PersistentCache


class URLMetadata:
    """Extracted metadata from a URL."""

    def __init__(self, url: str):
        self.url = url
        self.title: Optional[str] = None
        self.description: Optional[str] = None
        self.image: Optional[str] = None
        self.favicon: Optional[str] = None
        self.site_name: Optional[str] = None
        self.author: Optional[str] = None
        self.keywords: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "image": self.image,
            "favicon": self.favicon,
            "site_name": self.site_name,
            "author": self.author,
            "keywords": self.keywords,
        }

    def has_rich_data(self) -> bool:
        """Check if metadata has any rich content (image or description)."""
        return bool(self.image or self.description)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'URLMetadata':
        """Create URLMetadata from dictionary."""
        metadata = cls(data['url'])
        metadata.title = data.get('title')
        metadata.description = data.get('description')
        metadata.image = data.get('image')
        metadata.favicon = data.get('favicon')
        metadata.site_name = data.get('site_name')
        metadata.author = data.get('author')
        metadata.keywords = data.get('keywords')
        return metadata


class URLMetadataExtractor:
    """Extract rich metadata from web pages.

    Supports Open Graph, Twitter Cards, standard meta tags, and favicons.
    Uses persistent disk cache with 30-day TTL to minimize fetches.
    Extremely robust to handle slow sites, 404s, timeouts, and any other failures.

    Example usage:
        extractor = get_url_metadata_extractor()
        metadata = extractor.extract("https://example.com/article")

        if metadata and metadata.has_rich_data():
            OutputManager.log(f"Title: {metadata.title}")
            OutputManager.log(f"Image: {metadata.image}")
            OutputManager.log(f"Description: {metadata.description}")
    """

    def __init__(self, persistent_cache: Optional[PersistentCache[URLMetadata]] = None):
        """Initialize metadata extractor with HTTP client and persistent cache.

        Args:
            persistent_cache: Persistent cache instance (creates new one if not provided)
        """
        self.http_client = get_url_fetch_manager()
        self.persistent_cache = persistent_cache or PersistentCache[URLMetadata](
            cache_subdir="url_metadata",
            ttl_days=30,
            serializer=lambda meta: meta.to_dict(),
            deserializer=URLMetadata.from_dict
        )

    def extract(
        self,
        url: str,
        timeout: int = 5,
        use_cache: bool = True,
        force_refetch: bool = False,
    ) -> Optional[URLMetadata]:
        """Extract metadata from a URL with aggressive caching.

        Args:
            url: Target URL to extract metadata from
            timeout: Request timeout in seconds (default: 5 - fast timeout for robustness)
            use_cache: Whether to use persistent cache (default: True)
            force_refetch: Force refetch even if cached (default: False)

        Returns:
            URLMetadata object with extracted data, or empty URLMetadata if extraction fails

        Notes:
            - **Aggressive caching**: Uses 30-day persistent cache, almost always returns cached data
            - **Extremely robust**: Handles 404, timeouts, slow sites, malformed HTML, etc.
            - **Never raises exceptions**: Always returns URLMetadata (may be empty)
            - **Fast timeout**: 5 seconds to avoid blocking on slow sites
            - Follows redirects automatically
            - Prefers Open Graph tags over standard meta tags
        """
        # Check persistent cache first (unless force_refetch)
        if use_cache and not force_refetch:
            cached = self.persistent_cache.get(url)
            if cached is not None:
                return cached

        try:
            # Fetch HTML content with short timeout for robustness
            html = self.http_client.get(
                url,
                response_type="text",
                timeout=timeout,
                use_cache=False,  # Don't use HTTP cache, use persistent cache instead
            )

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            # Create metadata object
            metadata = URLMetadata(url)

            # Extract metadata in order of preference
            self._extract_open_graph(soup, metadata, url)
            self._extract_twitter_card(soup, metadata, url)
            self._extract_standard_meta(soup, metadata)
            self._extract_title(soup, metadata)
            self._extract_favicon(soup, metadata, url)

            # Save to persistent cache (even if empty - prevents retrying failed URLs)
            if use_cache:
                self.persistent_cache.set(url, metadata)

            return metadata

        except Exception as e:
            # Silently handle errors - metadata extraction is non-critical

            # Return empty metadata object (graceful degradation)
            empty_metadata = URLMetadata(url)

            # Cache the failure too (prevents retrying same failed URL repeatedly)
            if use_cache:
                self.persistent_cache.set(url, empty_metadata)

            return empty_metadata

    def _extract_open_graph(
        self,
        soup: BeautifulSoup,
        metadata: URLMetadata,
        base_url: str
    ):
        """Extract Open Graph metadata (og:* tags).

        Open Graph tags are the most reliable and comprehensive metadata source.
        They're used by Facebook, LinkedIn, Slack, and most social platforms.
        """
        og_tags = {
            'og:title': 'title',
            'og:description': 'description',
            'og:image': 'image',
            'og:site_name': 'site_name',
        }

        for og_property, attr_name in og_tags.items():
            tag = soup.find('meta', property=og_property)
            if tag and tag.get('content'):
                value = tag['content'].strip()

                # Resolve relative URLs for images
                if og_property == 'og:image' and value:
                    value = urljoin(base_url, value)

                setattr(metadata, attr_name, value)

    def _extract_twitter_card(
        self,
        soup: BeautifulSoup,
        metadata: URLMetadata,
        base_url: str
    ):
        """Extract Twitter Card metadata (twitter:* tags).

        Twitter Cards are used as fallback when Open Graph tags are missing.
        Only extract if the corresponding field is still empty.
        """
        twitter_tags = {
            'twitter:title': 'title',
            'twitter:description': 'description',
            'twitter:image': 'image',
        }

        for twitter_name, attr_name in twitter_tags.items():
            # Only extract if field is still empty (prefer Open Graph)
            if getattr(metadata, attr_name):
                continue

            tag = soup.find('meta', attrs={'name': twitter_name})
            if tag and tag.get('content'):
                value = tag['content'].strip()

                # Resolve relative URLs for images
                if twitter_name == 'twitter:image' and value:
                    value = urljoin(base_url, value)

                setattr(metadata, attr_name, value)

    def _extract_standard_meta(self, soup: BeautifulSoup, metadata: URLMetadata):
        """Extract standard HTML meta tags.

        These are the traditional <meta name="description"> tags.
        Only used as final fallback if Open Graph and Twitter Cards are missing.
        """
        # Description
        if not metadata.description:
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            if desc_tag and desc_tag.get('content'):
                metadata.description = desc_tag['content'].strip()

        # Keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag and keywords_tag.get('content'):
            metadata.keywords = keywords_tag['content'].strip()

        # Author
        author_tag = soup.find('meta', attrs={'name': 'author'})
        if author_tag and author_tag.get('content'):
            metadata.author = author_tag['content'].strip()

    def _extract_title(self, soup: BeautifulSoup, metadata: URLMetadata):
        """Extract page title from <title> tag.

        Only used if no title from Open Graph or Twitter Cards.
        """
        if not metadata.title:
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                metadata.title = title_tag.string.strip()

    def _extract_favicon(
        self,
        soup: BeautifulSoup,
        metadata: URLMetadata,
        base_url: str
    ):
        """Extract favicon URL.

        Looks for favicons in order of preference:
        1. Apple touch icon (high resolution)
        2. Standard rel="icon" link
        3. rel="shortcut icon" (legacy)
        4. Default /favicon.ico location
        """
        # Parse base URL for default favicon
        parsed = urlparse(base_url)
        default_favicon = f"{parsed.scheme}://{parsed.netloc}/favicon.ico"

        # Try different favicon types in order of preference
        favicon_selectors = [
            ('link', {'rel': 'apple-touch-icon'}),
            ('link', {'rel': 'icon'}),
            ('link', {'rel': 'shortcut icon'}),
        ]

        for tag_name, attrs in favicon_selectors:
            # Use find_all to check ALL matching links (not just first)
            tags = soup.find_all(tag_name, attrs=attrs)
            for tag in tags:
                if tag and tag.get('href'):
                    favicon_url = tag['href'].strip()

                    # Skip invalid hrefs (fragments, empty, javascript:, data:)
                    if not favicon_url or favicon_url in ['#', ''] or favicon_url.startswith(('javascript:', 'data:')):
                        continue

                    # Resolve relative URLs
                    resolved_url = urljoin(base_url, favicon_url)

                    # Skip if resolution resulted in the base URL (means it was just a fragment)
                    if resolved_url != base_url and resolved_url != base_url + '#':
                        metadata.favicon = resolved_url
                        return

        # Fallback to default /favicon.ico
        metadata.favicon = default_favicon

    def extract_batch(
        self,
        urls: list[str],
        timeout: int = 5,
        use_cache: bool = True,
        force_refetch: bool = False,
    ) -> Dict[str, URLMetadata]:
        """Extract metadata from multiple URLs with aggressive caching.

        Args:
            urls: List of URLs to extract metadata from
            timeout: Request timeout per URL in seconds (default: 5)
            use_cache: Whether to use persistent cache (default: True)
            force_refetch: Force refetch even if cached (default: False)

        Returns:
            Dictionary mapping URL to URLMetadata (never None, may be empty)

        Notes:
            - **Aggressive caching**: Most URLs will hit 30-day cache
            - **Extremely robust**: Individual failures don't stop batch processing
            - **Never returns None**: All URLs get URLMetadata (may be empty)
            - Processes URLs sequentially (respects rate limits)
            - Duplicate URLs only fetched once
        """
        results = {}

        for url in urls:
            # extract() never raises, always returns URLMetadata
            metadata = self.extract(
                url,
                timeout=timeout,
                use_cache=use_cache,
                force_refetch=force_refetch
            )
            results[url] = metadata

        # Count successful extractions with rich data
        return results


# Global instance
_url_metadata_extractor = None


def get_url_metadata_extractor() -> URLMetadataExtractor:
    """Get the global URL metadata extractor instance.

    Uses singleton pattern to ensure all widgets share the same HTTP cache.
    """
    global _url_metadata_extractor
    if _url_metadata_extractor is None:
        _url_metadata_extractor = URLMetadataExtractor()
    return _url_metadata_extractor


def extract_url_metadata(
    url: str,
    use_cache: bool = True,
    force_refetch: bool = False,
    timeout: int = 5,
) -> URLMetadata:
    """Convenience function to extract metadata from a single URL with aggressive caching.

    This is a simple wrapper around get_url_metadata_extractor().extract()
    for easier usage in widgets.

    Args:
        url: URL to extract metadata from
        use_cache: Whether to use persistent 30-day cache (default: True)
        force_refetch: Force refetch even if cached (default: False)
        timeout: Request timeout in seconds (default: 5)

    Returns:
        URLMetadata object with extracted data (never None, may be empty)

    Example:
        from ..core.url_metadata import extract_url_metadata

        metadata = extract_url_metadata("https://example.com/article")
        if metadata.has_rich_data():
            OutputManager.log(f"Title: {metadata.title}")
            OutputManager.log(f"Image: {metadata.image}")

    Notes:
        - Almost always returns cached data (30-day TTL)
        - Never raises exceptions
        - Fast 5-second timeout for robustness
    """
    extractor = get_url_metadata_extractor()
    return extractor.extract(
        url,
        use_cache=use_cache,
        force_refetch=force_refetch,
        timeout=timeout
    )
