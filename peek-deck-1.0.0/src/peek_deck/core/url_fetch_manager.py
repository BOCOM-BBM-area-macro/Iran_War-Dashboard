"""Centralized URL fetching with caching, rate limiting, and retry logic."""

import hashlib
import json
import requests
import time
from datetime import datetime, timezone, timedelta
from threading import Lock, Semaphore
from typing import Any, Dict, Optional, Literal
from urllib.parse import urlparse
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError


class URLFetchManager:
    """Centralized manager for all HTTP requests with caching and rate limiting.

    Features:
    - In-memory cache with TTL (prevents duplicate requests in single run)
    - Domain-level rate limiting (prevents overwhelming APIs)
    - Thread-safe for parallel widget fetching
    - Automatic retry with exponential backoff
    - Support for JSON, text, and binary responses

    Example usage:
        manager = get_url_fetch_manager()

        # Simple JSON request
        data = manager.get("https://api.example.com/data")

        # With query parameters
        data = manager.get(
            "https://api.example.com/search",
            params={"q": "bitcoin", "limit": 10}
        )

        # With custom headers
        data = manager.get(
            "https://api.example.com/data",
            headers={"Authorization": "Bearer token"},
            response_type="text"
        )
    """

    # Domain-specific delays (seconds) - single source of truth
    DOMAIN_DELAY = 1.0  # 1 second for all domains

    def __init__(
        self,
        cache_ttl_seconds: int = 180,  # 3 minute TTL
        default_timeout: int = 10,
        default_user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        max_retries: int = 3,
        retry_min_wait: int = 2,
        retry_max_wait: int = 10,
    ):
        """Initialize URL fetch manager.

        Args:
            cache_ttl_seconds: Cache TTL in seconds
            default_timeout: Default request timeout in seconds
            default_user_agent: Default User-Agent header
            max_retries: Maximum number of retry attempts
            retry_min_wait: Minimum wait time between retries (seconds)
            retry_max_wait: Maximum wait time between retries (seconds)
        """
        # Cache
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._cache_lock = Lock()
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)

        # Domain rate limiting
        self._domain_semaphores: Dict[str, Semaphore] = {}
        self._semaphore_lock = Lock()

        # HTTP settings
        self.default_timeout = default_timeout
        self.default_user_agent = default_user_agent
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait

    def _get_domain_semaphore(self, domain: str) -> Semaphore:
        """Get or create semaphore for domain (thread-safe).

        Each domain gets its own semaphore to ensure only one request
        at a time per domain.
        """
        if domain not in self._domain_semaphores:
            with self._semaphore_lock:
                # Double-check pattern for thread safety
                if domain not in self._domain_semaphores:
                    self._domain_semaphores[domain] = Semaphore(1)
        return self._domain_semaphores[domain]

    def _generate_cache_key(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate consistent cache key from URL, params, and relevant headers.

        Cache key includes:
        - URL
        - Sorted query parameters
        - Authorization headers (if present)

        This ensures the same request always generates the same cache key.
        """
        key_parts = [url]

        # Add sorted params
        if params:
            sorted_params = "&".join(
                f"{k}={v}" for k, v in sorted(params.items())
            )
            key_parts.append(sorted_params)

        # Add auth headers (they affect response content)
        if headers:
            auth_headers = {
                k: v for k, v in headers.items()
                if k.lower() in ("authorization", "api-key", "x-api-key")
            }
            if auth_headers:
                sorted_auth = "&".join(
                    f"{k}={v}" for k, v in sorted(auth_headers.items())
                )
                key_parts.append(sorted_auth)

        # Generate hash for long URLs
        cache_key = "|".join(key_parts)
        if len(cache_key) > 200:
            # Use hash for very long cache keys
            return hashlib.sha256(cache_key.encode()).hexdigest()

        return cache_key

    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check cache for valid entry (thread-safe)."""
        with self._cache_lock:
            if cache_key in self._cache:
                data, timestamp = self._cache[cache_key]
                if datetime.now(timezone.utc) - timestamp < self._cache_ttl:
                    return data
                else:
                    # Expired, remove it
                    del self._cache[cache_key]
        return None

    def _store_cache(self, cache_key: str, data: Any):
        """Store data in cache (thread-safe)."""
        with self._cache_lock:
            self._cache[cache_key] = (data, datetime.now(timezone.utc))

    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
    ) -> requests.Response:
        """Make HTTP request with retry logic.

        This method is decorated with @retry to handle transient failures.
        """
        # Merge with default headers (browser-like headers for better compatibility)
        final_headers = {
            "User-Agent": self.default_user_agent,
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        if headers:
            final_headers.update(headers)  # Custom headers override defaults

        # Use default timeout if not specified
        final_timeout = timeout or self.default_timeout

        # Make request
        response = requests.get(
            url,
            params=params,
            headers=final_headers,
            timeout=final_timeout
        )
        response.raise_for_status()

        return response

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        response_type: Literal["json", "text", "binary"] = "json",
        use_cache: bool = True,
    ) -> Any:
        """Make GET request with automatic caching, rate limiting, and retry logic.

        Thread-safe: Can be called from multiple threads simultaneously.
        Rate limiting: Only one request per domain at a time.

        Args:
            url: Target URL
            params: Query parameters
            headers: HTTP headers (merged with defaults)
            timeout: Request timeout in seconds (uses default if not specified)
            response_type: How to parse response ("json", "text", or "binary")
            use_cache: Whether to use cache (default: True)

        Returns:
            Parsed response data based on response_type

        Raises:
            requests.exceptions.RequestException: On HTTP errors after retries
            json.JSONDecodeError: If response_type="json" but response is not valid JSON
        """
        # 1. Generate cache key
        cache_key = self._generate_cache_key(url, params, headers)

        # 2. Check cache (thread-safe)
        if use_cache:
            cached = self._check_cache(cache_key)
            if cached is not None:
                return cached

        # 3. Extract domain for rate limiting
        domain = urlparse(url).netloc

        # 4. Get domain semaphore
        semaphore = self._get_domain_semaphore(domain)

        # 5. Acquire domain lock, fetch, delay, release
        with semaphore:
            # Make request with retry logic
            try:
                # Apply retry decorator dynamically
                retrying_request = retry(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential(
                        multiplier=1,
                        min=self.retry_min_wait,
                        max=self.retry_max_wait
                    ),
                    reraise=True
                )(self._make_request)

                response = retrying_request(url, params, headers, timeout)

                # Parse response based on type
                if response_type == "json":
                    data = response.json()
                elif response_type == "text":
                    data = response.text
                elif response_type == "binary":
                    data = response.content
                else:
                    raise ValueError(f"Invalid response_type: {response_type}")

                # Cache successful response (thread-safe)
                if use_cache:
                    self._store_cache(cache_key, data)

                # Rate limit delay BEFORE releasing semaphore
                # This ensures next request to this domain waits
                time.sleep(self.DOMAIN_DELAY)

                return data

            except RetryError as e:
                # All retries exhausted
                raise e.last_attempt.exception()
            except requests.exceptions.RequestException as e:
                raise
            except json.JSONDecodeError as e:
                raise

    def clear_cache(self):
        """Clear all cached responses (thread-safe)."""
        with self._cache_lock:
            self._cache.clear()


# Global instance for the application
_url_fetch_manager = URLFetchManager()


def get_url_fetch_manager() -> URLFetchManager:
    """Get the global URL fetch manager instance."""
    return _url_fetch_manager
