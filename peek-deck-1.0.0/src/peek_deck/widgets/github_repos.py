"""GitHub trending repositories widget using GitHub Search API."""
from ..core.output_manager import OutputManager

import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

from ..core.base_widget import BaseWidget
from ..core.url_fetch_manager import get_url_fetch_manager
from ..core.url_metadata import get_url_metadata_extractor


class GithubReposWidget(BaseWidget):
    """Displays trending repositories from GitHub.

    Optional params:
        - query: Search query (default: "ai machine-learning llm")
        - days: Number of days to look back (default: 30)
        - min_stars: Minimum stars filter (default: 0)
        - language: Filter by language (default: None)
        - limit: Number of repos to show (default: 10)
        - date_filter: Date filter type - "pushed" for recent activity or "created" for new repos (default: "pushed")
    """

    def get_required_params(self) -> list[str]:
        return []

    def fetch_data(self) -> Dict[str, Any]:
        """Fetch trending repos from GitHub."""
        self.validate_params()

        query = self.merged_params.get("query", "ai machine-learning llm")
        days = self.merged_params.get("days", 30)
        min_stars = self.merged_params.get("min_stars", 0)
        language = self.merged_params.get("language")
        limit = self.merged_params.get("limit", 10)
        date_filter = self.merged_params.get("date_filter", "pushed")  # "pushed" or "created"
        client = get_url_fetch_manager()

        try:
            # Build search query
            # Calculate date threshold based on filter type
            date_threshold = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

            # Build query string - separate parts with spaces
            # Use either pushed or created date filter
            q_parts = [query, f"{date_filter}:>{date_threshold}"]
            if min_stars > 0:
                q_parts.append(f"stars:>{min_stars}")
            if language:
                q_parts.append(f"language:{language}")

            q = " ".join(q_parts)

            # Fetch repos from GitHub Search API
            url = "https://api.github.com/search/repositories"
            params = {
                "q": q,
                "sort": "stars",
                "order": "desc",
                "per_page": limit
            }

            # Use GitHub token if available (from GitHub Actions or local .env)
            # GITHUB_TOKEN: 1,000 req/hour, PAT: 5,000 req/hour
            headers = {
                "Accept": "application/vnd.github.v3+json"
            }
            github_token = os.getenv("GITHUB_TOKEN")
            if github_token:
                headers["Authorization"] = f"Bearer {github_token}"
                OutputManager.log("🔑 Using GitHub token for authentication (1,000 req/hour)")
            else:
                OutputManager.log("ℹ️  No GitHub token found, using unauthenticated API (60 req/hour)")

            response = client.get(url, params=params, headers=headers, response_type="json")

            # Get metadata extractor for preview images
            metadata_extractor = get_url_metadata_extractor()

            # Extract repos from response
            repos = []
            for repo in response.get("items", []):
                # Format star count
                stars = repo.get("stargazers_count", 0)
                if stars >= 1000:
                    stars_display = f"{stars / 1000:.1f}k"
                else:
                    stars_display = str(stars)

                # Format fork count
                forks = repo.get("forks_count", 0)
                if forks >= 1000:
                    forks_display = f"{forks / 1000:.1f}k"
                else:
                    forks_display = str(forks)

                # Extract metadata for preview image
                repo_url = repo["html_url"]
                metadata = metadata_extractor.extract(repo_url)
                preview_image = metadata.image if metadata else None

                # Get owner avatar from API
                owner_avatar = repo["owner"].get("avatar_url", "")

                repos.append({
                    "full_name": repo["full_name"],
                    "name": repo["name"],
                    "owner": repo["owner"]["login"],
                    "owner_avatar": owner_avatar,
                    "description": repo.get("description", ""),
                    "url": repo_url,
                    "preview_image": preview_image,
                    "stars": stars,
                    "stars_display": stars_display,
                    "forks": forks,
                    "forks_display": forks_display,
                    "language": repo.get("language"),
                    "topics": repo.get("topics", [])[:5],  # Show up to 5 topics; CSS enforces 2-line display limit
                    "updated_at": repo.get("pushed_at"),  # Use pushed_at to match GitHub.com
                })

            # Enforce limit - defensive check in case API returns more than requested
            if len(repos) > limit:
                repos = repos[:limit]

            data = {
                "repos": repos,
                "query": query,
                "search_query": q,  # Full search query with filters
                "days": days,
                "limit": limit,
                "total_count": response.get("total_count", 0),
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

            OutputManager.log(f"✅ Fetched {len(repos)} GitHub repos for query '{query}' (filter: {date_filter}, days: {days})")
            return data

        except Exception as e:
            OutputManager.log(f"❌ Failed to fetch GitHub repos: {e}")
            raise

    def render(self, processed_data: Dict[str, Any]) -> str:
        """Render GitHub repos widget HTML."""
        repos = processed_data["repos"]
        query = processed_data["query"]
        search_query = processed_data["search_query"]
        days = processed_data["days"]
        limit = processed_data["limit"]
        total_count = processed_data["total_count"]
        timestamp_iso = processed_data["fetched_at"]

        return self.render_template(
            "widgets/github_repos.html",
            repos=repos,
            query=query,
            search_query=search_query,
            days=days,
            limit=limit,
            total_count=total_count,
            timestamp_iso=timestamp_iso
        )
