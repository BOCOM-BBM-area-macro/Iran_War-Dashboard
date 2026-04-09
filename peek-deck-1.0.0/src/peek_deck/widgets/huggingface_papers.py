"""HuggingFace daily papers widget using HuggingFace API."""
from ..core.output_manager import OutputManager

from datetime import datetime, timezone
from typing import Any, Dict, List

from ..core.base_widget import BaseWidget
from ..core.url_fetch_manager import get_url_fetch_manager


class HuggingfacePapersWidget(BaseWidget):
    """Displays daily AI research papers from HuggingFace.

    Optional params:
        - limit: Number of papers to show (default: 10)
        - sort: Sort order - "trending" or "publishedAt" (default: "trending")
    """

    def get_required_params(self) -> list[str]:
        return []

    def fetch_data(self) -> Dict[str, Any]:
        """Fetch daily papers from HuggingFace."""
        self.validate_params()

        limit = self.merged_params.get("limit", 10)
        sort = self.merged_params.get("sort", "trending")
        client = get_url_fetch_manager()

        try:
            # Fetch daily papers from HuggingFace API
            url = "https://huggingface.co/api/daily_papers"
            params = {
                "limit": limit,
                "sort": sort
            }

            response = client.get(url, params=params, response_type="json")

            # Extract papers from response
            papers = []
            for item in response:
                # Paper data is nested inside "paper" key
                paper = item.get("paper", {})

                # Extract author names
                authors = [author.get("name", "") for author in paper.get("authors", [])]
                author_str = ", ".join(authors[:3])  # Show first 3 authors
                if len(authors) > 3:
                    author_str += f" et al. ({len(authors)} authors)"

                # Extract organization info (if available)
                org = item.get("organization") or paper.get("organization")
                org_name = None
                org_fullname = None
                org_avatar = None
                if org:
                    org_name = org.get("name")
                    org_fullname = org.get("fullname")
                    org_avatar = org.get("avatar")

                # Use root-level fields which have some duplicates
                paper_id = paper["id"]

                papers.append({
                    "id": paper_id,
                    "title": item.get("title") or paper.get("title"),
                    "authors": author_str,
                    "organization_name": org_name,
                    "organization_fullname": org_fullname,
                    "organization_avatar": org_avatar,
                    "summary": item.get("summary") or paper.get("summary", ""),
                    "ai_summary": paper.get("ai_summary", ""),  # Concise AI-generated summary
                    "hf_url": f"https://huggingface.co/papers/{paper_id}",  # Primary link
                    "arxiv_url": f"https://arxiv.org/abs/{paper_id}",  # Secondary link
                    "thumbnail": item.get("thumbnail", ""),  # Paper preview image
                    "upvotes": paper.get("upvotes", 0),
                    "num_comments": item.get("numComments", 0),
                    "published_at": item.get("publishedAt") or paper.get("publishedAt"),
                    "github_repo": paper.get("githubRepo"),
                    "github_stars": paper.get("githubStars"),
                    "project_page": paper.get("projectPage"),
                })

            # Enforce limit - defensive check in case API returns more than requested
            if len(papers) > limit:
                papers = papers[:limit]

            data = {
                "papers": papers,
                "limit": limit,
                "sort": sort,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

            OutputManager.log(f"✅ Fetched {len(papers)} HuggingFace daily papers")
            return data

        except Exception as e:
            OutputManager.log(f"❌ Failed to fetch HuggingFace daily papers: {e}")
            raise

    def render(self, processed_data: Dict[str, Any]) -> str:
        """Render HuggingFace papers widget HTML."""
        papers = processed_data["papers"]
        limit = processed_data["limit"]
        sort = processed_data["sort"]
        timestamp_iso = processed_data["fetched_at"]

        return self.render_template(
            "widgets/huggingface_papers.html",
            papers=papers,
            limit=limit,
            sort=sort,
            timestamp_iso=timestamp_iso
        )
