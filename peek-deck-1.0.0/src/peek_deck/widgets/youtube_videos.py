"""YouTube Videos widget using YouTube Data API v3.

IMPORTANT: This widget requires YOUTUBE_API_KEY environment variable.

Quota usage:
- Free tier: 10,000 units/day
- Search cost: 100 units per request
- Video details: 1 unit per batch (up to 50 videos)
- Channel avatars: 1 unit per batch (up to 50 channels)
- Total per widget: 102 units
- Practical limit: ~98 searches per day
"""
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from ..core.base_widget import BaseWidget
from ..core.output_manager import OutputManager
from ..core.url_fetch_manager import get_url_fetch_manager


def parse_iso8601_duration(duration: str) -> str:
    """Parse ISO 8601 duration to readable format.

    Args:
        duration: ISO 8601 duration (e.g., "PT12M34S", "PT1H2M3S", "PT45S")

    Returns:
        Readable duration (e.g., "12:34", "1:02:03", "0:45")

    Examples:
        >>> parse_iso8601_duration("PT12M34S")
        "12:34"
        >>> parse_iso8601_duration("PT1H2M3S")
        "1:02:03"
        >>> parse_iso8601_duration("PT45S")
        "0:45"
    """
    if not duration:
        return ""

    # Parse ISO 8601 duration: PT#H#M#S
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
    if not match:
        return ""

    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"


def format_count(count: int) -> str:
    """Format large numbers with K/M/B suffixes.

    Args:
        count: Number to format

    Returns:
        Formatted string (e.g., "1.2M", "45K", "234")

    Examples:
        >>> format_count(1234567)
        "1.2M"
        >>> format_count(45678)
        "45K"
        >>> format_count(234)
        "234"
    """
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.1f}B"
    elif count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.0f}K"
    else:
        return str(count)


class YoutubeVideosWidget(BaseWidget):
    """Displays recent YouTube videos based on search query.

    Uses YouTube Data API v3 to search for videos and display them in a carousel.

    IMPORTANT: Requires YOUTUBE_API_KEY environment variable.
    Free tier quota: 10,000 units/day (100 searches/day at 100 units each).

    Required params:
        - query: Search query (e.g., "Bitcoin", "AI tutorial", "machine learning")

    Optional params:
        - title: Custom widget title (default: "YouTube Videos")
        - limit: Number of videos to show (default: 10, max: 50)
        - order: Sort order - "relevance" (default), "date", "viewCount", "rating"
        - region_code: Two-letter ISO country code (default: "US")

    Time-based filters:
        - days: Only show videos from last N days (e.g., 3 for last 3 days)
        - published_after: RFC 3339 datetime (e.g., "2025-01-15T00:00:00Z")
        - published_before: RFC 3339 datetime

    Quality filters (improve value of shown videos):
        - video_duration: "any" (default), "short" (<4min), "medium" (4-20min), "long" (>20min)
        - video_definition: "any" (default), "high" (HD only), "standard"
        - video_caption: "any" (default), "closedCaption" (only captioned), "none"
        - video_embeddable: "any" (default), "true" (only embeddable)
        - video_license: "any" (default), "creativeCommon", "youtube"
        - safe_search: "moderate" (default), "none", "strict"
        - relevance_language: ISO 639-1 code (e.g., "en", "zh-Hans")
    """

    def get_required_params(self) -> list[str]:
        return ["query"]

    def fetch_data(self) -> Dict[str, Any]:
        """Fetch videos from YouTube Data API v3."""
        self.validate_params()

        query = self.merged_params["query"]
        title = self.merged_params.get("title", "YouTube Videos")
        limit = min(self.merged_params.get("limit", 10), 50)  # API max is 50
        order = self.merged_params.get("order", "relevance")
        region_code = self.merged_params.get("region_code", "US")

        # Get API key from environment
        api_key = os.environ.get("YOUTUBE_API_KEY")
        if not api_key:
            raise ValueError(
                "YOUTUBE_API_KEY environment variable not set. "
                "Get your API key from https://console.cloud.google.com/apis/credentials"
            )

        client = get_url_fetch_manager()

        try:
            # Call YouTube Data API v3 search endpoint
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": query,
                "type": "video",  # Only search for videos
                "maxResults": limit,
                "order": order,
                "regionCode": region_code,
                "key": api_key,
            }

            # Time-based filters
            if "days" in self.merged_params:
                # Calculate publishedAfter from days ago
                from datetime import timedelta
                days_ago = datetime.now(timezone.utc) - timedelta(days=self.merged_params["days"])
                params["publishedAfter"] = days_ago.isoformat().replace("+00:00", "Z")

            if "published_after" in self.merged_params:
                params["publishedAfter"] = self.merged_params["published_after"]

            if "published_before" in self.merged_params:
                params["publishedBefore"] = self.merged_params["published_before"]

            # Quality filters (all require type=video)
            if "video_duration" in self.merged_params:
                params["videoDuration"] = self.merged_params["video_duration"]

            if "video_definition" in self.merged_params:
                params["videoDefinition"] = self.merged_params["video_definition"]

            if "video_caption" in self.merged_params:
                params["videoCaption"] = self.merged_params["video_caption"]

            if "video_embeddable" in self.merged_params:
                params["videoEmbeddable"] = self.merged_params["video_embeddable"]

            if "video_license" in self.merged_params:
                params["videoLicense"] = self.merged_params["video_license"]

            if "safe_search" in self.merged_params:
                params["safeSearch"] = self.merged_params["safe_search"]

            if "relevance_language" in self.merged_params:
                params["relevanceLanguage"] = self.merged_params["relevance_language"]

            OutputManager.log(f"📡 Fetching YouTube videos for '{query}'...")
            data = client.get(url, params=params)

            # Extract video information
            videos = []
            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                video_id = item.get("id", {}).get("videoId")

                if not video_id:
                    continue

                # Parse published date
                published_at = snippet.get("publishedAt")
                pub_timestamp = None
                if published_at:
                    try:
                        # Parse ISO 8601 format: "2025-01-15T10:30:00Z"
                        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                        pub_timestamp = dt.timestamp()
                    except (ValueError, AttributeError):
                        pass

                # Get best thumbnail (prefer high > medium > default)
                thumbnails = snippet.get("thumbnails", {})
                thumbnail = None
                for quality in ["high", "medium", "default"]:
                    if quality in thumbnails:
                        thumbnail = thumbnails[quality].get("url")
                        break

                videos.append({
                    "video_id": video_id,
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "channel_name": snippet.get("channelTitle", ""),
                    "channel_id": snippet.get("channelId", ""),
                    "thumbnail": thumbnail,
                    "published_at": pub_timestamp,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                })

            # Enforce limit - defensive check in case API returns more than requested
            if len(videos) > limit:
                OutputManager.log(f"⚠️  YouTube API returned {len(videos)} videos, limiting to {limit}")
                videos = videos[:limit]

            OutputManager.log(f"✅ Fetched {len(videos)} YouTube videos for '{query}'")

            # Step 2: Fetch video details (duration, statistics) in batch
            if videos:
                OutputManager.log(f"📊 Fetching video details (duration, stats) for {len(videos)} videos...")
                video_ids = [v["video_id"] for v in videos]
                video_ids_str = ",".join(video_ids)

                try:
                    # Batch fetch video details (1 unit for up to 50 videos)
                    details_url = "https://www.googleapis.com/youtube/v3/videos"
                    details_params = {
                        "part": "contentDetails,statistics",
                        "id": video_ids_str,
                        "key": api_key,
                    }
                    details_data = client.get(details_url, params=details_params)

                    # Step 3: Fetch channel avatars in batch
                    # Get unique channel IDs
                    channel_ids = list(set(v["channel_id"] for v in videos if v["channel_id"]))
                    channel_ids_str = ",".join(channel_ids)

                    OutputManager.log(f"👤 Fetching channel avatars for {len(channel_ids)} channels...")
                    channels_url = "https://www.googleapis.com/youtube/v3/channels"
                    channels_params = {
                        "part": "snippet",
                        "id": channel_ids_str,
                        "key": api_key,
                    }
                    channels_data = client.get(channels_url, params=channels_params)

                    # Create channel avatar lookup map
                    channel_avatars = {}
                    for item in channels_data.get("items", []):
                        channel_id = item.get("id")
                        thumbnails = item.get("snippet", {}).get("thumbnails", {})
                        # Prefer default (88x88) size for channel avatars
                        avatar_url = thumbnails.get("default", {}).get("url", "")
                        if avatar_url:
                            channel_avatars[channel_id] = avatar_url

                    # Create lookup map: video_id -> details
                    details_map = {}
                    for item in details_data.get("items", []):
                        video_id = item.get("id")
                        content_details = item.get("contentDetails", {})
                        statistics = item.get("statistics", {})

                        details_map[video_id] = {
                            "duration": parse_iso8601_duration(content_details.get("duration", "")),
                            "view_count": int(statistics.get("viewCount", 0)),
                            "view_count_display": format_count(int(statistics.get("viewCount", 0))),
                            "like_count": int(statistics.get("likeCount", 0)),
                            "like_count_display": format_count(int(statistics.get("likeCount", 0))),
                            "comment_count": int(statistics.get("commentCount", 0)),
                            "comment_count_display": format_count(int(statistics.get("commentCount", 0))),
                        }

                    # Merge details and channel avatars back into videos
                    for video in videos:
                        video_id = video["video_id"]
                        channel_id = video["channel_id"]

                        # Add video details
                        if video_id in details_map:
                            video.update(details_map[video_id])
                        else:
                            # Fallback if details not found
                            video.update({
                                "duration": "",
                                "view_count": 0,
                                "view_count_display": "0",
                                "like_count": 0,
                                "like_count_display": "0",
                                "comment_count": 0,
                                "comment_count_display": "0",
                            })

                        # Add channel avatar
                        video["channel_avatar"] = channel_avatars.get(channel_id, "")

                    OutputManager.log(f"✅ Enriched {len(videos)} videos with duration, stats, and avatars")

                except Exception as e:
                    OutputManager.log(f"⚠️  Failed to fetch video details: {e}")
                    # Continue without details - videos will have empty stats
                    for video in videos:
                        video.update({
                            "duration": "",
                            "view_count": 0,
                            "view_count_display": "0",
                            "like_count": 0,
                            "like_count_display": "0",
                            "comment_count": 0,
                            "comment_count_display": "0",
                            "channel_avatar": "",
                        })

            result = {
                "title": title,
                "query": query,
                "order": order,
                "videos": videos,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

            # Log quota usage
            quota_used = 102 if videos else 100  # Search (100) + video details (1) + channels (1)
            OutputManager.log(f"⚠️  Quota usage: {quota_used} units (10,000 units/day limit)")

            return result

        except Exception as e:
            OutputManager.log(f"❌ Failed to fetch YouTube videos for '{query}': {e}")
            raise

    def render(self, processed_data: Dict[str, Any]) -> str:
        """Render YouTube Videos widget HTML."""
        title = processed_data["title"]
        query = processed_data["query"]
        order = processed_data["order"]
        videos = processed_data["videos"]
        timestamp_iso = processed_data["fetched_at"]

        return self.render_template(
            "widgets/youtube_videos.html",
            title=title,
            query=query,
            order=order,
            videos=videos,
            timestamp_iso=timestamp_iso,
        )
