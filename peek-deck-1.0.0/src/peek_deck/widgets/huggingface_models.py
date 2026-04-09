"""HuggingFace trending models widget using HuggingFace API."""
from ..core.output_manager import OutputManager

from datetime import datetime, timezone
from typing import Any, Dict, List

from ..core.base_widget import BaseWidget
from ..core.url_fetch_manager import get_url_fetch_manager


class HuggingfaceModelsWidget(BaseWidget):
    """Displays trending models from HuggingFace.

    Optional params:
        - limit: Number of models to show (default: 10)
    """

    def get_required_params(self) -> list[str]:
        return []

    def fetch_data(self) -> Dict[str, Any]:
        """Fetch trending models from HuggingFace."""
        self.validate_params()

        limit = self.merged_params.get("limit", 10)
        client = get_url_fetch_manager()

        try:
            # Fetch trending models from HuggingFace API
            url = "https://huggingface.co/api/trending"
            params = {
                "limit": limit,
                "type": "model"
            }

            response = client.get(url, params=params, response_type="json")

            # Extract models from response
            models = []
            for item in response.get("recentlyTrending", []):
                model = item.get("repoData", {})

                # Format parameters (e.g., "7B", "13B")
                num_params = model.get("numParameters")
                if num_params:
                    # numParameters is a number
                    if isinstance(num_params, (int, float)):
                        if num_params >= 1e9:
                            num_params = f"{num_params / 1e9:.1f}B"
                        elif num_params >= 1e6:
                            num_params = f"{num_params / 1e6:.1f}M"
                        else:
                            num_params = f"{num_params:,}"

                # Get author info from authorData
                author_data = model.get("authorData", {})
                avatar_url = author_data.get("avatarUrl", "")
                author_fullname = author_data.get("fullname", "")

                # Construct thumbnail URL (same pattern as papers)
                model_id = model["id"]
                thumbnail_url = f"https://cdn-thumbnails.huggingface.co/social-thumbnails/models/{model_id}.png"

                # Fetch README content for AI summarization
                readme_url = f"https://huggingface.co/{model_id}/resolve/main/README.md"
                readme_content = ""
                try:
                    readme_response = client.get(readme_url, response_type="text")
                    # Take first 6000 chars to respect token limits (~1500 tokens)
                    readme_content = readme_response[:6000] if readme_response else ""
                except Exception as e:
                    OutputManager.log(f"⚠️  Failed to fetch README for {model_id}: {e}")

                models.append({
                    "id": model_id,
                    "author": model.get("author", model_id.split("/")[0]),
                    "author_fullname": author_fullname,
                    "name": model_id.split("/")[-1],
                    "url": f"https://huggingface.co/{model_id}",
                    "thumbnail": thumbnail_url,
                    "avatar": avatar_url,
                    "downloads": model.get("downloads", 0),
                    "likes": model.get("likes", 0),
                    "pipeline_tag": model.get("pipeline_tag"),
                    "num_parameters": num_params,
                    "last_modified": model.get("lastModified"),
                    "readme_content": readme_content,
                })

            # Enforce limit - defensive check in case API returns more than requested
            if len(models) > limit:
                models = models[:limit]

            data = {
                "models": models,
                "limit": limit,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

            OutputManager.log(f"✅ Fetched {len(models)} trending HuggingFace models")
            return data

        except Exception as e:
            OutputManager.log(f"❌ Failed to fetch HuggingFace trending models: {e}")
            raise

    def process_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI descriptions for models using Gemini with persistent caching."""
        import os
        import yaml
        import json
        from pathlib import Path
        from pydantic import BaseModel

        try:
            from google import genai
        except ImportError:
            OutputManager.log("ℹ️  google-genai not installed, skipping AI descriptions")
            return raw_data

        from ..core.persistent_cache import PersistentCache

        models = raw_data["models"]
        gemini_api_key = os.environ.get("GEMINI_API_KEY")

        if not gemini_api_key:
            OutputManager.log("ℹ️  No GEMINI_API_KEY found, skipping AI descriptions")
            return raw_data

        # Load LLM config from index.yaml
        config_path = Path.cwd() / "config" / "index.yaml"
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                llm_config = config.get("llm", {})
                model_name = llm_config.get("model")

                if not model_name:
                    OutputManager.log("ℹ️  No LLM model configured in config/index.yaml, skipping AI descriptions")
                    return raw_data
        except Exception as e:
            OutputManager.log(f"❌ Failed to load config: {e}")
            return raw_data

        # Initialize persistent cache (30-day TTL)
        description_cache = PersistentCache[str](
            cache_subdir="huggingface_model_descriptions",
            ttl_days=30
        )

        # Define JSON schema for structured output
        class ModelDescription(BaseModel):
            description: str

        # Initialize Gemini client
        client = genai.Client(api_key=gemini_api_key)

        # Track if any descriptions exist
        has_any_description = False

        for model in models:
            model_id = model["id"]

            # Check cache first
            cached_description = description_cache.get(model_id)
            if cached_description:
                model["description"] = cached_description
                has_any_description = True
                OutputManager.log(f"📦 Using cached description for {model['name']}")
                continue

            # No cache - generate new description
            if not model.get("readme_content"):
                # No README, skip LLM call
                continue

            # Construct summarization prompt
            prompt = f"""Summarize this HuggingFace model in 1-2 concise sentences for a dashboard widget.
Focus on: what it does, key capabilities, and primary use cases.
Be specific and technical but concise.

Model: {model['name']}
Pipeline: {model.get('pipeline_tag', 'unknown')}
README Content:
{model['readme_content']}"""

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": ModelDescription,
                        "temperature": 0.3,
                        "max_output_tokens": 300,
                    },
                )

                # Parse JSON response
                result = json.loads(response.text)
                description = result.get("description", "").strip()

                if description:
                    model["description"] = description
                    has_any_description = True

                    # Save to persistent cache
                    description_cache.set(
                        model_id,
                        description,
                        metadata={"model_id": model_id}  # Store original ID for debugging
                    )

                    OutputManager.log(f"✅ Generated and cached description for {model['name']}")
                else:
                    OutputManager.log(f"⚠️  Empty description returned for {model['name']}")

            except Exception as e:
                # Fail gracefully - no description is fine, will retry next run
                OutputManager.log(f"⚠️  Failed to generate description for {model['name']}: {e}")

        # Store whether any descriptions were generated (for UI rendering)
        raw_data["has_descriptions"] = has_any_description

        return raw_data

    def render(self, processed_data: Dict[str, Any]) -> str:
        """Render HuggingFace models widget HTML."""
        models = processed_data["models"]
        limit = processed_data["limit"]
        timestamp_iso = processed_data["fetched_at"]
        has_descriptions = processed_data.get("has_descriptions", False)

        return self.render_template(
            "widgets/huggingface_models.html",
            models=models,
            limit=limit,
            timestamp_iso=timestamp_iso,
            has_descriptions=has_descriptions
        )
