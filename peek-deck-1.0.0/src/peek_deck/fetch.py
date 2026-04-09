"""Stage 1: Fetch raw data from external APIs."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from .core.cache import Cache
from .core.loader import discover_all_pages, load_page_config, create_widget_instance
from .core.output_manager import OutputManager


def fetch_widget(
    widget_type: str,
    widget_params: Dict[str, Any],
    page_params: Dict[str, Any],
    update_minutes: int,
    cache_key: str,
    data_raw_dir: Path,
    cache: Cache,
) -> Tuple[bool, List[str]]:
    """Fetch data for a single widget, capturing output.

    Args:
        widget_type: Type of widget to fetch
        widget_params: Widget-specific parameters
        page_params: Page-level parameters
        update_minutes: Update frequency in minutes
        cache_key: Cache key for this widget
        data_raw_dir: Directory to save raw data
        cache: Cache instance

    Returns:
        Tuple of (success: bool, output_lines: List[str])
    """
    # Enable output capture for this thread
    OutputManager.set_capture(True)

    try:
        # Create widget instance
        widget = create_widget_instance(
            widget_type=widget_type,
            params=widget_params,
            page_params=page_params,
            update_minutes=update_minutes
        )

        # Fetch data
        OutputManager.log(f"📡 Fetching {widget_type}...")
        raw_data = widget.fetch_data()

        # Save raw data
        raw_file = data_raw_dir / f"{cache_key}.json"
        with open(raw_file, 'w') as f:
            json.dump(raw_data, f, indent=2)

        # Mark widget as updated in cache
        cache.mark_updated(cache_key)

        OutputManager.log(f"✅ Saved to {raw_file.name}")
        return (True, OutputManager.get_output())

    except Exception as e:
        OutputManager.log(f"❌ Failed to fetch {widget_type}: {e}")
        return (False, OutputManager.get_output())


def fetch_all():
    """Fetch data for all widgets across all pages."""
    project_root = Path.cwd()
    data_raw_dir = project_root / "data" / "raw"
    cache_dir = project_root / "data" / "cache"

    # Create directories
    data_raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialize cache
    cache = Cache(cache_dir)

    print("🚀 Stage 1: Fetching data from external APIs\n")

    # Track stats
    total_widgets = 0
    fetched_count = 0
    skipped_count = 0
    failed_count = 0

    # Discover all pages
    page_files = discover_all_pages()
    if not page_files:
        print("❌ No pages found in pages/ directory")
        return

    print(f"📄 Found {len(page_files)} page(s)\n")

    # Collect all widgets to fetch
    widgets_to_fetch = []

    for page_file in page_files:
        # Load page config
        try:
            page_config = load_page_config(page_file)
        except Exception as e:
            print(f"❌ Failed to load {page_file.name}: {e}")
            continue

        if not page_config.enabled:
            print(f"⏭️  Skipping disabled page: {page_config.id}")
            continue

        print(f"📄 Page: {page_config.id} ({page_config.name}) [{page_config.category}]\n")

        # Collect widgets for this page
        for widget_config in page_config.widgets:
            total_widgets += 1
            widget_type = widget_config.type

            # Generate cache key
            cache_key = cache.get_cache_key(
                page_config.category,
                page_config.id,
                widget_type,
                widget_config.params
            )

            # Check if widget needs update
            if not cache.needs_update(cache_key, widget_config.update_minutes):
                skipped_count += 1
                continue

            # Add to fetch list
            widgets_to_fetch.append({
                'widget_type': widget_type,
                'widget_params': widget_config.params,
                'page_params': page_config.params,
                'update_minutes': widget_config.update_minutes,
                'cache_key': cache_key,
            })

    # Fetch widgets in parallel
    if widgets_to_fetch:
        print(f"\n🚀 Fetching {len(widgets_to_fetch)} widget(s) in parallel (max 10 concurrent)...\n")

        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    fetch_widget,
                    widget_data['widget_type'],
                    widget_data['widget_params'],
                    widget_data['page_params'],
                    widget_data['update_minutes'],
                    widget_data['cache_key'],
                    data_raw_dir,
                    cache,
                ): widget_data
                for widget_data in widgets_to_fetch
            }

            # Process results as they complete
            for future in as_completed(futures):
                success, output_lines = future.result()

                # Print output atomically (no interleaving)
                for line in output_lines:
                    print(f"    {line}")

                # Update counters
                if success:
                    fetched_count += 1
                else:
                    failed_count += 1

                print()  # Empty line between widgets

    # Save cache timestamps
    cache.save()

    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 Fetch Summary:")
    print(f"   Total widgets: {total_widgets}")
    print(f"   ✅ Fetched: {fetched_count}")
    print(f"   ⏭️  Skipped (cached): {skipped_count}")
    print(f"   ❌ Failed: {failed_count}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    fetch_all()
