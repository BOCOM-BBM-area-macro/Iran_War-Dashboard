"""Stage 3: Render HTML from processed data."""

import json
from datetime import datetime, timezone
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from .core.cache import Cache
from .core.loader import (
    discover_all_pages,
    load_page_config,
    create_widget_instance
)
# Import project name from central config
from peek_deck import PROJECT_NAME, PROJECT_TAGLINE


def render_all():
    """Render HTML pages from processed data."""
    import yaml

    project_root = Path.cwd()
    data_processed_dir = project_root / "data" / "processed"
    docs_dir = project_root / "docs"
    cache_dir = project_root / "data" / "cache"
    templates_dir = project_root / "templates"

    # Create directories
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Initialize cache
    cache = Cache(cache_dir)

    # Load index config (for base_url, github_url, google_analytics_id, and SEO)
    index_config_file = project_root / "config" / "index.yaml"
    index_config = None
    base_url = None
    github_url = None
    google_analytics_id = None

    if index_config_file.exists():
        try:
            with open(index_config_file, 'r') as f:
                index_config = yaml.safe_load(f)
                if index_config:
                    base_url = index_config.get('base_url')
                    github_url = index_config.get('github_url')
                    google_analytics_id = index_config.get('google_analytics_id')
        except Exception as e:
            print(f"⚠️  Failed to load config/index.yaml: {e}")

    if not base_url:
        print("⚠️  No base_url in config/index.yaml - canonical URLs will be skipped")

    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(templates_dir))

    print("🎨 Stage 3: Rendering HTML pages\n")

    # Track stats
    rendered_pages = 0
    failed_pages = 0

    # Discover all pages
    page_files = discover_all_pages()
    if not page_files:
        print("❌ No pages found in pages/ directory")
        return

    print(f"📄 Found {len(page_files)} page(s)\n")

    for page_file in page_files:
        # Load page config
        try:
            page_config = load_page_config(page_file)
        except Exception as e:
            print(f"❌ Failed to load {page_file.name}: {e}")
            failed_pages += 1
            continue

        if not page_config.enabled:
            continue

        print(f"📄 Rendering: {page_config.id} ({page_config.name}) [{page_config.category}]")

        # Render each widget
        widget_htmls = []
        for widget_config in page_config.widgets:
            widget_type = widget_config.type

            # Generate cache key
            cache_key = cache.get_cache_key(
                page_config.category,
                page_config.id,
                widget_type,
                widget_config.params
            )

            # Load processed data
            processed_file = data_processed_dir / f"{cache_key}.json"
            if not processed_file.exists():
                print(f"    ⚠️  No processed data for {widget_type}, skipping")
                continue

            try:
                with open(processed_file, 'r') as f:
                    processed_data = json.load(f)
            except Exception as e:
                print(f"    ❌ Failed to read {processed_file.name}: {e}")
                continue

            # Create widget instance
            try:
                widget = create_widget_instance(
                    widget_type=widget_type,
                    params=widget_config.params,
                    page_params=page_config.params,
                    update_minutes=widget_config.update_minutes
                )
            except Exception as e:
                print(f"    ❌ Failed to create widget {widget_type}: {e}")
                continue

            # Render widget HTML
            try:
                widget_html = widget.render(processed_data)
                widget_htmls.append(widget_html)
            except Exception as e:
                print(f"    ❌ Failed to render {widget_type}: {e}")
                continue

        # Render page
        try:
            template = env.get_template("pages/page.html")
            # Use page's theme if defined, otherwise use default theme
            page_theme = page_config.theme if page_config.theme else {}
            page_html = template.render(
                page=page_config,
                theme=page_theme,
                widgets=widget_htmls,
                generated_at=datetime.now(timezone.utc).isoformat(),
                project_name=PROJECT_NAME,
                project_tagline=PROJECT_TAGLINE,
                base_url=base_url,
                github_url=github_url,
                google_analytics_id=google_analytics_id
            )

            # Save page HTML to flat structure: docs/{page_id}.html
            page_output = docs_dir / f"{page_config.id}.html"
            with open(page_output, 'w') as f:
                f.write(page_html)

            print(f"    ✅ Saved to {page_config.id}.html")
            rendered_pages += 1

        except Exception as e:
            print(f"    ❌ Failed to render page: {e}")
            failed_pages += 1

    # Generate index.html
    print("\n📑 Generating index.html...")
    try:
        generate_index(page_files, docs_dir, templates_dir, index_config)
        print("    ✅ Index generated")
    except Exception as e:
        print(f"    ❌ Failed to generate index: {e}")

    # Generate sitemap.xml and robots.txt
    if base_url:
        print("\n🗺️  Generating sitemap.xml and robots.txt...")
        try:
            # Get all enabled pages
            enabled_pages = []
            for page_file in page_files:
                try:
                    page_config = load_page_config(page_file)
                    if page_config.enabled:
                        enabled_pages.append(page_config)
                except Exception:
                    continue

            generate_sitemap(enabled_pages, base_url, docs_dir)
            generate_robots_txt(base_url, docs_dir)
            print("    ✅ sitemap.xml and robots.txt generated")
        except Exception as e:
            print(f"    ❌ Failed to generate sitemap/robots: {e}")
    else:
        print("\n⚠️  Skipping sitemap.xml and robots.txt (no base_url configured)")

    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 Render Summary:")
    print(f"   ✅ Rendered: {rendered_pages} pages")
    print(f"   ❌ Failed: {failed_pages} pages")
    print(f"   📁 Output: {docs_dir}")
    print(f"{'='*60}\n")


def generate_index(page_files: list, docs_dir: Path, templates_dir: Path, index_config: dict = None):
    """Generate index.html that lists all pages grouped by category."""
    from collections import defaultdict

    # Load all page configs and group by category
    pages_by_category = defaultdict(list)

    for page_file in page_files:
        try:
            page_config = load_page_config(page_file)
            if page_config.enabled:
                pages_by_category[page_config.category].append(page_config)
        except Exception as e:
            print(f"    ⚠️  Skipping {page_file.name} in index: {e}")
            continue

    # Sort pages within each category by name
    for category in pages_by_category:
        pages_by_category[category].sort(key=lambda p: p.name)

    # Extract base_url, github_url, google_analytics_id, and description from config
    base_url = None
    github_url = None
    google_analytics_id = None
    index_description = None
    if index_config:
        base_url = index_config.get('base_url')
        github_url = index_config.get('github_url')
        google_analytics_id = index_config.get('google_analytics_id')
        if 'seo' in index_config:
            index_description = index_config['seo'].get('description')

    # Fallback: auto-generate description from pages
    if not index_description:
        all_pages = [p for pages in pages_by_category.values() for p in pages]
        page_names = [p.name for p in all_pages[:5]]  # First 5 pages
        if page_names:
            index_description = f"{PROJECT_TAGLINE} - Monitoring dashboards for {', '.join(page_names)}"
            if len(all_pages) > 5:
                index_description += ", and more"
        else:
            index_description = PROJECT_TAGLINE

    # Set up Jinja2 environment
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template("pages/index.html")

    # Render index
    index_html = template.render(
        pages_by_category=dict(pages_by_category),
        categories=sorted(pages_by_category.keys()),
        generated_at=datetime.now(timezone.utc).isoformat(),
        project_name=PROJECT_NAME,
        project_tagline=PROJECT_TAGLINE,
        index_description=index_description,
        base_url=base_url,
        github_url=github_url,
        google_analytics_id=google_analytics_id
    )

    # Save index.html
    index_output = docs_dir / "index.html"
    with open(index_output, 'w') as f:
        f.write(index_html)


def generate_sitemap(pages: list, base_url: str, docs_dir: Path):
    """Generate sitemap.xml for search engines."""
    sitemap = ['<?xml version="1.0" encoding="UTF-8"?>']
    sitemap.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')

    # Add index page
    sitemap.append('  <url>')
    sitemap.append(f'    <loc>{base_url}/index.html</loc>')
    sitemap.append('    <priority>1.0</priority>')
    sitemap.append('    <changefreq>hourly</changefreq>')
    sitemap.append('  </url>')

    # Add all pages
    for page in pages:
        sitemap.append('  <url>')
        sitemap.append(f'    <loc>{base_url}/{page.id}.html</loc>')
        sitemap.append('    <priority>0.8</priority>')
        sitemap.append('    <changefreq>hourly</changefreq>')
        sitemap.append('  </url>')

    sitemap.append('</urlset>')

    # Save sitemap.xml
    sitemap_output = docs_dir / "sitemap.xml"
    with open(sitemap_output, 'w') as f:
        f.write('\n'.join(sitemap))


def generate_robots_txt(base_url: str, docs_dir: Path):
    """Generate robots.txt to guide search engine crawlers."""
    robots_content = f"""User-agent: *
Allow: /

Sitemap: {base_url}/sitemap.xml
"""

    # Save robots.txt
    robots_output = docs_dir / "robots.txt"
    with open(robots_output, 'w') as f:
        f.write(robots_content)


if __name__ == "__main__":
    render_all()
