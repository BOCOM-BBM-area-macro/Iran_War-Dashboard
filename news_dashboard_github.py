#!/usr/bin/env python3
"""
news_dashboard.py
─────────────────
News Dashboard Pipeline
  1. Reads a YAML config file
  2. Fetches trending articles via GoogleNews (pygooglenews)
  3. Summarises each article with Gemini and Grok
  4. Generates a polished, interactive HTML via Jinja2

Usage:
    python news_dashboard.py --config config.yaml
"""

import argparse
import asyncio
import json
import os
import sys
import textwrap
import time
import webbrowser
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter

# ── optional deps with friendly error messages ────────────────────────────────
try:
    import requests
except ImportError:
    sys.exit("❌  requests not found. Run: pip install requests")

try:
    import yaml
except ImportError:
    sys.exit("❌  PyYAML not found. Run: pip install pyyaml")

try:
    from pygooglenews import GoogleNews
    import feedparser
except ImportError:
    sys.exit("❌  pygooglenews or feedparser not found. Run: pip install pygooglenews feedparser")

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import websockets
except ImportError:
    print("⚠️   websockets not found. AIS tracking will be disabled. Run: pip install websockets")
    websockets = None

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    from markupsafe import Markup
except ImportError:
    sys.exit("❌  Jinja2 not found. Run: pip install jinja2")

try:
    import yfinance as yf
except ImportError:
    sys.exit("❌  yfinance not found. Run: pip install yfinance")

try:
    import pandas as pd
except ImportError:
    sys.exit("❌  pandas not found. Run: pip install pandas")

try:
    from google.cloud import bigquery
except ImportError:
    print("⚠️  google-cloud-bigquery not found. GDELT tracking will be disabled. Run: pip install google-cloud-bigquery")
    bigquery = None

# ── peek-deck integration ──────────────────────────────────────────────────
# Robust path discovery for both GitHub Actions (Linux) and Local Dev (Windows)
search_paths = [
    Path.cwd() / "peek_deck" / "src",            # Repo root (peek_deck)
    Path.cwd() / "peek-deck-1.0.0" / "src",      # Repo root (versioned name)
    Path(__file__).parent / "peek_deck" / "src", # Relative to script
    Path.home() / "Downloads" / "peek-deck-1.0.0" / "src" # Windows Downloads
]

found_path = None
for p in search_paths:
    if p.exists():
        found_path = str(p.resolve())
        if found_path not in sys.path:
            sys.path.append(found_path)
        print(f"✅ peek_deck source found at: {found_path}")
        break

try:
    from peek_deck.core.utils import resolve_google_news_url
    from peek_deck.core.url_metadata import extract_url_metadata
    from peek_deck.core.url_fetch_manager import get_url_fetch_manager
    from bs4 import BeautifulSoup
except ImportError as e:
    if not found_path:
        print("⚠️  peek_deck core modules not found. Check your folder names in the repo.")
    else:
        print(f"❌ Found path but import failed: {e}")

    # Fallback to dummy functions to prevent script crash
    def resolve_google_news_url(url, timeout=10): return url
    def extract_url_metadata(url, **kwargs): 
        class MockMeta:
            def __init__(self): self.description = ""; self.keywords = ""; self.image = None
        return MockMeta()
    def get_url_fetch_manager(): return None


# ── helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    dash = cfg.get("dashboard", {})
    if not dash.get("topic"):
        sys.exit("❌  config.yaml must define dashboard.topic")
    if not dash.get("title"):
        sys.exit("❌  config.yaml must define dashboard.title")

    # Apply defaults
    dash.setdefault("max_articles_recent", 10)
    dash.setdefault("max_articles_older", 10)
    dash.setdefault("language", "en")
    dash.setdefault("country", "US")
    dash.setdefault("period", "7d")

    cfg.setdefault("llm", {})
    cfg["llm"].setdefault("enabled", True)
    
    # API Keys are required if LLM is enabled
    if cfg["llm"]["enabled"]:
        if not cfg["llm"].get("gemini_api_key") or "key_goes_here" in cfg["llm"].get("gemini_api_key", ""):
            print("⚠️  gemini_api_key not found in config. AI features might fail.")
        if not cfg["llm"].get("sentiment_api_key") or "key_goes_here" in cfg["llm"].get("sentiment_api_key", ""):
            print("⚠️  sentiment_api_key not found in config. Sentiment analysis might fail.")
        
    cfg["llm"].setdefault("digest_model", "gemini-1.5-flash")
    cfg["llm"].setdefault("watch_model", "gemini-1.5-flash")
    cfg["llm"].setdefault("sentiment_model", "llama-3.3-70b-versatile")

    cfg["llm"]["summary_focus"] = cfg["llm"].get("summary_focus") or []
    cfg["llm"]["digest_focus"] = cfg["llm"].get("digest_focus") or []

    cfg.setdefault("output", {})
    cfg["output"].setdefault("filename", "dashboard.html")
    cfg["output"].setdefault("theme", "dark")
    cfg["output"].setdefault("show_source_logos", True)
    cfg["output"].setdefault("open_browser", True)

    cfg.setdefault("commodities", {})
    cfg["commodities"].setdefault("enabled", True)
    default_items = [
        {"name": "Crude Oil (WTI)", "symbol": "CL=F"},
        {"name": "Brent Crude Oil", "symbol": "BZ=F"},
        {"name": "Natural Gas", "symbol": "NG=F"},
        {"name": "Gold", "symbol": "GC=F"}
    ]
    cfg["commodities"].setdefault("items", default_items)

    cfg.setdefault("trade_tracker", {})
    cfg["trade_tracker"].setdefault("enabled", True)
    cfg["trade_tracker"].setdefault("chokepoint", "Strait of Hormuz")
    cfg["trade_tracker"].setdefault("lookback_days", 365)

    cfg.setdefault("missile_tracker", {})
    cfg["missile_tracker"].setdefault("enabled", True)
    cfg["missile_tracker"].setdefault("url", "https://xyeshxlnlompwrzzcaqf.supabase.co/rest/v1/attacks?select=*&order=date.asc")

    cfg.setdefault("maritime_tracker", {})
    cfg["maritime_tracker"].setdefault("enabled", True)
    cfg["maritime_tracker"].setdefault("chokepoint", "Strait of Hormuz")
    cfg["maritime_tracker"].setdefault("bounding_box", [[[26.0, 55.5], [27.5, 57.0]]])
    cfg["maritime_tracker"].setdefault("collect_duration", 15)

    cfg.setdefault("gdelt_tracker", {})
    cfg["gdelt_tracker"].setdefault("enabled", True)
    cfg["gdelt_tracker"].setdefault("project_id", None)
    cfg["gdelt_tracker"].setdefault("lookback_days", 7)
    cfg["gdelt_tracker"].setdefault("location_filter", "IR")

    return cfg


def sanitize_for_format(text: str) -> str:
    """Escape curly braces so they don't break string.format()."""
    if not isinstance(text, str):
        return str(text)
    return text.replace("{", "{{").replace("}", "}}")


def try_extract_json(text: str) -> dict:
    """Robustly extract JSON from LLM response, handling markdown fences and chatter."""
    text = text.strip()
    
    # Remove markdown code fences if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
        
    # Try to find the first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    
    if start != -1 and end != -1:
        json_str = text[start : end + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Final fallback to direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return "unknown"


def resolve_url(url: str) -> str:
    """Follow redirects to get the final article URL. 
    Uses peek_deck.core.utils.resolve_google_news_url for robust resolution.
    """
    if "news.google.com" not in url:
        return url

    print(f"    🔗  Resolving Google News redirect URL...")
    try:
        # Use the exact method from google_news.py
        resolved = resolve_google_news_url(url, timeout=10)
        if resolved and resolved != url:
            return resolved
    except Exception as e:
        print(f"    ⚠️  Failed to resolve URL: {e}")

    # Fallback to standard requests resolution if peek_deck fails or returns same URL
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, allow_redirects=True, timeout=10, headers=headers, stream=True)
        return response.url
    except Exception:
        return url


def favicon_url(article_url: str) -> str:
    domain = domain_from_url(article_url)
    return f"https://www.google.com/s2/favicons?sz=32&domain={domain}"


# ── news fetching ─────────────────────────────────────────────────────────────

def _parse_entry(entry: dict) -> dict:
    """Convert a pygooglenews feed entry into a normalised article dict (URLs unresolved)."""
    link = entry.get("link", "")
    source_entries = entry.get("links", [])
    if source_entries:
        link = source_entries[0].get("href", link)

    pub = entry.get("published", "")
    pub_date_iso = ""
    pub_ts = 0
    try:
        pub_dt = datetime(*entry.get("published_parsed", [])[:6], tzinfo=timezone.utc)
        pub_fmt = pub_dt.strftime("%b %d, %Y · %H:%M UTC")
        pub_date_iso = pub_dt.strftime("%Y-%m-%d")
        pub_ts = int(pub_dt.timestamp())
    except Exception:
        pub_fmt = pub
        pub_ts = 0

    source_title = entry.get("source", {}).get("title", domain_from_url(link))

    return {
        "title": entry.get("title", "Untitled"),
        "url": link,
        "source": source_title,
        "domain": domain_from_url(link),
        "favicon": favicon_url(link),
        "published": pub_fmt,
        "pub_date": pub_date_iso,
        "pub_ts": pub_ts,
        "summary": "",
        "tags": [],
        "sentiment": "neutral",
        "_rank": 0,
    }


def _entry_matches_sources(entry: dict, art: dict, allowed_domains: set[str]) -> bool:
    """Return True if an entry's domain (pre- or post-resolution) is in allowed_domains."""
    entry_domain = art["domain"].lower().replace("www.", "")
    if entry_domain in allowed_domains:
        return True
    source_hint = entry.get("source", {}).get("href", "")
    if source_hint:
        hint_domain = domain_from_url(source_hint).lower().replace("www.", "")
        if hint_domain in allowed_domains:
            return True
    return False


def _resolve_and_check(art: dict, allowed_domains: set[str], restrict: bool) -> bool:
    """Resolve the article URL in-place. Returns False if domain check fails after resolution."""
    final_url = resolve_url(art["url"])
    if final_url != art["url"]:
        print(f"    🔗  Resolved redirect: {domain_from_url(final_url)}")
    art["url"] = final_url
    art["domain"] = domain_from_url(final_url)
    art["favicon"] = favicon_url(final_url)

    if restrict:
        resolved_domain = art["domain"].lower().replace("www.", "")
        return resolved_domain in allowed_domains
    return True


def fetch_articles(cfg: dict, period: str = None, max_articles: int = None) -> list[dict]:
    """Fetch articles, optionally restricted to configured source domains."""
    dash = cfg["dashboard"]
    period = period or dash["period"]
    max_articles = max_articles or dash.get("max_articles_recent", 10)
    gn = GoogleNews(lang=dash["language"], country=dash["country"])

    sources = dash.get("sources") or []
    restrict = dash.get("restrict_to_sources", False) and bool(sources)
    allowed_domains = {d.lower().replace("www.", "") for d in sources}

    print(f"🔍  Searching Google News for: '{dash['topic']}' ({period})")
    if restrict:
        print(f"    📋  Filtering to sources: {', '.join(sorted(allowed_domains))}")
    try:
        search = gn.search(dash["topic"], when=period)
    except Exception as exc:
        sys.exit(f"❌  Google News fetch failed: {exc}")

    entries = search.get("entries", [])
    if not entries:
        print(f"⚠️   No articles found for period {period}.")
        return []

    articles: list[dict] = []
    seen_urls: set[str] = set()

    for entry in entries:
        if len(articles) >= max_articles:
            break
        art = _parse_entry(entry)

        if restrict and not _entry_matches_sources(entry, art, allowed_domains):
            continue

        if not _resolve_and_check(art, allowed_domains, restrict):
            continue

        if art["url"] not in seen_urls:
            seen_urls.add(art["url"])
            articles.append(art)

    print(f"✅  Fetched {len(articles)} articles.")
    if restrict and articles:
        source_dist = Counter(art["domain"] for art in articles)
        print(f"    📊  Source distribution: { {k: v for k, v in source_dist.most_common()} }")

    return articles


# ── commodity prices ──────────────────────────────────────────────────────────

def fetch_commodity_prices(cfg: dict) -> list[dict]:
    if not cfg["commodities"].get("enabled", True):
        return []

    period_str = cfg["dashboard"].get("period", "7d")
    fetch_period = "1y"

    commodities_data = []
    print(f"📈  Fetching commodity trajectories for 1y period...")

    for item in cfg["commodities"].get("items", []):
        name = item.get("name")
        symbol = item.get("symbol")
        if not symbol: continue

        print(f"    📊  Loading: {name} ({symbol})…")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=fetch_period)
            
            if df.empty:
                df = ticker.history(period="1mo")
            
            if df.empty:
                print(f"    ⚠️   No data found for {symbol}.")
                continue

            df = df.sort_index()
            
            all_labels = [d.strftime("%Y-%m-%d") for d in df.index]
            all_values = [round(float(p), 2) for p in df['Close']]
            all_timestamps = [int(d.timestamp() * 1000) for d in df.index]

            days_map = {"1d": 1, "7d": 7, "30d": 30}
            lookback_days = days_map.get(period_str, 7)
            
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date()
            period_df = df[df.index.date >= cutoff_date]
            
            if period_df.empty:
                period_df = df.tail(lookback_days) if len(df) >= lookback_days else df
                
            latest_price = float(period_df['Close'].iloc[-1])
            first_price = float(period_df['Close'].iloc[0])
            peak_price = float(period_df['Close'].max())
            change_pct = ((latest_price - first_price) / first_price) * 100

            trend = "stable"
            if change_pct > 2: trend = "upward"
            elif change_pct < -2: trend = "downward"
            
            commodities_data.append({
                "name": name,
                "code": symbol,
                "latest_price": round(latest_price, 2),
                "first_price": round(first_price, 2),
                "peak_price": round(peak_price, 2),
                "change_pct": round(change_pct, 2),
                "trend": trend,
                "history_labels": all_labels,
                "history_values": all_values,
                "history_timestamps": all_timestamps,
                "period_label": period_str
            })
        except Exception as exc:
            print(f"    ⚠️  Failed to fetch {name}: {exc}")

    return commodities_data


def fetch_commodity_intraday(cfg: dict) -> list[dict]:
    """Fetch intraday (2-minute) price data for the last 1 month."""
    if not cfg["commodities"].get("enabled", True):
        return []

    intraday_data = []
    print("⏱️   Fetching intraday 2-minute commodity prices (1-month window)...")

    for item in cfg["commodities"].get("items", []):
        name = item.get("name")
        symbol = item.get("symbol")
        if not symbol:
            continue

        print(f"    ⏰  Intraday: {name} ({symbol})…")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1mo", interval="2m")
            if df.empty:
                df = ticker.history(period="1mo", interval="5m")
            if df.empty:
                df = ticker.history(period="7d", interval="1m")
            
            if df.empty:
                print(f"    ⚠️   No intraday data for {symbol}.")
                continue

            df = df.sort_index()
            labels = [d.strftime("%b %d %H:%M") for d in df.index]
            timestamps = [int(d.timestamp() * 1000) for d in df.index]
            values = [round(float(p), 2) for p in df["Close"]]

            base = values[0] if values[0] != 0 else 1
            pct_values = [round((v - base) / base * 100, 4) for v in values]

            latest = values[-1]
            change_pct = round((latest - values[0]) / values[0] * 100, 2)

            intraday_data.append({
                "name": name,
                "code": symbol,
                "latest_price": latest,
                "change_pct": change_pct,
                "labels": labels,
                "timestamps": timestamps,
                "raw_values": values,
                "pct_values": pct_values,
            })
        except Exception as exc:
            print(f"    ⚠️  Failed intraday fetch for {name}: {exc}")

    return intraday_data


def fetch_trade_tracker_data(cfg: dict) -> list[dict]:
    """Fetch trade data from IMF PortWatch (ArcGIS Service)."""
    if not cfg.get("trade_tracker", {}).get("enabled", True):
        return []

    chokepoint = cfg["trade_tracker"].get("chokepoint", "Strait of Hormuz")
    lookback = cfg["trade_tracker"].get("lookback_days", 365)
    
    print(f"🚢  Fetching IMF PortWatch data for {chokepoint}...")
    
    base_url = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Chokepoints_Data/FeatureServer/0/query"
    cutoff_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")
    
    params = {
        "where": f"portname = '{chokepoint}' AND date >= date '{cutoff_date}'",
        "outFields": "date,n_tanker,n_container,n_dry_bulk,n_general_cargo,n_roro",
        "orderByFields": "date ASC",
        "resultRecordCount": 2000,
        "f": "json"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=20)
        data = response.json()
        
        features = data.get("features", [])
        if not features:
            print(f"    ⚠️  No trade data found for {chokepoint}.")
            return []
            
        trade_history = []
        for f in features:
            attrs = f["attributes"]
            dt = datetime.fromtimestamp(attrs["date"] / 1000, tz=timezone.utc)
            trade_history.append({
                "date": dt.strftime("%Y-%m-%d"),
                "tanker": attrs.get("n_tanker", 0),
                "container": attrs.get("n_container", 0),
                "dry_bulk": attrs.get("n_dry_bulk", 0),
                "general_cargo": attrs.get("n_general_cargo", 0),
                "roro": attrs.get("n_roro", 0),
            })
            
        print(f"    ✅  Loaded {len(trade_history)} days of trade volume data.")
        return trade_history
    except Exception as exc:
        print(f"    ⚠️  Failed to fetch trade tracker data: {exc}")
        return []


def fetch_missile_tracker_data(cfg: dict) -> list[dict]:
    """Fetch daily attack data directly from the official tracker site via Supabase REST API."""
    tracker_cfg = cfg.get("missile_tracker", {})
    if not tracker_cfg.get("enabled", True):
        return []

    print("🚀  Fetching missile data from official site (Supabase API)...")
    # Default to the Supabase URL provided by the user
    url = tracker_cfg.get("url", "https://xyeshxlnlompwrzzcaqf.supabase.co/rest/v1/attacks?select=*&order=date.asc")
    
    # Public anon key found from uaedefensemonitor.com
    headers = {
        "apikey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inh5ZXNoeGxubG9tcHdyenpjYXFmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzM1OTczNDcsImV4cCI6MjA4OTE3MzM0N30.DS2shY3x9vFDel3iyUd6HRdxTiYTYsPegQM9xhicf2Y"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if not response.ok:
            print(f"    ❌  Missile tracker fetch failed [{response.status_code}]: {response.text[:100]}")
            return []
            
        data = response.json()
        
        if not data:
            print("    ⚠️  No data found in the missile tracker response.")
            return []
            
        print(f"    ✅  Successfully retrieved {len(data)} records from missile tracker.")
        
        normalized_data = []
        for row in data:
            # The Supabase response has ballistic, cruise, uav fields
            ballistic = row.get("ballistic", 0)
            cruise = row.get("cruise", 0)
            uav = row.get("uav", 0)
            
            # Calculate total if not provided
            total = row.get("total", row.get("total_attacks", ballistic + cruise + uav))
            
            # Use created_at to help with chronological sorting
            created_at = row.get("created_at", "")
            date_str = row.get("date", "Unknown Date")
            
            # Try to create a robust sort key (YYYY-MM-DD)
            sort_key = created_at
            try:
                # Parse "Mar 01" - assume year from created_at or current year
                year = created_at[:4] if len(created_at) >= 4 else str(datetime.now().year)
                # datetime.strptime %b handles short month names like "Mar"
                dt = datetime.strptime(f"{year} {date_str}", "%Y %b %d")
                sort_key = dt.strftime("%Y-%m-%d")
            except Exception:
                pass
            
            normalized_data.append({
                "date": date_str,
                "ballistic_missiles": ballistic,
                "cruise_missiles": cruise,
                "drones": uav,
                "israel_munitions": row.get("israel_munitions", 0),
                "us_munitions": row.get("us_munitions", 0),
                "total_iranian": total,
                "summary": row.get("summary", row.get("description", "")),
                "_sort_key": sort_key
            })
            
        # Sort chronologically using the _sort_key
        return sorted(normalized_data, key=lambda x: x["_sort_key"])

    except Exception as exc:
        print(f"    ⚠️  Failed to fetch data from missile tracker: {exc}")
        return []


async def _collect_ais_messages(api_key: str, bounding_boxes: list, duration: int) -> list[dict]:
    """Internal async function to connect to aisstream and collect messages."""
    url = "wss://stream.aisstream.io/v0/stream"
    ships = {}
    message_count = 0
    
    print(f"    🔌  Connecting to {url}...")
    try:
        async with websockets.connect(url) as websocket:
            subscribe_msg = {
                "APIKey": api_key,
                "BoundingBoxes": bounding_boxes,
                "FilterMessageTypes": ["PositionReport"]
            }

            print(f"    📡  Sending subscription for BBox: {bounding_boxes}")
            await websocket.send(json.dumps(subscribe_msg))
            
            start_time = time.time()
            print(f"    ⏳  Listening for messages (Duration: {duration}s)...")
            
            while time.time() - start_time < duration:
                try:
                    # Set a timeout for recv to check loop condition
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    message_count += 1
                    msg_json = json.loads(message)
                    
                    if message_count % 10 == 0:
                        print(f"    📥  Received {message_count} messages so far...")

                    metadata = msg_json.get("MetaData", {})
                    mmsi = metadata.get("MMSI")
                    if mmsi:
                        # Keep only latest position for each ship
                        ships[mmsi] = {
                            "name": metadata.get("ShipName", "Unknown").strip(),
                            "mmsi": mmsi,
                            "lat": metadata.get("latitude"),
                            "lon": metadata.get("longitude"),
                            "type": metadata.get("ShipType", "Unknown"),
                            "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
                        }
                except asyncio.TimeoutError:
                    # This is normal if no message arrives in 1s
                    continue
                except Exception as e:
                    print(f"    ⚠️   Error processing message: {e}")
                    break
            
            print(f"    🏁  Finished collecting. Total raw messages: {message_count}")
    except Exception as e:
        print(f"    ❌  AISStream Connection Error: {e}")
        
    return list(ships.values())


def fetch_ais_data(cfg: dict) -> list[dict]:
    """Fetch real-time ship positions in the chokepoint using aisstream.io."""
    m_cfg = cfg.get("maritime_tracker", {})
    if not m_cfg.get("enabled", True) or websockets is None:
        return []

    api_key = m_cfg.get("api_key")
    if not api_key or api_key == "YOUR_AISSTREAM_API_KEY":
        print("    ⚠️  AISStream API Key missing or not set in config.yaml. Skipping AIS data.")
        return []

    chokepoint = m_cfg.get("chokepoint", "Strait of Hormuz")
    duration = m_cfg.get("collect_duration", 15)
    bbox = m_cfg.get("bounding_box", [[[26.0, 55.5], [27.5, 57.0]]])

    print(f"⛴️   Connecting to AISStream for {chokepoint} ({duration}s snapshot)...")
    
    try:
        # Run the async collector
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ais_data = loop.run_until_complete(_collect_ais_messages(api_key, bbox, duration))
        loop.close()
        
        print(f"    ✅  Captured {len(ais_data)} unique ships in the area.")
        return ais_data
    except Exception as exc:
        print(f"    ⚠️   Failed to fetch AIS data: {exc}")
        return []


def fetch_gdelt_data(cfg: dict) -> dict:
    """Fetch conflict and infrastructure-related events from GDELT.
    Falls back to GDELT DOC API if BigQuery is unavailable or not configured.
    """
    g_cfg = cfg.get("gdelt_tracker", {})
    if not g_cfg.get("enabled", True):
        return {"events": [], "error": None}
    
    project_id = g_cfg.get("project_id")
    use_bigquery = (bigquery is not None and 
                    project_id and 
                    project_id != "your-google-cloud-project-id")

    if use_bigquery:
        print(f"🌍  Attempting BigQuery fetch for project: {project_id}...")
        # [Existing BigQuery Logic...]
        lookback = g_cfg.get("lookback_days", 7)
        location = g_cfg.get("location_filter", "IR")
        start_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y%m%d")
        
        query = f"""
        SELECT SQLDATE, EventCode, Actor1Name, Actor2Name, NumArticles, SOURCEURL, ActionGeo_FullName
        FROM `gdelt-intl.gdeltv2.events`
        WHERE (ActionGeo_CountryCode = '{location}' OR Actor1CountryCode = '{location}')
        AND EventRootCode IN ('19', '20') AND SQLDATE >= {start_date}
        ORDER BY SQLDATE DESC LIMIT 100
        """
        try:
            client = bigquery.Client(project=project_id)
            results = client.query(query).result()
            events = []
            for row in results:
                events.append({
                    "date": str(row.SQLDATE),
                    "event_type": f"Conflict ({row.EventCode})",
                    "actor1": row.Actor1Name or "Unknown",
                    "actor2": row.Actor2Name or "Unknown",
                    "location": row.ActionGeo_FullName,
                    "url": row.SOURCEURL,
                    "source": domain_from_url(row.SOURCEURL)
                })
            return {"events": events, "error": None}
        except Exception as e:
            print(f"    ⚠️  BigQuery failed, falling back to GDELT Web API: {e}")

    # --- FALLBACK: GDELT DOC API (No Google Cloud Required) ---
    print("🌍  Fetching from GDELT Web API (Public)...")
    location = g_cfg.get("location_filter", "IR")
    
    query_str = f'location:{location} (theme:WB_831_INFRASTRUCTURE_DESTRUCTION OR theme:CONV_MILITARY_FORCE OR "missile strike" OR "airstrike")'
    
    api_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query_str,
        "mode": "artlist",
        "maxrecords": 50,
        "format": "json",
        "sort": "date"
    }
    
    for attempt in range(3):
        try:
            response = requests.get(api_url, params=params, timeout=15)
            
            if response.status_code == 429:
                wait_time = (attempt + 1) * 5
                print(f"    ⚠️  GDELT API Rate Limited (429). Retrying in {wait_time}s... (Attempt {attempt+1}/3)")
                time.sleep(wait_time)
                continue

            if not response.ok:
                return {"events": [], "error": f"GDELT Web API error: {response.status_code} - {response.text[:100]}"}
            
            if not response.text.strip():
                return {"events": [], "error": "GDELT Web API returned an empty response."}
                
            try:
                data = response.json()
            except Exception:
                # If it's not JSON, it might be an HTML error page
                if "<html" in response.text.lower():
                    return {"events": [], "error": "GDELT Web API returned an HTML page instead of data (likely a temporary block or server error)."}
                return {"events": [], "error": f"GDELT Web API returned invalid JSON: {response.text[:100]}..."}

            articles = data.get("articles", [])
            
            events = []
            for art in articles:
                raw_date = art.get("seendate", "")
                clean_date = raw_date[:8] if len(raw_date) >= 8 else datetime.now().strftime("%Y%m%d")
                
                events.append({
                    "date": clean_date,
                    "event_type": art.get("title", "Infrastructure Event"),
                    "actor1": art.get("sourcecountry", "Multiple"),
                    "actor2": "Infrastructure",
                    "location": location,
                    "url": art.get("url", "#"),
                    "source": art.get("source", "GDELT")
                })
                
            print(f"    ✅  Successfully retrieved {len(events)} events via Web API.")
            return {"events": events, "error": None if events else "No recent infrastructure events found."}
            
        except Exception as exc:
            if attempt == 2:
                return {"events": [], "error": f"GDELT Web API failed after retries: {exc}"}
            time.sleep(2)
    
    return {"events": [], "error": "GDELT Web API failed: Maximum retries reached (Rate Limited)."}


# ── LLM operations ───────────────────────────────────────────────────────────

ARTICLE_PROMPT_TEMPLATE = """\
You are an expert analyst summarising a news article for a professional dashboard.

Article title: {title}
Source: {source}
URL: {url}

{focus_instructions}

Respond ONLY with a valid JSON object (no markdown fences) in this exact schema: 
{{
  "summary": "<2–3 sentence summary in clear, direct analyst language>",
  "tags": ["<tag1>", "<tag2>", "<tag3>"],
  "sentiment": "<positive|negative|neutral|mixed>"
}}
"""

EXECUTIVE_DIGEST_PROMPT_TEMPLATE = """\
You are an Economist writing an executive summary for a professional intelligence dashboard.

Topic: {topic}
Number of articles: {count}

Here are the article titles and summaries:
{articles_text}

Write a comprehensive executive digest that:
- Identifies the most relevant events of the current and previous day
- Is comprehensive yet objective, providing a thorough analysis with 8-10 lines
- DO NOT cut the analysis short; ensure it is a complete, well-rounded executive summary that ends with a full sentence
- Avoid qualitative analysis, focus solely on exposing information
- PLEASE don't assume things or create subjective texts, focus on concrete facts and events
- Also, PLEASE don't cite the font for every statement, keep the flow natural
- DON'T CITE THE ARTICLES OR FONTS FROM EACH STATEMENT IN PARENTHESIS 
- PLEASE respect the paragraph size (8-12 lines)
{digest_instructions}

Respond ONLY with a valid JSON object (no markdown fences).
Ensure the "digest" field is the LAST field in the JSON and contains the full analysis.

{{
"top_themes": ["<theme1>", "<theme2>", "<theme3>"],
"digest": "<your full, complete, non-truncated executive summary>"
}}
"""

THINGS_TO_WATCH_PROMPT_TEMPLATE = """\
You are a Strategic Analyst identifying future risks and events for a professional intelligence dashboard.

Topic: {topic}

Here are the recent article titles and summaries:
{articles_text}

Based on these news, highlight what analysts should watch next in a "THINGS TO WATCH" section.

For each item, provide:
1. A clear, impactful title
2. A 2-3 sentence description explaining why this is important and what to look for.

Don't keep any relevant events away!

Respond ONLY with a valid JSON object (no markdown fences):
{{
"next_events": [
  {{
    "title": "<event title>",
    "description": "<detailed description>"
  }}
]
}}
"""

SENTIMENT_PROMPT_TEMPLATE = """\
You are an expert news analyst. Categorize the sentiment of the following news article as positive, negative, or neutral based on its title and description.

Article Title: {title}
Article Description: {description}

Respond ONLY with a valid JSON object (no markdown fences) in this exact schema:
{{
  "sentiment": "<positive|negative|neutral>"
}}
"""

def summarise_articles(articles: list[dict], cfg: dict) -> list[dict]:
    """Extract summaries and full text for articles using peek_deck core metadata extractor."""
    fetcher = get_url_fetch_manager()
    for i, art in enumerate(articles, 1):
        print(f"  🤖  Extracting content [{i}/{len(articles)}] {art['title'][:60]}…")
        
        try:
            print(f"    📸  Extracting metadata: {art['url'][:50]}...")
            metadata = extract_url_metadata(art['url'])
            
            if metadata:
                art["summary"] = metadata.description or ""
                if metadata.keywords:
                    raw_tags = [t.strip() for t in metadata.keywords.split(',')]
                    art["tags"] = [t for t in raw_tags if t and len(t) < 30][:5]
                else:
                    art["tags"] = ["extracted", art["domain"]]
            
            # Full text extraction (as requested)
            if fetcher:
                try:
                    print(f"    📄  Extracting full text...")
                    html = fetcher.get(art['url'], response_type="text", timeout=10)
                    soup = BeautifulSoup(html, "html.parser")
                    
                    # Clean up HTML
                    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                        element.decompose()
                    
                    # Extract text
                    lines = (line.strip() for line in soup.get_text(separator="\n").splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    full_text = "\n".join(chunk for chunk in chunks if chunk)
                    
                    # Store first 10k characters to avoid blowing up context too much
                    art["full_text"] = full_text[:10000]
                    print(f"    ✅  Extracted {len(art['full_text'])} chars of full text")
                except Exception as e:
                    print(f"    ⚠️   Full text extraction failed: {e}")
                    art["full_text"] = art["summary"]
            else:
                art["full_text"] = art["summary"]

            art["sentiment"] = "neutral"
                
        except Exception as exc:
            print(f"    ⚠️   Failed to extract metadata: {exc}")
            art["summary"] = ""
            art["full_text"] = ""
            art["sentiment"] = "neutral"

    return articles


def categorize_sentiment(articles: list[dict], cfg: dict) -> list[dict]:
    """Categorize sentiment of news articles using a configurable OpenAI-compatible API (Groq/xAI)."""
    if not OpenAI:
        print("⚠️  OpenAI library not found. Sentiment analysis skipped.")
        return articles
        
    api_key = cfg["llm"].get("sentiment_api_key")
    base_url = cfg["llm"].get("api_base_url", "https://api.groq.com/openai/v1")
    
    if not api_key or "key_goes_here" in api_key:
        print(f"⚠️  Sentiment API key missing. Sentiment analysis skipped.")
        return articles
        
    model = cfg["llm"].get("sentiment_model", "llama-3.3-70b-versatile")
    print(f"\n🧠  Categorizing article sentiments with {model} via {base_url}...")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    for i, art in enumerate(articles, 1):
        description = art.get("summary", "") or "No description available."
        
        print(f"    🏷️  Sentiment [{i}/{len(articles)}] {art['title'][:60]}…")
        
        prompt = SENTIMENT_PROMPT_TEMPLATE.format(
            title=art["title"],
            description=description
        )
        
        messages = [
            {"role": "system", "content": "You are a sentiment analyst. Output JSON only."},
            {"role": "user", "content": prompt}
        ]
        
        art["sentiment"] = "neutral"
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            data = try_extract_json(content)
            if data and "sentiment" in data:
                sentiment = data["sentiment"].lower().strip()
                if any(s in sentiment for s in ["positive", "negative", "neutral"]):
                    if "positive" in sentiment: art["sentiment"] = "positive"
                    elif "negative" in sentiment: art["sentiment"] = "negative"
                    else: art["sentiment"] = "neutral"
        except Exception as exc:
            print(f"    ⚠️   Sentiment categorization failed for '{art['title'][:30]}': {exc}")
            
    return articles


def generate_digest(articles: list[dict], commodities: list[dict], cfg: dict) -> dict:
    """Generate daily digest using Gemini with full text support."""
    if not genai:
        print("⚠️  Google Generative AI library not found. Digest skipped.")
        return {"digest": "Digest unavailable.", "top_themes": [], "next_events": []}
        
    api_key = cfg["llm"].get("gemini_api_key")
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("⚠️  Gemini API key missing. Digest skipped.")
        return {"digest": "Digest unavailable.", "top_themes": [], "next_events": []}
        
    genai.configure(api_key=api_key)
    
    digest_model_name = cfg["llm"].get("digest_model", "gemini-1.5-flash")
    watch_model_name = cfg["llm"].get("watch_model", "gemini-1.5-flash")
    
    print(f"\n🧠  Initializing models: Digest ({digest_model_name}), Watch ({watch_model_name})...")
    
    # Select Gemini models
    try:
        digest_model = genai.GenerativeModel(digest_model_name)
        watch_model = genai.GenerativeModel(watch_model_name)
    except Exception as e:
        print(f"⚠️  Failed to initialize Gemini models: {e}")
        return {"digest": "Digest unavailable.", "top_themes": [], "next_events": []}

    # Format articles with full text if available
    articles_text = "\n\n".join(
        f"{i}. [{sanitize_for_format(a['source'])}] {sanitize_for_format(a['title'])}\n   "
        f"EXTRACTED CONTENT: {sanitize_for_format(a.get('full_text', a['summary']))}"
        for i, a in enumerate(articles, 1)
    )

    digest_instructions = ""
    if cfg["llm"]["digest_focus"]:
        items = "\n".join(f"  - {sanitize_for_format(d)}" for d in cfg["llm"]["digest_focus"])
        digest_instructions = f"Additionally address:\n{items}"

    # --- Step 1: Executive Digest ---
    print(f"  🧠  Generating executive digest with {digest_model_name}…")
    digest_prompt = (
        EXECUTIVE_DIGEST_PROMPT_TEMPLATE
        .replace("{topic}", cfg["dashboard"]["topic"])
        .replace("{count}", str(len(articles)))
        .replace("{articles_text}", articles_text)
        .replace("{digest_instructions}", digest_instructions)
        .replace("{{", "{").replace("}}", "}")
    )

    digest_data = {"digest": "Digest unavailable.", "top_themes": []}
    for attempt in range(3):
        try:
            response = digest_model.generate_content(
                digest_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            data = try_extract_json(response.text)
            if data and "digest" in data:
                digest_data = data
                break
        except Exception as exc:
            if attempt == 2: print(f"    ⚠️   Executive digest failed: {exc}")
            time.sleep(1)

    # --- Step 2: Things to Watch ---
    print(f"  🔭  Identifying things to watch with {watch_model_name}…")
    watch_prompt = (
        THINGS_TO_WATCH_PROMPT_TEMPLATE
        .replace("{topic}", cfg["dashboard"]["topic"])
        .replace("{articles_text}", articles_text)
        .replace("{{", "{").replace("}}", "}")
    )

    watch_data = {"next_events": []}
    for attempt in range(3):
        try:
            response = watch_model.generate_content(
                watch_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            data = try_extract_json(response.text)
            if data and "next_events" in data:
                watch_data = data
                break
        except Exception as exc:
            if attempt == 2: print(f"    ⚠️   Things to watch failed: {exc}")
            time.sleep(1)

    return {
        "digest": digest_data.get("digest", "Digest unavailable."),
        "top_themes": digest_data.get("top_themes", []),
        "next_events": watch_data.get("next_events", [])
    }


# ── HTML rendering ─────────────────────────────────────────────────────────────

# ── HTML & JS rendering ──────────────────────────────────────────────────────



def render_html(articles: list[dict], commodities: list[dict], intraday_commodities: list[dict], trade_data: list[dict], missile_data: list[dict], ais_data: list[dict], gdelt_data: list[dict], digest: dict, cfg: dict) -> str:
    env = Environment(autoescape=True)
    
    def _safe_tojson(d):
        serialized = json.dumps(d).replace('</', '<\\/')
        return Markup(serialized)
    env.filters['tojson'] = _safe_tojson
        
    template_content = """
<!DOCTYPE html>
<html lang="en" data-theme="{{ theme }}">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{{ title }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet" />
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<style>
/* ── Reset & tokens ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:         #0a0c10;
  --surface:    #111620;
  --surface2:   #1a2030;
  --border:     #1e2a3a;
  --accent:     #00e5ff;
  --accent2:    #7c3aed;
  --text:       #c8d8e8;
  --muted:      #5a7090;
  --positive:   #22c55e;
  --negative:   #ef4444;
  --neutral:    #94a3b8;
  --mixed:      #f59e0b;
  --radius:     6px;
  --font-head:  'Inter', sans-serif;
  --font-mono:  'IBM Plex Mono', monospace;
}

[data-theme="light"] {
  --bg:       #f0f4f8; --surface:  #ffffff; --surface2: #e8eef6; --border:   #cdd5e0; --text:     #1e2a3a; --muted:    #7a90a8;
}

body { background: var(--bg); color: var(--text); font-family: var(--font-mono); font-size: 13px; line-height: 1.6; min-height: 100vh; }
.container { max-width: 1280px; margin: 0 auto; padding: 24px; position: relative; z-index: 1; }

/* ── Header ── */
header { border-bottom: 1px solid var(--border); padding-bottom: 20px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }
.header-left { display: flex; flex-direction: column; gap: 4px; }
h1 { font-family: var(--font-head); font-size: 28px; font-weight: 800; color: #fff; }
.topic-badge { background: var(--surface2); border: 1px solid var(--border); border-radius: 20px; padding: 4px 12px; font-size: 11px; color: var(--accent); }
.header-right { display: flex; gap: 20px; text-align: right; }
.stat-num { font-family: var(--font-head); font-size: 24px; font-weight: 800; color: var(--accent); }
.stat-label { font-size: 10px; color: var(--muted); text-transform: uppercase; }

/* ── Tabs ── */
.tab-container { display: flex; gap: 10px; margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 12px; }
.tab-btn {
  background: transparent; border: none; color: var(--muted); font-family: var(--font-head);
  font-size: 16px; font-weight: 700; cursor: pointer; padding: 8px 16px; border-radius: var(--radius);
  transition: all 0.2s;
}
.tab-btn:hover { color: var(--accent); }
.tab-btn.active { background: var(--surface2); color: var(--accent); }
.tab-content { display: none; }
.tab-content.active { display: block; }

/* ── Controls ── */
.controls { display: flex; gap: 8px; margin-bottom: 24px; flex-wrap: wrap; align-items: center; }
.filter-btn {
  background: var(--surface); border: 1px solid var(--border); color: var(--muted); border-radius: var(--radius);
  padding: 6px 14px; font-size: 11px; cursor: pointer; transition: all 0.2s;
}
.filter-btn:hover, .filter-btn.active { background: var(--accent); border-color: var(--accent); color: #000; font-weight: 600; }
.search-wrap { margin-left: auto; }
.search-wrap input {
  background: var(--surface); border: 1px solid var(--border); color: var(--text); border-radius: var(--radius);
  padding: 6px 12px; font-size: 12px; outline: none; width: 220px;
}

/* ── Layout ── */
.main-grid { display: grid; grid-template-columns: 1fr 340px; gap: 24px; }

/* ── Components ── */
.digest-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; margin-bottom: 24px; border-left: 4px solid var(--accent2); }
.digest-card h2 { font-family: var(--font-head); font-size: 11px; text-transform: uppercase; color: var(--accent2); margin-bottom: 12px; letter-spacing: 0.1em; }
.digest-text { font-size: 14px; line-height: 1.7; }

.commodities-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 24px; }
.commodity-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; display: flex; flex-direction: column; gap: 12px; }
.comm-header { display: flex; justify-content: space-between; align-items: flex-start; }
.comm-name { font-size: 10px; text-transform: uppercase; color: var(--muted); }
.comm-price { font-family: var(--font-head); font-size: 22px; font-weight: 800; color: #fff; }
.comm-change { font-weight: 600; font-size: 12px; }
.comm-change.up { color: var(--positive); }
.comm-change.down { color: var(--negative); }
.chart-container { height: 180px; width: 100%; position: relative; margin-top: 10px; }
.chart-controls { display: flex; gap: 4px; margin-top: 10px; justify-content: center; }
.chart-btn { 
  background: var(--surface2); border: 1px solid var(--border); color: var(--muted); 
  border-radius: 4px; padding: 2px 8px; font-size: 9px; cursor: pointer; transition: 0.2s;
}
.chart-btn:hover, .chart-btn.active { background: var(--accent); border-color: var(--accent); color: #000; font-weight: 600; }

.article-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; margin-bottom: 12px; transition: 0.2s; }
.article-card:hover { border-color: var(--accent); transform: translateX(4px); }
.article-card.hidden { display: none; }
.card-head { display: flex; gap: 12px; margin-bottom: 12px; }
.favicon { width: 18px; height: 18px; border-radius: 4px; }
.card-title { font-family: var(--font-head); font-size: 15px; font-weight: 600; color: #fff; line-height: 1.4; }
.card-title a { color: inherit; text-decoration: none; }
.card-meta { font-size: 10px; color: var(--muted); display: flex; gap: 10px; align-items: center; margin-top: 4px; }
.card-source { color: var(--accent); }
.sentiment-dot { width: 8px; height: 8px; border-radius: 50%; }
.sentiment-dot.positive { background: var(--positive); }
.sentiment-dot.negative { background: var(--negative); }
.sentiment-dot.neutral  { background: var(--neutral); }
.card-summary { font-size: 12px; opacity: 0.85; margin: 12px 0; }
.card-tags { display: flex; flex-wrap: wrap; gap: 6px; }
.tag { background: var(--surface2); border: 1px solid var(--border); border-radius: 4px; padding: 2px 8px; font-size: 10px; color: var(--muted); cursor: pointer; }
.tag:hover { color: var(--accent); border-color: var(--accent); }

/* ── Sidebar ── */
.widget { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; margin-bottom: 16px; }
.widget-title { font-size: 10px; text-transform: uppercase; color: var(--muted); margin-bottom: 15px; border-bottom: 1px solid var(--border); padding-bottom: 8px; letter-spacing: 0.1em; }
.sent-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.sent-label { width: 60px; font-size: 11px; text-transform: capitalize; }
.sent-bar-wrap { flex: 1; height: 6px; background: var(--surface2); border-radius: 3px; overflow: hidden; }
.sent-bar { height: 100%; border-radius: 3px; }
.sent-bar.positive { background: var(--positive); }
.sent-bar.negative { background: var(--negative); }
.sent-bar.neutral  { background: var(--neutral); }

.event-card { background: var(--surface2); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; margin-bottom: 12px; display: flex; gap: 16px; align-items: flex-start; }
.event-num { width: 32px; height: 32px; background: rgba(255,255,255,0.05); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; color: var(--muted); flex-shrink: 0; font-size: 14px; }
.event-content { flex: 1; }
.event-title { font-family: var(--font-head); font-size: 16px; font-weight: 700; color: #fff; line-height: 1.3; margin-bottom: 10px; }
.event-tags { display: flex; gap: 8px; margin-bottom: 12px; }
.event-desc { font-size: 12px; color: var(--muted); line-height: 1.5; }

footer { margin-top: 48px; border-top: 1px solid var(--border); padding-top: 20px; font-size: 11px; color: var(--muted); text-align: center; }

/* ── Hourly section ── */
.hourly-section { margin-bottom: 24px; }
.section-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.section-title { font-size: 10px; text-transform: uppercase; color: var(--muted); letter-spacing: 0.12em; border-bottom: 1px solid var(--border); padding-bottom: 8px; flex: 1; }
.hourly-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px; }
.hourly-legend { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 14px; }
.legend-item { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--text); }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.legend-change { font-size: 10px; margin-left: 2px; font-weight: 600; }
.legend-change.up { color: var(--positive); }
.legend-change.down { color: var(--negative); }
.hourly-chart-wrap { height: 260px; position: relative; }
.hourly-note { font-size: 10px; color: var(--muted); margin-top: 10px; text-align: right; }

/* ── Mobile Responsiveness ── */
@media (max-width: 900px) {
  .main-grid { grid-template-columns: 1fr; }
  .header-right { display: none; }
}

@media (max-width: 600px) {
  .container { padding: 12px; }
  header { flex-direction: column; align-items: flex-start; gap: 12px; border-bottom: none; margin-bottom: 16px; }
  h1 { font-size: 22px; }
  
  .tab-container { 
    display: flex; 
    overflow-x: auto; 
    gap: 8px; 
    padding-bottom: 8px; 
    margin-bottom: 16px;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
  }
  .tab-container::-webkit-scrollbar { display: none; }
  .tab-btn { padding: 8px 12px; font-size: 13px; white-space: nowrap; background: var(--surface); }
  
  .controls { gap: 12px; }
  .search-wrap { width: 100%; order: -1; }
  .search-wrap input { width: 100%; font-size: 14px; padding: 10px; }
  
  .digest-card { padding: 16px; margin-bottom: 16px; }
  .digest-text { font-size: 13px; }
  
  .article-card { padding: 16px; }
  .card-title { font-size: 14px; }
  
  .commodities-grid { grid-template-columns: 1fr; }
  
  .hourly-chart-wrap { height: 200px; }
  .chart-controls { overflow-x: auto; justify-content: flex-start; padding-bottom: 4px; }
  .chart-btn { padding: 4px 10px; }
  
  .widget { padding: 16px; }
  .hourly-chart-wrap { height: 300px !important; }
}
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="header-left">
      <h1>{{ title }}</h1>
      <div style="display: flex; align-items: center; gap: 12px; margin-top: 4px;">
        <span class="topic-badge">⌖ {{ topic }}</span>
        <span style="font-size: 11px; color: var(--muted); background: rgba(255,255,255,0.03); padding: 4px 10px; border-radius: 4px; border: 1px solid var(--border);">
          ✨ AI-Generated Summaries · {{ generated_at }}
        </span>
      </div>
    </div>
  </header>

  <div class="tab-container">
    <button class="tab-btn active" onclick="switchTab('news', this)">📰 News Feed</button>
    <button class="tab-btn" onclick="switchTab('markets', this)">📊 Market Analysis</button>
    {% if cfg.trade_tracker.enabled and trade_data %}
    <button class="tab-btn" onclick="switchTab('trade', this)">🚢 Trade Tracker</button>
    {% endif %}
    {% if cfg.maritime_tracker.enabled %}
    <button class="tab-btn" onclick="switchTab('maritime', this)">⛴️ Maritime Movement</button>
    {% endif %}
    {% if cfg.missile_tracker.enabled %}
    <button class="tab-btn" onclick="switchTab('missile', this)">🚀 Missile Tracker</button>
    {% endif %}
    {% if cfg.gdelt_tracker.enabled %}
    <button class="tab-btn" onclick="switchTab('gdelt', this)">🌍 GDELT Events</button>
    {% endif %}
  </div>

  <div id="newsTab" class="tab-content active">
    <div class="controls">
      <div class="search-wrap" style="margin-left: 0;">
        <input type="text" id="searchInput" placeholder="Search keywords…" oninput="applyFilters()" />
      </div>
      
      <div style="margin-left: 20px; display: flex; align-items: center; gap: 8px;" class="hide-mobile">
        <label for="newsLimit" style="font-size: 11px; color: var(--muted); text-transform: uppercase;">Show:</label>
        <select id="newsLimit" onchange="applyFilters()" style="background: var(--surface); border: 1px solid var(--border); color: var(--text); border-radius: var(--radius); padding: 4px; font-size: 11px;">
          <option value="5">5</option>
          <option value="10">10</option>
          <option value="20">20</option>
          <option value="all" selected>All</option>
        </select>
      </div>

      <div style="margin-left: 20px; display: flex; align-items: center; gap: 8px;">
        <label for="sortOrder" style="font-size: 11px; color: var(--muted); text-transform: uppercase;">Sort:</label>
        <select id="sortOrder" onchange="applySort()" style="background: var(--surface); border: 1px solid var(--border); color: var(--text); border-radius: var(--radius); padding: 4px; font-size: 11px;">
          <option value="relevance" selected>Most Relevant</option>
          <option value="desc">Newest First</option>
          <option value="asc">Oldest First</option>
        </select>
      </div>

      <div style="width: 100%; margin-top: 16px; border-top: 1px solid var(--border); padding-top: 16px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
        <span style="font-size: 11px; color: var(--muted); text-transform: uppercase;">Filter Dates:</span>
        <button class="filter-btn active" id="date-all" onclick="toggleDate('all')">All Dates</button>
        {% for date in unique_dates %}
        <button class="filter-btn" id="date-{{ date.iso }}" onclick="toggleDate('{{ date.iso }}')">{{ date.label }}</button>
        {% endfor %}
      </div>

      <div style="margin-left: auto; display: flex; gap: 8px;">
        <button class="filter-btn active" onclick="filterSentiment('all', this)">All</button>
        <button class="filter-btn" onclick="filterSentiment('positive', this)">🟢 Pos</button>
        <button class="filter-btn" onclick="filterSentiment('negative', this)">🔴 Neg</button>
        <button class="filter-btn" onclick="filterSentiment('neutral', this)">⚪ Neut</button>
      </div>
    </div>

    <div class="main-grid">
      <div>
        <div class="digest-card">
          <h2>📊 Daily Intelligence Summary</h2>
          <p class="digest-text">{{ digest.digest }}</p>
        </div>

        <div class="articles-grid" id="articlesGrid">
          {% for art in articles %}
          <div class="article-card" data-sentiment="{{ art.sentiment }}" data-date="{{ art.pub_date }}" data-ts="{{ art.pub_ts }}" data-relevance="{{ loop.index0 }}" data-content="{{ art.title | lower }} {{ art.summary | lower }}">
            <div class="card-head">
              <img class="favicon" src="{{ art.favicon }}" alt="" onerror="this.style.display='none'" />
              <div class="card-title-wrap">
                <div class="card-title"><a href="{{ art.url }}" target="_blank">{{ art.title }}</a></div>
                <div class="card-meta">
                  <span class="card-source">{{ art.source }}</span>
                  <span class="sentiment-dot {{ art.sentiment }}"></span>
                  <span>{{ art.published }}</span>
                </div>
              </div>
            </div>
            <p class="card-summary">{{ art.summary }}</p>
            <div class="card-tags">
              {% for tag in art.tags %}<span class="tag" onclick="filterByTag('{{ tag | lower }}')">{{ tag }}</span>{% endfor %}
            </div>
          </div>
          {% endfor %}
        </div>
      </div>

      <aside class="sidebar">
        <div class="widget">
          <div class="widget-title">Sentiment Analysis</div>
          {% for s, count in sentiment_counts.items() %}
          <div class="sent-row">
            <span class="sent-label">{{ s }}</span>
            <div class="sent-bar-wrap">
              <div class="sent-bar {{ s }}" style="width: {{ (count / articles|length * 100) | int }}%"></div>
            </div>
            <span style="font-size: 10px; color: var(--muted)">{{ count }}</span>
          </div>
          {% endfor %}
        </div>

        <div class="widget" style="background: transparent; border: none; padding: 0;">
          <div class="widget-title" style="padding-left: 0;">Things to watch</div>
          {% for event in digest.next_events %}
          <div class="event-card">
            <div class="event-num">{{ loop.index }}</div>
            <div class="event-content">
              <div class="event-title">{{ event.title }}</div>
              <div class="event-desc">{{ event.description }}</div>
            </div>
          </div>
          {% endfor %}
        </div>

        <div class="widget">
          <div class="widget-title">Key Themes</div>
          <div class="card-tags">
            {% for tag in all_tags %}
            <span class="tag" onclick="filterByTag('{{ tag | lower }}')">{{ tag }}</span>
            {% endfor %}
          </div>
        </div>
      </aside>
    </div>
  </div>

  <div id="marketsTab" class="tab-content">
    {% if hourly_commodities %}
    <div class="hourly-section" id="hourlySection">
      <div class="section-header">
        <div class="section-title">⏱ Intraday Price Variation — Weekly View (% Change)</div>
        <button class="filter-btn" id="toggleHourly" onclick="toggleHourly()" style="margin-left:12px; flex-shrink:0;">Hide Intraday</button>
      </div>
      <div id="hourlyBody">
        <div class="hourly-card">
          <div class="hourly-legend">
            {% for c in hourly_commodities %}
            <div class="legend-item">
              <span class="legend-dot" style="background: {{ ['#00e5ff','#7c3aed','#f59e0b','#22c55e','#ef4444','#ec4899'][loop.index0 % 6] }}"></span>
              <span>{{ c.name }}</span>
              <span class="legend-change {{ 'up' if c.change_pct >= 0 else 'down' }}">{{ '+' if c.change_pct >= 0 }}{{ c.change_pct }}%</span>
            </div>
            {% endfor %}
          </div>
          <div class="hourly-chart-wrap">
            <canvas id="hourlyChart"></canvas>
          </div>
          <div class="chart-controls" style="margin-top: 15px;">
            <button class="chart-btn" onclick="updateIntradayOverlay('1h', this)">1H</button>
            <button class="chart-btn" onclick="updateIntradayOverlay('4h', this)">4H</button>
            <button class="chart-btn" onclick="updateIntradayOverlay('open', this)">Open</button>
            <button class="chart-btn active" id="defaultOverlayBtn" onclick="updateIntradayOverlay('all', this)">7D</button>
            <button class="chart-btn" onclick="updateIntradayOverlay('1mo', this)">1M</button>
            <button class="chart-btn" onclick="updateIntradayOverlay('3mo', this)">3M</button>
            <button class="chart-btn" onclick="updateIntradayOverlay('1y', this)">1Y</button>
          </div>
          <div class="hourly-note">Y-axis shows % change from period open · Data via Yahoo Finance (5m interval)</div>
        </div>

        <div class="section-title" style="margin-top: 24px; margin-bottom: 14px;">📈 Intraday Absolute Prices — Weekly View</div>
        <div class="commodities-grid">
          {% for c in hourly_commodities %}
          <div class="commodity-card">
            <div class="comm-header">
              <div>
                <div class="comm-name">{{ c.name }}</div>
                <div class="comm-price">${{ c.latest_price }}</div>
              </div>
              <div class="comm-change {{ 'up' if c.change_pct >= 0 else 'down' }}">
                {{ '+' if c.change_pct >= 0 }}{{ c.change_pct }}%
              </div>
            </div>
            <div class="chart-container">
              <canvas id="hourly-abs-chart-{{ loop.index }}"></canvas>
            </div>
            <div class="chart-controls">
              <button class="chart-btn" onclick="updateIntradayAbsChart({{ loop.index0 }}, '1h', this)">1H</button>
              <button class="chart-btn" onclick="updateIntradayAbsChart({{ loop.index0 }}, '4h', this)">4H</button>
              <button class="chart-btn" onclick="updateIntradayAbsChart({{ loop.index0 }}, 'open', this)">Open</button>
              <button class="chart-btn active" onclick="updateIntradayAbsChart({{ loop.index0 }}, 'all', this)">7D</button>
              <button class="chart-btn" onclick="updateIntradayAbsChart({{ loop.index0 }}, '1mo', this)">1M</button>
              <button class="chart-btn" onclick="updateIntradayAbsChart({{ loop.index0 }}, '3mo', this)">3M</button>
              <button class="chart-btn" onclick="updateIntradayAbsChart({{ loop.index0 }}, '1y', this)">1Y</button>
            </div>
          </div>
          {% endfor %}
        </div>
      </div>
    </div>
    {% endif %}
  </div>

  {% if cfg.trade_tracker.enabled %}
  <div id="tradeTab" class="tab-content">
    {% if trade_data %}
    <div class="hourly-section">
      <div class="section-header">
        <div class="section-title">🚢 {{ cfg.trade_tracker.chokepoint }} Daily Transit & Capacity</div>
      </div>
      <div class="hourly-card">
        <div class="hourly-chart-wrap" style="height: 500px;">
          <canvas id="tradeChart"></canvas>
        </div>
        <div class="chart-controls" style="margin-top: 15px;">
          <button class="chart-btn" onclick="updateTradeChart('1mo', this)">1M</button>
          <button class="chart-btn" onclick="updateTradeChart('3mo', this)">3M</button>
          <button class="chart-btn" onclick="updateTradeChart('6mo', this)">6M</button>
          <button class="chart-btn active" id="defaultTradeBtn" onclick="updateTradeChart('ytd', this)">YTD</button>
          <button class="chart-btn" onclick="updateTradeChart('1y', this)">1Y</button>
        </div>
        <div class="hourly-note">Data source: IMF PortWatch / University of Oxford.</div>
      </div>
    </div>
    {% endif %}
  </div>
  {% endif %}

  {% if cfg.maritime_tracker.enabled %}
  <div id="maritimeTab" class="tab-content">
    <div class="hourly-section">
      <div class="section-header">
        <div class="section-title">⛴️ {{ cfg.maritime_tracker.chokepoint }} Real-Time Traffic</div>
      </div>
      <div class="main-grid" style="grid-template-columns: 1fr 340px;">
        <div class="hourly-card" style="padding: 0; overflow: hidden; height: 600px; position: relative;">
          <div id="map" style="height: 100%; width: 100%; background: #0a0c10;"></div>
        </div>
        <aside class="sidebar hide-mobile">
          <div class="widget">
            <div class="widget-title">Live Vessels ({{ ais_data | length }})</div>
            <div id="shipList" style="max-height: 540px; overflow-y: auto;">
              {% for ship in ais_data %}
              <div class="article-card" style="padding: 12px; margin-bottom: 8px; cursor: pointer;" onclick="focusShip({{ ship.lat }}, {{ ship.lon }}, '{{ ship.name }}')">
                <div class="card-title" style="font-size: 13px;">{{ ship.name or 'Unknown' }}</div>
                <div class="card-meta">
                  <span class="card-source">MMSI: {{ ship.mmsi }}</span>
                  <span>{{ ship.type }}</span>
                </div>
              </div>
              {% endfor %}
            </div>
          </div>
        </aside>
      </div>
    </div>
  </div>
  {% endif %}

  {% if cfg.missile_tracker.enabled %}
  <div id="missileTab" class="tab-content">
    <div class="hourly-section">
      <div class="section-header">
        <div class="section-title">🚀 Daily Attacks & Munitions Tracker</div>
      </div>
      {% if missile_data %}
      <div class="hourly-card">
        <div class="hourly-chart-wrap" style="height: 500px;">
          <canvas id="missileChart"></canvas>
        </div>
      </div>
      {% endif %}
    </div>
  </div>
  {% endif %}

  {% if cfg.gdelt_tracker.enabled %}
  <div id="gdeltTab" class="tab-content">
    <div class="hourly-section">
      <div class="section-header">
        <div class="section-title">🌍 GDELT Conflict & Infrastructure Events</div>
      </div>
      {% if gdelt_data.events %}
      <div class="hourly-card" style="padding: 0; overflow-x: auto;">
        <table style="width: 100%; border-collapse: collapse; color: var(--text); font-size: 12px;">
          <thead>
            <tr style="background: var(--surface2); text-align: left;">
              <th style="padding: 12px;">Date</th>
              <th style="padding: 12px;">Event Type</th>
              <th style="padding: 12px;">Location</th>
              <th style="padding: 12px;">Source</th>
            </tr>
          </thead>
          <tbody>
            {% for event in gdelt_data.events %}
            <tr style="border-bottom: 1px solid var(--border);">
              <td style="padding: 12px;">{{ event.date }}</td>
              <td style="padding: 12px;">{{ event.event_type }}</td>
              <td style="padding: 12px;">{{ event.location }}</td>
              <td style="padding: 12px;"><a href="{{ event.url }}" target="_blank" style="color: var(--accent);">Link ↗</a></td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      {% endif %}
    </div>
  </div>
  {% endif %}

  <footer>
    {{ title }} · Summaries via {{ model }} · {{ generated_at }}
  </footer>
</div>

<script>
let activeSentiment = 'all';
let activeTag = null;
let activeDates = new Set(['all']);

// Global chart references
let intradayOverlayChart = null;
const intradayAbsCharts = [];
let tradeChartRef = null;
let missileChartRef = null;

// Data constants — must be declared before any function that references them
const DAILY_DATA = {{ commodities | tojson }};
const INTRADAY_DATA = {{ hourly_commodities | tojson }};
const FULL_TRADE_DATA = {{ trade_data | tojson }};
const FULL_MISSILE_DATA = {{ missile_data | tojson }};
const FULL_AIS_DATA = {{ ais_data | tojson }};

let map = null;
let shipMarkers = [];

function initMap() {
  if (map) return;
  
  // Default center for Strait of Hormuz
  map = L.map('map').setView([26.7, 56.3], 8);
  
  L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
    subdomains: 'abcd',
    maxZoom: 20
  }).addTo(map);

  FULL_AIS_DATA.forEach(ship => {
    if (ship.lat && ship.lon) {
      const marker = L.circleMarker([ship.lat, ship.lon], {
        radius: 6,
        fillColor: "#00e5ff",
        color: "#fff",
        weight: 1,
        opacity: 1,
        fillOpacity: 0.8
      }).addTo(map);
      
      marker.bindPopup(`<b>${ship.name || 'Unknown'}</b><br>MMSI: ${ship.mmsi}<br>Type: ${ship.type}<br>Pos: ${ship.lat.toFixed(4)}, ${ship.lon.toFixed(4)}`);
      shipMarkers.push({mmsi: ship.mmsi, marker: marker});
    }
  });
}

function focusShip(lat, lon, name) {
  if (map && lat && lon) {
    map.setView([lat, lon], 12);
    // Find the marker and open its popup
    const shipMarker = shipMarkers.find(m => m.marker.getLatLng().lat === lat && m.marker.getLatLng().lng === lon);
    if (shipMarker) {
      shipMarker.marker.openPopup();
    }
  }
}

function updateMissileChart(period, btn) {
  if (!missileChartRef || !FULL_MISSILE_DATA || !Array.isArray(FULL_MISSILE_DATA)) return;
  const labels = FULL_MISSILE_DATA.map(d => d.date || '');
  if (labels.length === 0) return;

  let sliceSize = labels.length;
  if (period === '1mo') sliceSize = 30;
  else if (period === '3mo') sliceSize = 90;
  else if (period === '6mo') sliceSize = 180;
  else if (period === '1y') sliceSize = 365;

  if (sliceSize > labels.length) sliceSize = labels.length;
  const slice = FULL_MISSILE_DATA.slice(labels.length - sliceSize);
  
  // Store current slice for tooltips
  if (missileChartRef.data) missileChartRef.data.FULL_MISSILE_SLICE = slice;

  missileChartRef.data.labels = slice.map(d => d.date || '');
  if (missileChartRef.data.datasets[0]) missileChartRef.data.datasets[0].data = slice.map(d => d.ballistic_missiles || 0);
  if (missileChartRef.data.datasets[1]) missileChartRef.data.datasets[1].data = slice.map(d => d.cruise_missiles || 0);
  if (missileChartRef.data.datasets[2]) missileChartRef.data.datasets[2].data = slice.map(d => d.drones || 0);
  if (missileChartRef.data.datasets[3]) missileChartRef.data.datasets[3].data = slice.map(d => d.total_iranian || 0);
  missileChartRef.update();

  if (btn && btn.parentElement) {
    btn.parentElement.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }
}

function updateTradeChart(period, btn) {
  if (!tradeChartRef || !FULL_TRADE_DATA || !Array.isArray(FULL_TRADE_DATA)) return;
  const labels = FULL_TRADE_DATA.map(d => d.date || '');
  if (labels.length === 0) return;

  let sliceSize = labels.length;
  const now = new Date();
  if (period === '1mo') sliceSize = 30;
  else if (period === '3mo') sliceSize = 90;
  else if (period === '6mo') sliceSize = 180;
  else if (period === '1y') sliceSize = 365;
  else if (period === 'ytd') {
    const currentYear = now.getFullYear();
    const startOfYearStr = currentYear + '-01-01';
    const startIdx = labels.findIndex(l => l >= startOfYearStr);
    sliceSize = startIdx === -1 ? 0 : labels.length - startIdx;
  }
  
  if (sliceSize > labels.length) sliceSize = labels.length;
  const slice = FULL_TRADE_DATA.slice(labels.length - sliceSize);
  
  tradeChartRef.data.labels = slice.map(d => d.date || '');
  if (tradeChartRef.data.datasets[0]) tradeChartRef.data.datasets[0].data = slice.map(d => d.tanker || 0);
  if (tradeChartRef.data.datasets[1]) tradeChartRef.data.datasets[1].data = slice.map(d => d.container || 0);
  if (tradeChartRef.data.datasets[2]) tradeChartRef.data.datasets[2].data = slice.map(d => d.dry_bulk || 0);
  if (tradeChartRef.data.datasets[3]) tradeChartRef.data.datasets[3].data = slice.map(d => d.general_cargo || 0);
  if (tradeChartRef.data.datasets[4]) tradeChartRef.data.datasets[4].data = slice.map(d => d.roro || 0);
  if (tradeChartRef.data.datasets[5]) {
    tradeChartRef.data.datasets[5].data = slice.map(d => 
      (d.tanker || 0) + (d.container || 0) + (d.dry_bulk || 0) + (d.general_cargo || 0) + (d.roro || 0)
    );
  }
  
  tradeChartRef.update();
  
  if (btn && btn.parentElement) {
    btn.parentElement.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }
}

function switchTab(tabName, btn) {
  const content = document.getElementById(tabName + 'Tab');
  if (!content || !btn) return;

  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  
  content.classList.add('active');
  btn.classList.add('active');
  
  if (tabName === 'markets' || tabName === 'trade' || tabName === 'maritime') {
    window.dispatchEvent(new Event('resize'));
  }
  if (tabName === 'maritime') {
    setTimeout(initMap, 100);
  }
  if (tabName === 'missile') {
    setTimeout(function() {
      if (missileChartRef) {
        missileChartRef.resize();
        updateMissileChart('all', null);
      }
    }, 50);
  }
}

function toggleDate(date) {
  const btn = document.getElementById('date-' + date);
  if (!btn) return;
  
  if (date === 'all') {
    activeDates.clear();
    activeDates.add('all');
    document.querySelectorAll('[id^="date-"]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  } else {
    if (activeDates.has('all')) {
      activeDates.delete('all');
      const allBtn = document.getElementById('date-all');
      if (allBtn) allBtn.classList.remove('active');
    }
    if (activeDates.has(date)) {
      activeDates.delete(date);
      btn.classList.remove('active');
      if (activeDates.size === 0) {
        activeDates.add('all');
        const allBtn = document.getElementById('date-all');
        if (allBtn) allBtn.classList.add('active');
      }
    } else {
      activeDates.add(date);
      btn.classList.add('active');
    }
  }
  applyFilters();
}

function applyFilters() {
  const searchInput = document.getElementById('searchInput');
  const newsLimit = document.getElementById('newsLimit');
  if (!searchInput || !newsLimit) return;

  const q = searchInput.value.toLowerCase();
  const limit = newsLimit.value;
  let visibleCount = 0;

  document.querySelectorAll('.article-card').forEach(card => {
    const matchSentiment = activeSentiment === 'all' || card.dataset.sentiment === activeSentiment;
    const matchSearch = !q || (card.dataset.content && card.dataset.content.includes(q));
    const matchTag = !activeTag || card.innerText.toLowerCase().includes(activeTag);
    const matchDate = activeDates.has('all') || activeDates.has(card.dataset.date);
    
    const shouldShow = matchSentiment && matchSearch && matchTag && matchDate;
    
    if (shouldShow && (limit === 'all' || visibleCount < parseInt(limit))) {
      card.classList.remove('hidden');
      visibleCount++;
    } else {
      card.classList.add('hidden');
    }
  });
}

function applySort() {
  const sortOrder = document.getElementById('sortOrder');
  const grid = document.getElementById('articlesGrid');
  if (!sortOrder || !grid) return;

  const order = sortOrder.value;
  const articles = Array.from(grid.querySelectorAll('.article-card'));

  articles.sort((a, b) => {
    if (order === 'relevance') {
      return (parseInt(a.dataset.relevance) || 0) - (parseInt(b.dataset.relevance) || 0);
    }
    const tsA = parseInt(a.dataset.ts) || 0;
    const tsB = parseInt(b.dataset.ts) || 0;
    return order === 'desc' ? tsB - tsA : tsA - tsB;
  });

  articles.forEach(art => grid.appendChild(art));
}

function filterSentiment(sentiment, btn) {
  if (!btn) return;
  activeSentiment = sentiment;
  const parent = btn.parentElement;
  if (parent) {
    parent.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  }
  btn.classList.add('active');
  applyFilters();
}

function filterByTag(tag) {
  activeTag = (activeTag === tag) ? null : tag;
  applyFilters();
}

function toggleHourly() {
  const body = document.getElementById('hourlyBody');
  const btn = document.getElementById('toggleHourly');
  if (!body || !btn) return;

  if (body.style.display === 'none') {
    body.style.display = 'block';
    btn.innerText = 'Hide Intraday';
  } else {
    body.style.display = 'none';
    btn.innerText = 'Show Intraday';
  }
}

function getPeriodDays(period) {
  const map = { '1w': 5, '7d': 7, '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365 };
  return map[period] || 365;
}

function updateIntradayOverlay(period, btn) {
  const chart = intradayOverlayChart;
  if (!chart || !INTRADAY_DATA || INTRADAY_DATA.length === 0) return;
  
  const isDaily = ['3mo', '6mo', '1y'].includes(period);
  const dataRef = (isDaily && DAILY_DATA && DAILY_DATA.length > 0) ? DAILY_DATA : INTRADAY_DATA;
  
  if (!dataRef || dataRef.length === 0) return;
  const labels = isDaily ? dataRef[0].history_labels : dataRef[0].labels;
  const timestamps = isDaily ? dataRef[0].history_timestamps : dataRef[0].timestamps;
  
  if (!labels || labels.length === 0) return;
  
  let sliceSize = labels.length;
  
  if (period === '1h' || period === '4h') {
    const hours = period === '1h' ? 1 : 4;
    const lastTs = timestamps[timestamps.length - 1];
    const targetTs = lastTs - (hours * 60 * 60 * 1000);
    const startIdx = timestamps.findIndex(t => t >= targetTs);
    sliceSize = labels.length - (startIdx === -1 ? 0 : startIdx);
  } else if (period === 'open') {
    const lastLabel = labels[labels.length - 1];
    if (lastLabel) {
      const lastDate = lastLabel.split(' ').slice(0, 2).join(' '); // e.g. "Oct 24"
      const firstIndexToday = labels.findIndex(l => l && l.startsWith(lastDate));
      sliceSize = labels.length - (firstIndexToday === -1 ? 0 : firstIndexToday);
    }
  } else if (period === 'all') {
    const lastTs = timestamps[timestamps.length - 1];
    const targetTs = lastTs - (7 * 24 * 60 * 60 * 1000);
    const startIdx = timestamps.findIndex(t => t >= targetTs);
    sliceSize = labels.length - (startIdx === -1 ? 0 : startIdx);
  } else if (period === '1mo' && !isDaily) {
    sliceSize = labels.length;
  } else if (isDaily) {
    sliceSize = getPeriodDays(period);
  }
  
  const hourlySection = document.getElementById('hourlySection');
  const changeElements = hourlySection ? hourlySection.querySelectorAll('.legend-change') : [];
  
  chart.data.datasets.forEach((dataset, idx) => {
    // Find the matching commodity in dataRef by name if possible, otherwise use index
    const commName = dataset.label;
    let comm = dataRef.find(c => c.name === commName) || dataRef[idx];
    if (!comm) return;

    const values = isDaily ? comm.history_values : comm.raw_values;
    if (!values || values.length === 0) return;

    const slice = values.slice(-sliceSize);
    const base = slice[0] || 1;
    const latest = slice[slice.length - 1];
    const changeValue = ((latest - base) / base * 100);
    const change = isNaN(changeValue) ? "0.00" : changeValue.toFixed(2);
    
    dataset.data = slice.map(v => ((v - base) / base * 100));
    
    if (changeElements[idx]) {
      changeElements[idx].innerText = (changeValue >= 0 ? '+' : '') + change + '%';
      changeElements[idx].className = 'legend-change ' + (changeValue >= 0 ? 'up' : 'down');
    }
  });
  
  let currentLabels = labels.slice(-sliceSize);
  if (isDaily || period === 'all' || period === '1mo') {
    currentLabels = currentLabels.map(l => {
        if (!l) return '';
        return isDaily ? l.split('-').slice(1).join('/') : l.split(' ').slice(0, 2).join(' ');
    });
  } else {
    currentLabels = currentLabels.map(l => l ? l.split(' ').pop() : '');
  }
  
  chart.data.labels = currentLabels;
  chart.update();
  
  if (btn && btn.parentElement) {
    btn.parentElement.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }
}

function updateIntradayAbsChart(idx, period, btn) {
  const chart = intradayAbsCharts[idx];
  if (!chart) return;

  const isDaily = ['3mo', '6mo', '1y'].includes(period);
  const dataSet = (isDaily && DAILY_DATA && DAILY_DATA.length > 0) ? DAILY_DATA : INTRADAY_DATA;
  
  // Try to find by name if we can, but we usually have idx from the loop
  const data = dataSet[idx];
  if (!data) return;

  const labels = isDaily ? data.history_labels : data.labels;
  const timestamps = isDaily ? data.history_timestamps : data.timestamps;
  
  if (!labels || labels.length === 0) return;

  let sliceSize = labels.length;
  
  if (period === '1h' || period === '4h') {
    const hours = period === '1h' ? 1 : 4;
    const lastTs = timestamps[timestamps.length - 1];
    const targetTs = lastTs - (hours * 60 * 60 * 1000);
    const startIdx = timestamps.findIndex(t => t >= targetTs);
    sliceSize = labels.length - (startIdx === -1 ? 0 : startIdx);
  } else if (period === 'open') {
    const lastLabel = labels[labels.length - 1];
    if (lastLabel) {
      const lastDate = lastLabel.split(' ').slice(0, 2).join(' ');
      const firstIndexToday = labels.findIndex(l => l && l.startsWith(lastDate));
      sliceSize = labels.length - (firstIndexToday === -1 ? 0 : firstIndexToday);
    }
  } else if (period === 'all') {
    const lastTs = timestamps[timestamps.length - 1];
    const targetTs = lastTs - (7 * 24 * 60 * 60 * 1000);
    const startIdx = timestamps.findIndex(t => t >= targetTs);
    sliceSize = labels.length - (startIdx === -1 ? 0 : startIdx);
  } else if (period === '1mo' && !isDaily) {
    sliceSize = labels.length;
  } else if (isDaily) {
    sliceSize = getPeriodDays(period);
  }
  
  const card = btn ? btn.closest('.commodity-card') : null;
  const changeEl = card ? card.querySelector('.comm-change') : null;
  const priceEl = card ? card.querySelector('.comm-price') : null;
  
  const values = isDaily ? data.history_values : data.raw_values;
  if (!values || values.length === 0) return;

  const slice = values.slice(-sliceSize);
  const first = slice[0];
  const last = slice[slice.length - 1];
  const changeValue = ((last - first) / first * 100);
  const change = isNaN(changeValue) ? "0.00" : changeValue.toFixed(2);
  
  if (changeEl) {
    changeEl.innerText = (changeValue >= 0 ? '+' : '') + change + '%';
    changeEl.className = 'comm-change ' + (changeValue >= 0 ? 'up' : 'down');
  }
  if (priceEl && last !== undefined) {
    priceEl.innerText = '$' + last.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
  }
  
  let currentLabels = labels.slice(-sliceSize);
  if (isDaily || period === 'all' || period === '1mo') {
    currentLabels = currentLabels.map(l => {
        if (!l) return '';
        return isDaily ? l.split('-').slice(1).join('/') : l.split(' ').slice(0, 2).join(' ');
    });
  } else {
    currentLabels = currentLabels.map(l => l ? l.split(' ').pop() : '');
  }
  
  chart.data.labels = currentLabels;
  chart.data.datasets[0].data = slice;
  
  const color = last >= first ? '#22c55e' : '#ef4444';
  const bg = last >= first ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)';
  
  chart.data.datasets[0].borderColor = color;
  chart.data.datasets[0].backgroundColor = bg;
  
  chart.update();
  
  if (btn && btn.parentElement) {
    btn.parentElement.querySelectorAll('.chart-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }
}

document.addEventListener('DOMContentLoaded', function() {
  // ── Hourly intraday overlay chart ────────────────────────────────────────
  {% if hourly_commodities %}
  const HOURLY_COLORS = ['#00e5ff', '#7c3aed', '#f59e0b', '#22c55e', '#ef4444', '#ec4899'];
  const hourlyDatasets = [
    {% for c in hourly_commodities %}
    {
      label: '{{ c.name }}',
      data: {{ c.pct_values | tojson }},
      borderColor: HOURLY_COLORS[{{ loop.index0 }} % HOURLY_COLORS.length],
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 0,
      pointHoverRadius: 5,
      tension: 0,
    },
    {% endfor %}
  ];
  const hourlyLabels = {{ hourly_commodities[0].labels | tojson }};

  intradayOverlayChart = new Chart(document.getElementById('hourlyChart'), {
    type: 'line',
    data: { labels: hourlyLabels, datasets: hourlyDatasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: true,
          backgroundColor: 'rgba(26, 32, 48, 0.95)',
          titleColor: '#00e5ff',
          bodyColor: '#fff',
          titleFont: { family: 'IBM Plex Mono', size: 11 },
          bodyFont: { family: 'IBM Plex Mono', size: 11 },
          displayColors: true,
          padding: 12,
          callbacks: {
            title: function(context) {
              if (!context || !context[0]) return '';
              const sliceSize = context[0].dataset.data.length;
              const hourlySection = document.getElementById('hourlySection');
              const activeBtn = hourlySection ? hourlySection.querySelector('.chart-btn.active') : null;
              const btnText = activeBtn ? activeBtn.innerText.trim() : '';
              const isDaily = ['3M', '1Y'].includes(btnText);
              const dataSet = (isDaily && DAILY_DATA && DAILY_DATA.length > 0) ? DAILY_DATA : INTRADAY_DATA;
              if (!dataSet || dataSet.length === 0) return '';
              const labels = isDaily ? dataSet[0].history_labels : dataSet[0].labels;
              if (!labels) return '';
              const ptIdx = labels.length - sliceSize + context[0].dataIndex;
              return labels[ptIdx] || '';
            },
            label: function(context) {
              if (!context) return '';
              const sign = context.parsed.y >= 0 ? '+' : '';
              const dsIdx = context.datasetIndex;
              const sliceSize = context.dataset.data.length;
              const hourlySection = document.getElementById('hourlySection');
              const activeBtn = hourlySection ? hourlySection.querySelector('.chart-btn.active') : null;
              const btnText = activeBtn ? activeBtn.innerText.trim() : '';
              const isDaily = ['3M', '1Y'].includes(btnText);
              const dataSet = (isDaily && DAILY_DATA && DAILY_DATA.length > 0) ? DAILY_DATA : INTRADAY_DATA;
              if (!dataSet || !dataSet[dsIdx]) return ' ' + context.dataset.label + ': ' + sign + context.parsed.y.toFixed(3) + '%';
              
              const comm = dataSet[dsIdx];
              const labels = isDaily ? comm.history_labels : comm.labels;
              const values = isDaily ? comm.history_values : comm.raw_values;
              if (!labels || !values) return ' ' + context.dataset.label + ': ' + sign + context.parsed.y.toFixed(3) + '%';

              const ptIdx = labels.length - sliceSize + context.dataIndex;
              const absPrice = (ptIdx >= 0 && values[ptIdx] !== undefined)
                ? '  $' + values[ptIdx].toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})
                : '';
              return ' ' + context.dataset.label + ': ' + sign + context.parsed.y.toFixed(3) + '%' + absPrice;
            }
          }
        }
      },
      scales: {
        x: {
          display: true,
          grid: { display: false },
          ticks: { color: '#fff', font: { size: 9 }, maxRotation: 0, maxTicksLimit: 10 }
        },
        y: {
          display: true,
          grid: { color: 'var(--border)', drawBorder: false },
          ticks: {
            color: '#fff', font: { size: 9 },
            callback: function(v) { return (v >= 0 ? '+' : '') + v.toFixed(2) + '%'; }
          }
        }
      }
    }
  });
  {% endif %}

  // ── Intraday absolute-price charts ───────────────────────────────────────
  {% if hourly_commodities %}
  {% for c in hourly_commodities %}
  intradayAbsCharts.push(new Chart(document.getElementById('hourly-abs-chart-{{ loop.index }}'), {
    type: 'line',
    data: {
      labels: {{ c.labels | tojson }},
      datasets: [{
        label: '{{ c.name }}',
        data: {{ c.raw_values | tojson }},
        borderColor: '{{ "#22c55e" if c.change_pct >= 0 else "#ef4444" }}',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 5,
        tension: 0,
        fill: true,
        backgroundColor: '{{ "rgba(34, 197, 94, 0.1)" if c.change_pct >= 0 else "rgba(239, 68, 68, 0.1)" }}'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: true,
          backgroundColor: 'rgba(26, 32, 48, 0.9)',
          titleColor: '#00e5ff',
          bodyColor: '#fff',
          titleFont: { family: 'IBM Plex Mono', size: 11 },
          bodyFont: { family: 'IBM Plex Mono', size: 12 },
          displayColors: false,
          padding: 10,
          callbacks: {
            title: function(context) {
              if (!context || !context[0]) return '';
              const sliceSize = context[0].dataset.data.length;
              const dsIdx = {{ loop.index0 }};
              const card = context[0].chart.canvas.closest('.commodity-card');
              const activeBtn = card ? card.querySelector('.chart-btn.active') : null;
              const btnText = activeBtn ? activeBtn.innerText.trim() : '';
              const isDaily = ['3M', '1Y'].includes(btnText);
              const dataSet = (isDaily && DAILY_DATA && DAILY_DATA.length > 0) ? DAILY_DATA : INTRADAY_DATA;
              if (!dataSet || !dataSet[dsIdx]) return '';
              const labels = isDaily ? dataSet[dsIdx].history_labels : dataSet[dsIdx].labels;
              if (!labels) return '';
              const ptIdx = labels.length - sliceSize + context[0].dataIndex;
              return labels[ptIdx] || '';
            },
            label: function(context) {
              if (!context) return '';
              return 'Price: $' + context.parsed.y.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
            }
          }
        }
      },
      scales: {
        x: { 
          display: true,
          grid: { display: false },
          ticks: { color: '#fff', font: { size: 9 }, maxRotation: 0, maxTicksLimit: 10 }
        },
        y: { 
          display: true,
          grid: { color: 'var(--border)', drawBorder: false },
          ticks: { 
            color: '#fff', font: { size: 9 }, 
            callback: function(value) { 
              if (value >= 100) return '$' + Math.round(value).toLocaleString();
              return '$' + value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
            } 
          }
        }
      }
    }
  }));
  {% endfor %}
  {% endif %}

  // ── Set default view to 7D ──────────────────────────────────────────────
  const defaultOverlayBtn = document.getElementById('defaultOverlayBtn');
  if (defaultOverlayBtn) updateIntradayOverlay('all', defaultOverlayBtn);

  document.querySelectorAll(".commodity-card .chart-btn.active[onclick*='updateIntradayAbsChart']").forEach((btn, idx) => {
    updateIntradayAbsChart(idx, 'all', btn);
  });

  // ── Trade Tracker Chart ──────────────────────────────────────────────────
  {% if trade_data %}
  (function() {
    const ctx = document.getElementById('tradeChart');
    if (ctx) {
      tradeChartRef = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: [],
          datasets: [
            { label: 'Tankers', data: [], backgroundColor: '#00e5ff' },
            { label: 'Containers', data: [], backgroundColor: '#7c3aed' },
            { label: 'Dry Bulk', data: [], backgroundColor: '#f59e0b' },
            { label: 'General Cargo', data: [], backgroundColor: '#22c55e' },
            { label: 'RoRo', data: [], backgroundColor: '#ef4444' },
            { 
              label: 'Total Transit', 
              data: [], 
              type: 'line', 
              borderColor: '#fff', 
              borderWidth: 2, 
              pointRadius: 0, 
              fill: false,
              tension: 0.1,
              order: -1
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { intersect: false, mode: 'index' },
          scales: {
            x: { stacked: true, ticks: { color: '#c8d8e8', maxRotation: 45, maxTicksLimit: 15 }, grid: { color: 'rgba(255, 255, 255, 0.05)' } },
            y: { stacked: true, ticks: { color: '#c8d8e8' }, grid: { color: 'rgba(255, 255, 255, 0.05)' }, title: { display: true, text: 'Vessel Count', color: '#c8d8e8' } }
          },
          plugins: {
            legend: { position: 'top', labels: { color: '#c8d8e8', font: { family: 'IBM Plex Mono', size: 11 } } },
            tooltip: {
              backgroundColor: 'rgba(10, 12, 16, 0.9)',
              titleColor: '#00e5ff',
              bodyColor: '#fff',
              borderColor: '#1e2a3a',
              borderWidth: 1,
              titleFont: { family: 'Inter', weight: 'bold' },
              bodyFont: { family: 'IBM Plex Mono' }
            }
          }
        }
      });
      // Set default view to YTD
      const defaultBtn = document.getElementById('defaultTradeBtn');
      if (defaultBtn) updateTradeChart('ytd', defaultBtn);
    }
  })();
  {% endif %}

  // ── Missile Tracker Chart ────────────────────────────────────────────────
  {% if missile_data %}
  (function() {
    const ctx = document.getElementById('missileChart');
    if (ctx) {
      const wrapper = ctx.parentElement;
      ctx.width = wrapper.clientWidth || 800;
      ctx.height = wrapper.clientHeight || 500;

      missileChartRef = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: FULL_MISSILE_DATA.map(d => d.date),
          datasets: [
            {
              label: 'Iranian Ballistic',
              data: FULL_MISSILE_DATA.map(d => d.ballistic_missiles || 0),
              backgroundColor: 'rgba(239, 68, 68, 0.7)',
              borderColor: '#ef4444',
              borderWidth: 1,
              stack: 'iran'
            },
            {
              label: 'Iranian Cruise',
              data: FULL_MISSILE_DATA.map(d => d.cruise_missiles || 0),
              backgroundColor: 'rgba(239, 68, 68, 0.4)',
              borderColor: '#ef4444',
              borderWidth: 1,
              stack: 'iran'
            },
            {
              label: 'Iranian Drones',
              data: FULL_MISSILE_DATA.map(d => d.drones || 0),
              backgroundColor: 'rgba(59, 130, 246, 0.7)',
              borderColor: '#3b82f6',
              borderWidth: 1,
              stack: 'iran'
            },
            {
              label: 'Total Amount',
              data: FULL_MISSILE_DATA.map(d => d.total_iranian || 0),
              type: 'line',
              borderColor: '#fff',
              borderWidth: 2,
              pointRadius: 0,
              fill: false,
              tension: 0.1,
              order: -1
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { intersect: false, mode: 'index' },
          scales: {
            x: {
              stacked: true,
              ticks: { color: '#c8d8e8', maxRotation: 45, maxTicksLimit: 14 },
              grid: { color: 'rgba(255, 255, 255, 0.05)' }
            },
            y: {
              stacked: true,
              beginAtZero: true,
              ticks: { color: '#c8d8e8' },
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              title: { display: true, text: 'Munitions / Launches', color: '#c8d8e8' }
            }
          },
          plugins: {
            legend: {
              position: 'top',
              labels: { color: '#c8d8e8', font: { family: 'IBM Plex Mono', size: 10 }, usePointStyle: true }
            },
            tooltip: {
              backgroundColor: 'rgba(10, 12, 16, 0.95)',
              titleColor: '#fff',
              bodyColor: '#fff',
              borderColor: '#1e2a3a',
              borderWidth: 1,
              callbacks: {
                label: function(context) {
                  let label = context.dataset.label || '';
                  if (label) label += ': ';
                  if (context.parsed.y !== null) label += context.parsed.y;
                  return label;
                },
                afterBody: function(context) {
                  const dataIdx = context[0].dataIndex;
                  const slice = (missileChartRef.data && missileChartRef.data.FULL_MISSILE_SLICE) ? missileChartRef.data.FULL_MISSILE_SLICE : FULL_MISSILE_DATA;
                  const d = slice[dataIdx];
                  if (!d) return '';
                  let lines = ['Total Iranian: ' + (d.total_iranian || 0)];
                  if (d.summary) lines.push('\\nSummary: ' + d.summary);
                  return lines.join('');
                }
              }
            }
          }
        }
      });
    }
  })();
  {% endif %}
});
</script>
</body>
</html>
"""

    safe_missile_data = missile_data

    tmpl_html = env.from_string(template_content)
    
    raw_dates = sorted(list(set(a["pub_date"] for a in articles if a.get("pub_date"))), reverse=True)
    
    today = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    
    unique_dates = []
    for d_str in raw_dates:
        try:
            d_obj = datetime.strptime(d_str, "%Y-%m-%d").date()
            if d_obj == today:
                label = "Today"
            elif d_obj == yesterday:
                label = "Yesterday"
            else:
                label = d_obj.strftime("%b %d")
        except:
            label = d_str
        unique_dates.append({"iso": d_str, "label": label})

    sentiment_counts = Counter(a["sentiment"] for a in articles)
    all_tags = []
    for a in articles: all_tags.extend(a["tags"])
    top_tags = [t for t, _ in Counter(all_tags).most_common(20)]
    
    dash = cfg["dashboard"]
    context = dict(
        title=dash["title"],
        topic=dash["topic"],
        period=dash["period"],
        generated_at=f"{datetime.now(timezone.utc).strftime('%b %d, %Y · %H:%M UTC')} / {(datetime.now(timezone.utc) - timedelta(hours=3)).strftime('%H:%M BRT')}",
        articles=articles,
        commodities=commodities,
        hourly_commodities=intraday_commodities,
        trade_data=trade_data,
        missile_data=safe_missile_data,
        ais_data=ais_data,
        gdelt_data=gdelt_data,
        digest=digest,
        sentiment_counts=sentiment_counts,
        all_tags=top_tags,
        unique_dates=unique_dates,
        model=f"{cfg['llm'].get('digest_model', 'N/A')} & {cfg['llm'].get('watch_model', 'N/A')}" if cfg["llm"].get("enabled", True) else "Metadata Extraction",
        theme=cfg["output"]["theme"],
        cfg=cfg
    )

    return tmpl_html.render(**context)


def main():
    parser = argparse.ArgumentParser(description="News Dashboard Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dash = cfg["dashboard"]

    print(f"\\n📰  News Dashboard Pipeline: {dash['title']}")
    
    print("⏳  Step 1: Fetching recent news (current + previous day)...")
    recent_articles = fetch_articles(cfg, period="2d", max_articles=dash["max_articles_recent"])
    
    today_dt = datetime.now(timezone.utc).date()
    digest_dates = [(today_dt - timedelta(days=i)).isoformat() for i in range(2)]
    
    llama_context_articles = [
        a for a in recent_articles 
        if a.get("pub_date") in digest_dates
    ]
    
    print(f"📌  LLM Context: Filtered {len(llama_context_articles)} articles strictly from the current and previous day.")

    print(f"⏳  Step 2: Fetching remaining news (lookback: {dash['period']})...")
    older_articles = fetch_articles(cfg, period=dash["period"], max_articles=dash["max_articles_older"] + len(recent_articles))
    
    recent_urls = {a["url"] for a in recent_articles}
    filtered_older = [a for a in older_articles if a["url"] not in recent_urls][:dash["max_articles_older"]]
    
    articles = recent_articles + filtered_older
    
    if not articles: sys.exit("No articles found.")

    commodities = fetch_commodity_prices(cfg)
    intraday_commodities = fetch_commodity_intraday(cfg)
    trade_data = fetch_trade_tracker_data(cfg)
    missile_data = fetch_missile_tracker_data(cfg)
    ais_data = fetch_ais_data(cfg)
    gdelt_data = fetch_gdelt_data(cfg)

    print(f"\\n📸  Extracting news previews...")
    articles = summarise_articles(articles, cfg)

    if cfg["llm"].get("enabled", True):
        articles = categorize_sentiment(articles, cfg)

        digest = generate_digest(llama_context_articles, commodities, cfg)
    else:
        print("\\n🚫  AI generated features (sentiment, digest) are disabled in config.")
        digest = {
            "digest": "AI generated digest is disabled in the configuration.",
            "top_themes": [],
            "next_events": []
        }

    print("\\n🎨  Rendering HTML dashboard…")
    out_path = Path(cfg["output"]["filename"])
    
    html = render_html(articles, commodities, intraday_commodities, trade_data, missile_data, ais_data, gdelt_data, digest, cfg)
    
    out_path.write_text(html, encoding="utf-8")
    
    print(f"✅  Dashboard saved → {out_path.resolve()}")

    if cfg["output"]["open_browser"]:
        webbrowser.open(out_path.resolve().as_uri())


if __name__ == "__main__":
    main()
