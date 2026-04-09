import json
import os
import requests
import yaml
from datetime import datetime, timezone

# Files
CONFIG_PATH = "config.yaml"
DB_FILE = "maritime_history.json"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def get_latest_portwatch_date(cfg):
    """Finds the MOST RECENT date currently available in Port Watch."""
    tt_cfg = cfg.get("trade_tracker", {})
    chokepoint = tt_cfg.get("chokepoint", "Strait of Hormuz")
    
    base_url = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Chokepoints_Data/FeatureServer/0/query"
    params = {
        "where": f"portname = '{chokepoint}'",
        "outFields": "date",
        "orderByFields": "date DESC", # Get the latest one
        "resultRecordCount": 1,
        "f": "json"
    }
    try:
        response = requests.get(base_url, params=params, timeout=15)
        data = response.json()
        latest_ts = data["features"][0]["attributes"]["date"]
        # Convert to a date object (midnight of that lagged day)
        return datetime.fromtimestamp(latest_ts / 1000, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    except Exception as e:
        print(f"⚠️ PortWatch fetch failed: {e}")
        return None

def fetch_ais_snapshot(cfg):
    """
    Since GitHub Actions isn't a persistent socket, we use a simple 
    one-off HTTP request or a short async burst to grab current traffic.
    """
    # Note: Using your existing logic from news_dashboard_github.py here
    # (Simplified for brevity, ensure you keep your websocket logic)
    import asyncio
    from news_dashboard_github import _collect_ais_messages
    
    m_cfg = cfg["maritime_tracker"]
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        data = loop.run_until_complete(_collect_ais_messages(
            m_cfg["api_key"], m_cfg["bounding_box"], m_cfg.get("collect_duration", 30)
        ))
        loop.close()
        return data
    except Exception as e:
        print(f"❌ AIS Fetch failed: {e}")
        return []

def main():
    cfg = load_config()
    
    # 1. Get the 'Lagged' Latest Date from Port Watch
    latest_pw_dt = get_latest_portwatch_date(cfg)
    if not latest_pw_dt:
        return

    # 2. Fetch new AIS snapshot
    new_vessels = fetch_ais_snapshot(cfg)
    
    # 3. Load/Create local History
    history = []
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                history = json.load(f)
            except: history = []

    if new_vessels:
        history.append({
            "snapshot_ts": datetime.now(timezone.utc).isoformat(),
            "vessels": new_vessels
        })

    # 4. PRUNE: Only keep data from (latest_pw_dt) to (Present)
    # This ignores the lookback window and focuses purely on the Port Watch lag.
    pruned_history = [
        snap for snap in history 
        if datetime.fromisoformat(snap["snapshot_ts"]) >= latest_pw_dt
    ]

    # 5. Save locally (to be committed by GitHub Action)
    with open(DB_FILE, "w") as f:
        json.dump(pruned_history, f, indent=2)
    
    print(f"✅ DB Updated. Range: {latest_pw_dt.date()} to Present. Total snapshots: {len(pruned_history)}")

if __name__ == "__main__":
    main()
