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
    
    # 1. Get the 'Lagged' Latest Date from Port Watch (Reference Point)
    latest_pw_dt = get_latest_portwatch_date(cfg)
    if not latest_pw_dt:
        print("❌ Could not determine Port Watch lag date. Skipping update.")
        return

    # 2. Fetch new AIS snapshot
    print(f"📡 Requesting live AIS burst for {cfg['maritime_tracker']['chokepoint']}...")
    new_vessels = fetch_ais_snapshot(cfg)
    
    # 3. Load existing History
    history = []
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                history = json.load(f)
            except: 
                history = []

    # 4. Add new snapshot with enriched info
    # We save the exact ISO timestamp for pruning and a pretty version for your eyes
    now = datetime.now(timezone.utc)
    
    snapshot_entry = {
        "snapshot_ts": now.isoformat(),                   # For script processing
        "readable_time": now.strftime("%Y-%m-%d %H:%M:%S UTC"), # For humans
        "port_watch_lag_reference": latest_pw_dt.strftime("%Y-%m-%d"),
        "vessel_count": len(new_vessels),
        "vessels": new_vessels                            # The actual ship data
    }
    
    history.append(snapshot_entry)

    # 5. PRUNE: Only keep data from the 'Lag Date' forward
    # This ensures your DB only grows from the point Port Watch currently stops.
    pruned_history = [
        snap for snap in history 
        if datetime.fromisoformat(snap["snapshot_ts"]) >= latest_pw_dt
    ]

    # 6. Save back to the JSON file
    with open(DB_FILE, "w") as f:
        json.dump(pruned_history, f, indent=2)
    
    print(f"✅ DB Updated.")
    print(f"   - PW Latest Date: {latest_pw_dt.date()}")
    print(f"   - Vessels Found: {len(new_vessels)}")
    print(f"   - Total Snapshots Stored: {len(pruned_history)}")

if __name__ == "__main__":
    main()
