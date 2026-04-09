import json
import os
import requests
import yaml
import asyncio
from datetime import datetime, timezone, timedelta

# Import your collection logic
try:
    from news_dashboard_github import _collect_ais_messages
except ImportError:
    _collect_ais_messages = None

CONFIG_PATH = "config.yaml"
DB_FILE = "maritime_history.json"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def get_latest_portwatch_date(cfg):
    """
    Finds the MOST RECENT date available in Port Watch.
    Strait of Hormuz is typically 'chokepoint6'.
    """
    # Using 'portid' is more reliable than 'portname' strings
    base_url = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Chokepoints_Data/FeatureServer/0/query"
    params = {
        "where": "portid = 'chokepoint6' OR portname = 'Strait of Hormuz'", 
        "outFields": "date",
        "orderByFields": "date DESC",
        "resultRecordCount": 1,
        "f": "json"
    }
    try:
        response = requests.get(base_url, params=params, timeout=15)
        data = response.json()
        
        # Check for ArcGIS Error JSON
        if "error" in data:
            print(f"⚠️ PortWatch API Error: {data['error'].get('message')}")
            return None
            
        if "features" in data and len(data["features"]) > 0:
            latest_ts = data["features"][0]["attributes"]["date"]
            # PortWatch uses UTC timestamps in milliseconds
            return datetime.fromtimestamp(latest_ts / 1000, tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        print("⚠️ PortWatch returned no features for this chokepoint.")
        return None
    except Exception as e:
        print(f"⚠️ PortWatch request failed: {e}")
        return None

def main():
    print(f"🚀 Starting Maritime Update: {datetime.now(timezone.utc)}")
    cfg = load_config()
    
    # 1. Determine the 'Lag Point'
    latest_pw_dt = get_latest_portwatch_date(cfg)
    
    # FALLBACK: If API is down, keep the last 14 days to avoid a total skip
    if not latest_pw_dt:
        print("🔄 PortWatch API unavailable. Using fallback (14-day window).")
        latest_pw_dt = datetime.now(timezone.utc) - timedelta(days=14)
    
    print(f"📅 Keeping history from: {latest_pw_dt.strftime('%Y-%m-%d')} to Present")

    # 2. Fetch new AIS snapshot
    vessels = []
    if _collect_ais_messages:
        m_cfg = cfg.get("maritime_tracker", {})
        bbox = m_cfg.get("bounding_box", [[[22.0, 48.0], [30.5, 60.0]]])
        
        # Ensure correct bbox nesting for AISStream
        if bbox and not isinstance(bbox[0][0], list):
            bbox = [bbox]

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            vessels = loop.run_until_complete(_collect_ais_messages(
                m_cfg.get("api_key"), 
                bbox, 
                m_cfg.get("collect_duration", 30)
            ))
            loop.close()
            print(f"🚢 Found {len(vessels)} vessels.")
        except Exception as e:
            print(f"❌ AIS Connection Error: {e}")

    # 3. Process History
    history = []
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "r") as f:
            try:
                history = json.load(f)
            except: history = []

    # Add the current snapshot
    now = datetime.now(timezone.utc)
    history.append({
        "snapshot_ts": now.isoformat(),
        "readable_time": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "vessel_count": len(vessels),
        "vessels": vessels
    })

    # 4. PRUNE: Only keep snapshots >= latest_pw_dt
    # This aligns your database with the PortWatch timeline
    pruned_history = [
        snap for snap in history 
        if datetime.fromisoformat(snap["snapshot_ts"]) >= latest_pw_dt
    ]

    # 5. Save
    with open(DB_FILE, "w") as f:
        json.dump(pruned_history, f, indent=2)
    
    print(f"✅ Database saved. Snapshots in history: {len(pruned_history)}")

if __name__ == "__main__":
    main()
