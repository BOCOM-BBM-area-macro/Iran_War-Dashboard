"""Crypto Fear & Greed Index widget using Alternative.me API."""
from ..core.output_manager import OutputManager

from datetime import datetime, timezone
from typing import Any, Dict, List
from ..core.base_widget import BaseWidget
from ..core.url_fetch_manager import get_url_fetch_manager


class CryptoFearGreedWidget(BaseWidget):
    """Displays Bitcoin Fear & Greed Index from Alternative.me.

    No params required - this is a Bitcoin-only metric.
    Fetches current value + 7 days of historical data.
    """

    def get_required_params(self) -> list[str]:
        return []  # No params needed

    def fetch_data(self) -> Dict[str, Any]:
        """Fetch Fear & Greed Index from Alternative.me API."""
        client = get_url_fetch_manager()

        try:
            # Fetch current + 365 days of historical data (for different timeframes)
            url = "https://api.alternative.me/fng/"
            params = {"limit": 366}  # Today + 365 days history (1 year)
            response = client.get(url, params=params, response_type="json")

            # Parse data
            raw_data = response["data"]

            if not raw_data:
                raise ValueError("No data returned from Fear & Greed API")

            # Current value (first item)
            current = raw_data[0]
            current_value = int(current["value"])
            current_classification = current["value_classification"]
            current_timestamp = int(current["timestamp"])

            # Historical values for chart (up to 90 days in chronological order)
            historical = [
                {
                    "value": int(item["value"]),
                    "timestamp": int(item["timestamp"]),
                    "classification": item["value_classification"]
                }
                for item in reversed(raw_data)  # Reverse to get chronological order (oldest first)
            ]

            data = {
                "current_value": current_value,
                "current_classification": current_classification,
                "current_timestamp": current_timestamp,
                "historical": historical,  # Up to 365 days (1 year)
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

            OutputManager.log(f"✅ Fetched Fear & Greed Index: {current_value} ({current_classification})")
            return data

        except Exception as e:
            OutputManager.log(f"❌ Failed to fetch Fear & Greed Index: {e}")
            raise

    def render(self, processed_data: Dict[str, Any]) -> str:
        """Render Fear & Greed Index widget HTML."""
        current_value = processed_data["current_value"]
        current_classification = processed_data["current_classification"]
        historical = processed_data["historical"]
        timestamp_iso = processed_data["fetched_at"]

        # Prepare chart data for different timeframes
        # Filter historical data for different periods
        historical_7d = historical[-7:] if len(historical) >= 7 else historical
        historical_30d = historical[-30:] if len(historical) >= 30 else historical
        historical_90d = historical[-90:] if len(historical) >= 90 else historical
        historical_1y = historical  # All data (up to 365 days)

        # Determine color based on value
        # 0-24: Extreme Fear (red)
        # 25-44: Fear (orange)
        # 45-54: Neutral (yellow)
        # 55-74: Greed (light green)
        # 75-100: Extreme Greed (green)
        if current_value <= 24:
            gauge_color = "#ef4444"  # Red
        elif current_value <= 44:
            gauge_color = "#f97316"  # Orange
        elif current_value <= 54:
            gauge_color = "#eab308"  # Yellow
        elif current_value <= 74:
            gauge_color = "#84cc16"  # Light green
        else:
            gauge_color = "#22c55e"  # Green

        return self.render_template(
            "widgets/crypto_fear_greed.html",
            current_value=current_value,
            current_classification=current_classification,
            gauge_color=gauge_color,
            historical_7d=historical_7d,
            historical_30d=historical_30d,
            historical_90d=historical_90d,
            historical_1y=historical_1y,
            timestamp_iso=timestamp_iso
        )
