"""Crypto price chart widget using Binance US candles API."""
from ..core.output_manager import OutputManager

from datetime import datetime, timezone
from typing import Any, Dict, List

from ..core.base_widget import BaseWidget
from ..core.url_fetch_manager import get_url_fetch_manager


class CryptoPriceChartWidget(BaseWidget):
    """Displays cryptocurrency price history chart using candlestick data.

    Required params:
        - symbol: Trading pair symbol (e.g., "BTCUSD", "ETHUSD")
        - tabs: List of tab configurations with interval, limit, and label
          Supported intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    """

    def get_required_params(self) -> list[str]:
        return ["symbol", "tabs"]

    def fetch_data(self) -> Dict[str, Any]:
        """Fetch candlestick data for all configured tabs."""
        self.validate_params()

        symbol = self.merged_params["symbol"].upper()
        tabs_config = self.merged_params["tabs"]
        client = get_url_fetch_manager()

        # Fetch data for all tabs
        all_tabs_data = []
        for tab_config in tabs_config:
            interval = tab_config.get("interval", "1d")
            limit = tab_config.get("limit", 30)
            label = tab_config.get("label", interval)

            tab_data = self._fetch_single_chart(symbol, interval, limit)
            tab_data["label"] = label
            all_tabs_data.append(tab_data)

        data = {
            "symbol": symbol,
            "tabs": all_tabs_data,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        OutputManager.log(f"✅ Fetched {len(all_tabs_data)} tabs for {symbol}")
        return data

    def _fetch_single_chart(self, symbol: str, interval: str, limit: int) -> Dict[str, Any]:
        """Fetch data for a single chart from Binance US."""
        client = get_url_fetch_manager()
        try:
            # Fetch candlestick data from Binance US API
            url = "https://api.binance.us/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            candles = client.get(url, params=params, response_type="json")

            # Binance returns candles in chronological order (oldest first)
            # Parse candles into structured format
            # Binance format: [OpenTime, Open, High, Low, Close, Volume, CloseTime, QuoteVolume, NumTrades, TakerBuyBase, TakerBuyQuote, Ignore]
            parsed_candles = []
            for candle in candles:
                parsed_candles.append({
                    "timestamp": candle[0],  # Open time in milliseconds
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5])
                })

            return {
                "interval": interval,
                "candles": parsed_candles,
            }

        except Exception as e:
            OutputManager.log(f"❌ Failed to fetch or parse candles for {symbol}: {e}")
            raise

    def render(self, processed_data: Dict[str, Any]) -> str:
        """Render tabbed price chart widget HTML with Chart.js."""
        symbol = processed_data["symbol"]
        tabs_data = processed_data["tabs"]
        fetched_at = processed_data["fetched_at"]

        # Format the symbol for better readability
        base_currency = symbol[:3]
        if base_currency == "BTC":
            display_name = "Bitcoin"
        elif base_currency == "ETH":
            display_name = "Ethereum"
        elif base_currency == "SOL":
            display_name = "Solana"
        else:
            display_name = base_currency

        # Calculate price changes for each tab first (needed for tab buttons)
        price_changes = []
        for tab_data in tabs_data:
            candles = tab_data["candles"]
            if len(candles) >= 2:
                first_close = candles[0]["close"]
                last_close = candles[-1]["close"]
                price_change_percent = ((last_close - first_close) / first_close * 100)
            else:
                price_change_percent = 0
            price_changes.append(price_change_percent)

        # Generate tab buttons with stacked label and price change
        tab_buttons = []
        for i, tab_data in enumerate(tabs_data):
            active_class = "active" if i == 0 else ""
            label = tab_data["label"]
            change = price_changes[i]
            change_sign = "+" if change >= 0 else ""
            change_color = "var(--color-positive)" if change >= 0 else "var(--color-negative)"
            change_text = f'{change_sign}{change:.1f}%'

            tab_buttons.append(
                f'<button class="chart-tab-btn {active_class}" data-tab="tab-{i}">'
                f'<span class="tab-label">{label}</span>'
                f'<span class="tab-change" style="color: {change_color};">{change_text}</span>'
                f'</button>'
            )

        # Generate tab contents with charts
        tab_contents = []
        chart_scripts = []

        for i, tab_data in enumerate(tabs_data):
            interval = tab_data["interval"]
            candles = tab_data["candles"]
            price_change_percent = price_changes[i]

            # Prepare OHLC data for Chart.js candlestick
            # Format: {x: timestamp, o: open, h: high, l: low, c: close}
            candlestick_data = []
            for candle in candles:
                candlestick_data.append({
                    "x": candle["timestamp"],
                    "o": candle["open"],
                    "h": candle["high"],
                    "l": candle["low"],
                    "c": candle["close"]
                })

            # Generate unique ID for this chart
            chart_id = f"chart-{symbol.lower()}-tab-{i}"

            # Show first tab, hide others
            display_style = "" if i == 0 else "display: none;"

            # Tab content HTML
            tab_content = f"""
        <div class="chart-tab-content" data-tab-id="tab-{i}" style="{display_style}">
            <div class="widget-body">
                <canvas id="{chart_id}"></canvas>
            </div>
            <div class="widget-footer">
                <small data-timestamp="{fetched_at}">Updated {fetched_at}</small>
            </div>
        </div>"""
            tab_contents.append(tab_content)

            # Chart.js script with candlestick chart
            import json
            candlestick_json = json.dumps(candlestick_data)

            chart_script = f"""
        <script>
        (function() {{
            const ctx = document.getElementById('{chart_id}').getContext('2d');
            const interval = '{interval}';
            const candleData = {candlestick_json};

            // Determine appropriate time unit based on interval suffix
            const lastChar = interval.slice(-1);
            let timeUnit;
            if (lastChar === 'd' || lastChar === 'w' || lastChar === 'M') {{
                timeUnit = 'day';
            }} else if (lastChar === 'h') {{
                timeUnit = 'hour';
            }} else {{
                timeUnit = 'minute';
            }}

            new Chart(ctx, {{
                type: 'candlestick',
                data: {{
                    datasets: [{{
                        label: '{symbol}',
                        data: candleData,
                        color: {{
                            up: '#10b981',
                            down: '#ef4444',
                            unchanged: '#9ca3af'
                        }},
                        borderColor: {{
                            up: '#10b981',
                            down: '#ef4444',
                            unchanged: '#9ca3af'
                        }}
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    plugins: {{
                        legend: {{
                            display: false
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false,
                            displayColors: false,
                            callbacks: {{
                                title: function(context) {{
                                    const ts = context[0].raw.x;
                                    const date = new Date(ts);
                                    if (interval === '1d' || interval === '3d' || interval === '1w' || interval === '1M') {{
                                        return date.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric', year: 'numeric' }});
                                    }} else if (interval === '2h' || interval === '4h' || interval === '6h' || interval === '8h' || interval === '12h') {{
                                        return date.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric' }}) + ' ' +
                                               date.toLocaleTimeString('en-US', {{ hour: '2-digit', minute: '2-digit', hour12: false }});
                                    }} else {{
                                        return date.toLocaleTimeString('en-US', {{ hour: '2-digit', minute: '2-digit', hour12: false }});
                                    }}
                                }},
                                label: function(context) {{
                                    const data = context.raw;
                                    const formatPrice = (val) => '$' + val.toLocaleString('en-US', {{
                                        minimumFractionDigits: 2,
                                        maximumFractionDigits: 2
                                    }});
                                    return [
                                        'Open:  ' + formatPrice(data.o),
                                        'High:  ' + formatPrice(data.h),
                                        'Low:   ' + formatPrice(data.l),
                                        'Close: ' + formatPrice(data.c)
                                    ];
                                }}
                            }}
                        }}
                    }},
                    scales: {{
                        y: {{
                            ticks: {{
                                color: '#9ca3af',
                                callback: function(value) {{
                                    return '$' + value.toLocaleString('en-US', {{
                                        minimumFractionDigits: 0,
                                        maximumFractionDigits: 0
                                    }});
                                }}
                            }},
                            grid: {{
                                color: 'rgba(255, 255, 255, 0.1)'
                            }}
                        }},
                        x: {{
                            type: 'time',
                            time: {{
                                unit: timeUnit,
                                displayFormats: {{
                                    minute: 'HH:mm',
                                    hour: 'HH:mm',
                                    day: 'MMM d'
                                }}
                            }},
                            ticks: {{
                                display: false
                            }},
                            grid: {{
                                color: 'rgba(255, 255, 255, 0.1)'
                            }}
                        }}
                    }}
                }}
            }});
        }})();
        </script>"""
            chart_scripts.append(chart_script)

        # Render using template
        return self.render_template(
            "widgets/crypto_price_chart.html",
            display_name=display_name,
            tab_buttons=tab_buttons,
            tab_contents=tab_contents,
            chart_scripts=chart_scripts
        )
