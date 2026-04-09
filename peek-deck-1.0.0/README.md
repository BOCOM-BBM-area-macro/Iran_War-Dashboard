# PeekDeck

<div align="center">

<img src="readme-banner.png" alt="PeekDeck - A glance is all you need" width="600">

*A configurable, widget-based monitoring system that generates static dashboards for tracking various data sources.*

[Live Demo](https://peekdeck.ruidiao.dev/) • [Documentation](#documentation) • [Quick Start](#quick-start) • [Contributing](#contributing)

</div>

---

## What is PeekDeck?

PeekDeck is a flexible dashboard generator that transforms YAML configurations into beautiful, static HTML dashboards. Perfect for monitoring crypto prices, tech news, GitHub trends, and more - all without requiring a backend server.

**Key Features:**

- **Widget-Based Architecture** - Compose dashboards from reusable widgets (price charts, news feeds, Reddit posts, etc.)
- **Static Generation** - Pure HTML/CSS/JS output, deployable anywhere (GitHub Pages, Netlify, S3)
- **Smart Caching** - Respects API rate limits with intelligent update scheduling
- **Multi-Page Support** - Create unlimited dashboards organized by category
- **Customizable Themes** - Per-page color schemes and styling
- **GitHub Actions Ready** - Automated updates via scheduled workflows

## Live Examples

**[View Live Demo →](https://peekdeck.ruidiao.dev/)**

- [Bitcoin Dashboard](https://peekdeck.ruidiao.dev/bitcoin.html) - Real-time crypto prices, charts, market stats, news, and community discussions
- [Ethereum Dashboard](https://peekdeck.ruidiao.dev/ethereum.html) - Live Ethereum monitoring with price charts and news
- [AI News](https://peekdeck.ruidiao.dev/ai.html) - Latest developments in artificial intelligence
- [Robotics](https://peekdeck.ruidiao.dev/robotics.html) - Robotics research and industry news

## Quick Start

### Prerequisites

- Python 3.11+
- pip
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/diaorui/peek-deck.git
cd peek-deck

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
./run.sh all

# View your dashboards
open docs/index.html
```

### Create Your First Dashboard

Create a new YAML file in `pages/`:

```yaml
# pages/solana.yaml
category: crypto
id: solana
name: Solana
description: Live Solana monitoring
icon: "◎"
enabled: true

theme:
  primary_color: "#14F195"
  background: "#1a1a1a"
  text_color: "#ffffff"
  card_background: "#2d2d2d"
  border_radius: "8px"

params:
  symbol: SOLUSDT
  coin_id: solana

widgets:
  - type: crypto-price
    update_minutes: 5

  - type: crypto-price-chart
    update_minutes: 10

  - type: crypto-market-stats
    update_minutes: 30

  - type: google-news
    update_minutes: 30
    params:
      query: solana
      limit: 10
```

Run the pipeline:

```bash
./run.sh all
open docs/solana.html
```

## Available Widgets

### Crypto
- **crypto-price** - Current price display
- **crypto-price-chart** - Historical price charts with multiple timeframes
- **crypto-market-stats** - Market cap, supply, all-time high/low
- **crypto-fear-greed** - Fear & Greed Index with historical chart

### News & Discussion
- **google-news** - Latest news articles with thumbnails
- **reddit-posts** - Recent posts from any subreddit
- **hackernews-posts** - HackerNews stories by keyword
- **youtube-videos** - YouTube search results

### Tech & AI
- **github-repos** - Trending GitHub repositories
- **huggingface-models** - Trending ML models with AI-generated descriptions
- **huggingface-papers** - Latest research papers

## How It Works

PeekDeck uses a 3-stage pipeline:

1. **Fetch** - Gets data from APIs
2. **Process** - Transforms data (e.g., LLM summarization, calculations)
3. **Render** - Generates static HTML

Each widget has an `update_minutes` setting to control refresh frequency and respect API rate limits.

## Deployment

### GitHub Pages (Recommended)

**1. Fork/Clone this repository**

**2. Configure GitHub Secrets** (Optional)

Go to **Settings** → **Secrets and variables** → **Actions** and add:
- `YOUTUBE_API_KEY` - For YouTube widget (if using youtube-videos)
- `GEMINI_API_KEY` - For AI-generated descriptions in HuggingFace models widget (optional)

**3. Run the Workflow First**

- Go to **Actions** → **Update Widgets**
- Click **Run workflow**
- Wait for it to complete (this creates the `gh-pages` branch)

**4. Enable GitHub Pages**

- Go to **Settings** → **Pages**
- Set **Source** to `gh-pages` branch
- Save

Your dashboards will be live at:
```
https://YOUR_USERNAME.github.io/peek-deck/
```

The workflow runs every 5 minutes, automatically updating widgets based on their configured `update_minutes`.

**Note:** GitHub Pages is the recommended deployment method because it integrates seamlessly with GitHub Actions for automated scheduled updates. Other static hosting platforms would require you to set up your own cron jobs or manual updates.

## Project Structure

- `pages/` - Page configurations (YAML)
- `src/peek_deck/` - Python source (core, widgets, pipeline stages)
- `templates/` - Jinja2 templates
- `data/` - Generated data (cached, gitignored)
- `docs/` - Generated HTML (deployed to GitHub Pages)

## Configuration

Pages are defined by YAML files in `pages/`. Each page has:
- Metadata (id, name, icon, category)
- Theme colors
- Widget list with update frequencies

See [pages/](pages/) for examples.

Configure the index page in `config/index.yaml` (base URL, GitHub URL, analytics, SEO).

## Extending PeekDeck

To add custom widgets, subclass `BaseWidget` and implement `fetch_data()`, `process_data()`, and `render()` methods. See existing widgets in `src/peek_deck/widgets/` for examples.

## API Keys & Rate Limits

### APIs Used (All Free)

| API | Widgets | Free Tier | Auth Required |
|-----|---------|-----------|---------------|
| **Gemini Exchange** | crypto-price | 120 req/min | No |
| **Binance US** | crypto-price-chart | 6,000 req/min | No |
| **CoinGecko** | crypto-market-stats | 5-30 calls/min | No |
| **Alternative.me** | crypto-fear-greed | 60 req/min | No |
| **Google News RSS** | google-news | Undocumented | No |
| **Reddit RSS** | reddit-posts | Undocumented | No |
| **HackerNews Algolia** | hackernews-posts | Undocumented | No |
| **GitHub API** | github-repos | 60/hour (1k in Actions) | Auto in Actions |
| **Hugging Face Hub API** | huggingface-models, huggingface-papers | Unclear | No |
| **YouTube Data API** | youtube-videos | 10k units/day | **Yes - API key needed** |
| **Google Gemini API** | huggingface-models (AI descriptions) | 1000/day (free tier) | Optional - API key needed |

### Environment Variables

For local development, set environment variables before running:

```bash
export YOUTUBE_API_KEY=your_key_here
export GEMINI_API_KEY=your_key_here  # Optional - for AI-generated model descriptions
./run.sh all
```

To enable AI-generated descriptions for HuggingFace models, also configure `llm` in `config/index.yaml`:

```yaml
llm:
  provider: gemini
  model: gemini-2.5-flash-lite
```

**Note:** When running in GitHub Actions, set secrets in repository settings. The workflow automatically uses `GITHUB_TOKEN` for GitHub API (1,000 requests/hour vs 60 unauthenticated).

## Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute

- **Add new widgets** - Implement widgets for new data sources (stocks, weather, sports, etc.)
- **Improve existing widgets** - Enhance features, fix bugs, improve styling
- **Share your dashboards** - Create interesting page configurations and share them
- **Report issues** - Found a bug? [Open an issue](https://github.com/diaorui/peek-deck/issues)
- **Improve documentation** - Fix typos, add examples, clarify instructions

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/peek-deck.git
cd peek-deck

# Install dependencies
pip install -r requirements.txt

# Make changes and test
./run.sh all
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-widget`)
3. Make your changes
4. Test locally (`./run.sh all`)
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

## Community & Support

- **Issues**: [GitHub Issues](https://github.com/diaorui/peek-deck/issues) - Report bugs or request features
- **Discussions**: [GitHub Discussions](https://github.com/diaorui/peek-deck/discussions) - Ask questions, share ideas
- **Examples**: Check the [pages/](pages/) folder for configuration examples

## License

MIT License - Free to use, modify, and distribute.

---

<div align="center">

**Built with ❤️ for the open-source community**

[⬆ Back to Top](#peekdeck)

</div>
