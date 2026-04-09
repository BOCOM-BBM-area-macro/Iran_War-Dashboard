"""CLI entry point for PeekDeck."""

import sys
from pathlib import Path

# Import package name from central config
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))
from project_config import PACKAGE_NAME


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(f"Usage: python -m {PACKAGE_NAME} <command>")
        print("\nCommands:")
        print("  fetch    - Stage 1: Fetch data from external APIs")
        print("  process  - Stage 2: Process raw data")
        print("  render   - Stage 3: Render HTML pages")
        print("  all      - Run all stages in sequence")
        sys.exit(1)

    command = sys.argv[1]

    if command == "fetch":
        from .fetch import fetch_all
        fetch_all()
    elif command == "process":
        from .process import process_all
        process_all()
    elif command == "render":
        from .render import render_all
        render_all()
    elif command == "all":
        from .fetch import fetch_all
        from .process import process_all
        from .render import render_all
        print("🚀 Running all stages...\n")
        fetch_all()
        process_all()
        render_all()
        print("✅ All stages completed!")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
