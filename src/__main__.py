"""Entry point: uv run python -m src <command>"""
import sys
import fire
from src.cli import COMMANDS

try:
    fire.Fire(COMMANDS)
except (OSError, ValueError) as e:
    sys.exit(f"error: {e}")
except KeyboardInterrupt:
    sys.exit("interrupted")
