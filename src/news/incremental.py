"""
Incremental update support for the news fetcher.

When a bot runs frequently (hourly, or every few hours), re-processing
every item from every feed each time wastes LLM tokens on stories that
were already digested. IncrementalUpdater persists the last successful
run time in a small state file (.last_run); the fetcher then keeps only
items published after that time, falling back to a full fetch when too
few new items are available.
"""
import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional

from ..logger import setup_logger


logger = setup_logger(__name__)


def parse_pub_time(pub_string: str) -> Optional[datetime]:
    """
    Parse an RSS/Atom published timestamp into an aware datetime (UTC).

    Handles RFC 822 dates (RSS 2.0), ISO 8601 (Atom, with 'Z' or offset),
    and a plain 'YYYY-MM-DD HH:MM:SS' fallback. Returns None when the
    string cannot be parsed.
    """
    if not pub_string:
        return None

    parsed = None
    try:
        parsed = parsedate_to_datetime(pub_string)
    except (TypeError, ValueError):
        parsed = None
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(pub_string.replace('Z', '+00:00'))
        except ValueError:
            parsed = None
    if parsed is None:
        try:
            parsed = datetime.strptime(pub_string, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            logger.debug(f"Failed to parse time: {pub_string}")
            return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class IncrementalUpdater:
    """Persist and read the last successful run time and its stats."""

    def __init__(self, state_file: str = ".last_run"):
        self.state_file = Path(state_file)

    def get_last_run_time(self) -> Optional[datetime]:
        """Read the last run time, or None when missing/unreadable."""
        if not self.state_file.exists():
            return None

        try:
            data = json.loads(self.state_file.read_text(encoding='utf-8'))
            return datetime.fromisoformat(data['last_run_time'])
        except Exception as e:
            logger.warning(f"Failed to read last run time: {e}")
            return None

    def save_run_time(self, stats: dict):
        """Save the current time together with run statistics."""
        data = {
            'last_run_time': datetime.now(timezone.utc).isoformat(),
            'items_fetched': stats.get('fetched', 0),
            'items_selected': stats.get('selected', 0),
        }
        try:
            self.state_file.write_text(
                json.dumps(data, indent=2), encoding='utf-8'
            )
        except Exception as e:
            logger.warning(f"Failed to save last run time: {e}")
