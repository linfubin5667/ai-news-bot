"""
RSS feed cache - Stores fetched feed items on disk for a short TTL.

Repeated runs within the TTL (30 minutes by default) reuse the cached
items instead of hitting the network again, which speeds up local
development and debugging. Cache files live in `.cache/` (gitignored).
"""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from ..logger import setup_logger


logger = setup_logger(__name__)


class NewsCache:
    """File-based cache for RSS feed items, one JSON file per feed."""

    def __init__(self, cache_dir: str = ".cache"):
        """Initialize the cache, creating the cache directory if needed."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_file(self, feed_url: str, max_items: int) -> Path:
        # max_items is part of the key: a fetch that asked for more items
        # must not be served a smaller cached result.
        cache_key = hashlib.md5(f"{feed_url}|{max_items}".encode()).hexdigest()
        return self.cache_dir / f"rss_{cache_key}.json"

    def get_rss_cache(
        self,
        feed_url: str,
        max_items: int = 10,
        max_age_minutes: int = 30
    ) -> Optional[List]:
        """
        Return cached items for a feed if a fresh entry exists.

        Returns None on a miss, an expired entry, or an unreadable cache
        file (a broken cache must never break fetching).
        """
        cache_file = self._cache_file(feed_url, max_items)
        if not cache_file.exists():
            return None

        try:
            cache_data = json.loads(cache_file.read_text(encoding='utf-8'))
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
        except Exception as e:
            logger.warning(f"Ignoring unreadable cache file {cache_file}: {str(e)}")
            return None

        if datetime.now() - cached_time < timedelta(minutes=max_age_minutes):
            logger.info(f"Cache hit: {feed_url}")
            return cache_data['items']

        logger.info(f"Cache expired: {feed_url}")
        return None

    def set_rss_cache(self, feed_url: str, items: List, max_items: int = 10):
        """Store fetched items for a feed. Failures are logged, not raised."""
        cache_file = self._cache_file(feed_url, max_items)
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'items': items,
        }
        try:
            cache_file.write_text(
                json.dumps(cache_data, ensure_ascii=False),
                encoding='utf-8'
            )
        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_file}: {str(e)}")
