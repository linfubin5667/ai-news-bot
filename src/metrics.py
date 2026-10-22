"""
Performance metrics collection for AI News Bot.

A MetricsCollector is created in main.py and passed down to the news
fetcher and generator. Each stage records what it did (fetch durations,
LLM stage durations, news counts, notification results), and main.py
exports everything to .metrics/run.json at the end of the run so that
runs can be compared over time.

Recording is deliberately cheap (a few dict writes per operation) and
never raises: a metrics problem must not break a news run.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .logger import setup_logger


logger = setup_logger(__name__)


class MetricsCollector:
    """Collect performance metrics for one application run."""

    def __init__(self):
        self.metrics = {
            'rss_sources': [],
            'llm_stage1_time': 0.0,
            'llm_stage2_time': 0.0,
            'llm_tokens': {'input': 0, 'output': 0},
            'news_counts': {
                'fetched': 0,
                'deduplicated': 0,
                'selected': 0,
                'final_chars': 0,
            },
            'notifications': [],
        }

    def record_rss_fetch(self, source: str, duration: float, count: int, success: bool):
        """
        Record one RSS fetch attempt. A source that needed retries
        appears once per attempt, so flaky sources are visible.
        """
        self.metrics['rss_sources'].append({
            'source': source,
            'duration': round(duration, 4),
            'count': count,
            'success': success,
        })

    def record_llm_stage(self, stage: int, duration: float, tokens: dict = None):
        """
        Record one LLM stage call. Token counts are added when the
        provider reports them (tokens may be None or empty, in which case
        only the duration is kept).
        """
        key = f'llm_stage{stage}_time'
        self.metrics[key] = self.metrics.get(key, 0.0) + duration
        tokens = tokens or {}
        self.metrics['llm_tokens']['input'] += tokens.get('input', 0) or 0
        self.metrics['llm_tokens']['output'] += tokens.get('output', 0) or 0

    def record_news_counts(self, **counts):
        """Update news pipeline counts (fetched, deduplicated, selected, final_chars)."""
        for key, value in counts.items():
            if key in self.metrics['news_counts']:
                self.metrics['news_counts'][key] = value

    def record_notification(self, method: str, success: bool):
        """Record the result of one notification send."""
        self.metrics['notifications'].append({
            'method': method,
            'success': success,
        })

    def export_metrics(self, output_file: str = ".metrics/run.json") -> dict:
        """
        Export a summary plus the detailed metrics to a JSON file.

        Returns the summary dict. Export failures are logged, not raised.
        """
        sources = self.metrics['rss_sources']
        if sources:
            avg_duration = sum(s['duration'] for s in sources) / len(sources)
        else:
            avg_duration = 0.0

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_time': self.metrics['llm_stage1_time'] + self.metrics['llm_stage2_time'],
            'rss_summary': {
                'total_sources': len({s['source'] for s in sources}),
                'total_attempts': len(sources),
                'successful': sum(1 for s in sources if s['success']),
                'avg_duration': round(avg_duration, 4),
            },
            'llm_summary': {
                'total_tokens': self.metrics['llm_tokens']['input'] + self.metrics['llm_tokens']['output'],
                'estimated_cost': self._estimate_cost(),
            },
            'detailed_metrics': self.metrics,
        }

        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
            logger.info(f"Metrics exported to {output_file}")
        except Exception as e:
            logger.warning(f"Failed to export metrics to {output_file}: {str(e)}")

        return summary

    def _estimate_cost(self) -> float:
        """
        Estimate the LLM cost from recorded token counts.

        Uses Claude Sonnet pricing ($3/MTok input, $15/MTok output) as
        the reference rate. When providers do not report token usage,
        the recorded counts stay 0 and so does the estimate.
        """
        input_tokens = self.metrics['llm_tokens']['input']
        output_tokens = self.metrics['llm_tokens']['output']
        cost = (input_tokens / 1_000_000 * 3) + (output_tokens / 1_000_000 * 15)
        return round(cost, 4)
