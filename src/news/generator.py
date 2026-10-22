"""
AI News Generator using configurable LLM providers
"""
from typing import List, Optional, Dict
import json
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from ..logger import setup_logger
from ..config import LANGUAGE_NAMES
from .web_search import WebSearchTool, get_search_tool_definition
from .fetcher import NewsFetcher
from ..llm_providers import get_llm_provider
from ..llm_providers.base_provider import call_with_retry


logger = setup_logger(__name__)


class NewsGenerator:
    """Generate AI news digest using configurable LLM providers"""

    def __init__(
        self,
        provider_name: str = "claude",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        enable_web_search: bool = False,
        metrics=None
    ):
        """
        Initialize the NewsGenerator.

        Args:
            provider_name: Name of LLM provider to use ('claude' or 'deepseek')
            api_key: API key for the provider. If None, will read from environment
            model: Model name to use. If None, uses provider's default model
            enable_web_search: Whether to enable web search tool for fetching current news

        Raises:
            ValueError: If provider is not recognized or API key is not provided
        """
        # Initialize LLM provider
        self.provider = get_llm_provider(
            provider_name=provider_name,
            api_key=api_key,
            model=model
        )

        self.enable_web_search = enable_web_search
        self.search_tool = WebSearchTool() if enable_web_search else None
        self._metrics = metrics
        self.news_fetcher = NewsFetcher(metrics=metrics)
        logger.info(
            f"NewsGenerator initialized with {self.provider.provider_name} "
            f"(model: {self.provider.model}, web_search: {enable_web_search})"
        )

    def _format_news_with_ids(self, news_data: Dict) -> tuple:
        """
        Format news with unique IDs for selection stage.

        Args:
            news_data: Dictionary with 'international' and 'domestic' news lists

        Returns:
            Tuple of (formatted_text, news_items_dict)
        """
        formatted = "# Recent AI News Items for Selection\n\n"
        news_items = {}  # id -> full news item
        item_id = 1

        if news_data['international']:
            formatted += "## International News\n\n"
            for item in news_data['international']:
                news_id = f"INT-{item_id}"
                news_items[news_id] = item

                formatted += f"### [{news_id}] {item['title']}\n"
                formatted += f"**Source:** {item['source']}\n"
                if item['description']:
                    formatted += f"**Description:** {item['description'][:400]}...\n"
                if item['published']:
                    formatted += f"**Published:** {item['published']}\n"
                formatted += "\n"
                item_id += 1

        if news_data['domestic']:
            formatted += "## Domestic News\n\n"
            item_id = 1
            for item in news_data['domestic']:
                news_id = f"DOM-{item_id}"
                news_items[news_id] = item

                formatted += f"### [{news_id}] {item['title']}\n"
                formatted += f"**Source:** {item['source']}\n"
                if item['description']:
                    formatted += f"**Description:** {item['description'][:400]}...\n"
                if item['published']:
                    formatted += f"**Published:** {item['published']}\n"
                formatted += "\n"
                item_id += 1

        return formatted, news_items

    def _deduplicate_news(self, news_items: List[Dict]) -> List[Dict]:
        """
        Remove duplicate news items that different sources report about the
        same story.

        Two items are duplicates when they share the same link, or when their
        titles are highly similar (SequenceMatcher ratio > 0.85 on the
        normalized titles). For title duplicates the copy with the longer
        description is kept, in the original item's position. Items with
        different headlines are never removed, so distinct stories survive.

        Args:
            news_items: News items with 'title', 'link' and 'description'

        Returns:
            Deduplicated list, in the original order
        """
        from difflib import SequenceMatcher

        def normalize(title: str) -> str:
            return ' '.join((title or '').lower().split())

        unique_news: List[Dict] = []
        seen_links = set()

        for item in news_items:
            link = (item.get('link') or '').strip()

            # 1. Exact link dedup (empty links are not treated as duplicates)
            if link and link in seen_links:
                continue

            # 2. Title similarity dedup
            title = normalize(item.get('title', ''))
            replaced = False
            is_duplicate = False
            for idx, existing in enumerate(unique_news):
                if not title or not normalize(existing.get('title', '')):
                    continue
                similarity = SequenceMatcher(
                    None, title, normalize(existing.get('title', ''))
                ).ratio()
                if similarity > 0.85:
                    is_duplicate = True
                    # Keep the copy with the more detailed description
                    if len(item.get('description', '') or '') > len(existing.get('description', '') or ''):
                        old_link = (existing.get('link') or '').strip()
                        if old_link:
                            seen_links.discard(old_link)
                        unique_news[idx] = item
                        if link:
                            seen_links.add(link)
                        replaced = True
                    break

            if not is_duplicate:
                unique_news.append(item)
                if link:
                    seen_links.add(link)

        if len(unique_news) != len(news_items):
            logger.info(f"Deduplication: {len(news_items)} \u2192 {len(unique_news)} items")
        return unique_news

    def _parse_selection_response(self, response: str, news_items: dict) -> List[str]:
        """
        Parse the Stage 1 selection response into a list of news IDs.

        Three-tier fallback strategy, so a usable selection is returned no
        matter how the LLM formats its answer:
          1. Standard JSON array parsing (preferred)
          2. Regex extraction of INT-n / DOM-n IDs from free text
          3. Heuristic scoring selection (see _heuristic_selection)

        Tiers 1-2 are normalized to 15-20 items (see _normalize_selection);
        tier 3 targets 18 items directly.

        Args:
            response: Raw Stage 1 LLM response text
            news_items: Mapping of news ID -> news item

        Returns:
            Selected news IDs (15-20, or all items if fewer exist)
        """
        # Tier 1: standard JSON parsing
        json_match = re.search(r'\[[\s\S]*?\]', response or '')
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, list):
                    valid_ids = [i for i in parsed if isinstance(i, str) and i in news_items]
                    if valid_ids:
                        logger.info("Stage 1 parsing: JSON")
                        return self._normalize_selection(valid_ids, news_items)
            except json.JSONDecodeError:
                pass

        # Tier 2: regex extraction of IDs mentioned anywhere in the text
        found_ids = re.findall(r'(INT-\d+|DOM-\d+)', response or '')
        valid_ids = [i for i in found_ids if i in news_items]
        if valid_ids:
            logger.info("Stage 1 parsing: regex extraction")
            return self._normalize_selection(valid_ids, news_items)

        # Tier 3: heuristic selection
        logger.warning("Stage 1 parsing: heuristic selection (no IDs found in response)")
        return self._heuristic_selection(news_items, target_count=18)

    def _normalize_selection(self, selected_ids: List[str], news_items: dict) -> List[str]:
        """
        Normalize a parsed selection: deduplicate (keeping order), pad to
        18 items when fewer than 15 were selected, trim to at most 20.
        """
        seen = set()
        unique_ids = []
        for news_id in selected_ids:
            if news_id not in seen:
                seen.add(news_id)
                unique_ids.append(news_id)

        if len(unique_ids) < 15:
            logger.warning(f"Only {len(unique_ids)} items selected, adding more")
            remaining = [i for i in news_items.keys() if i not in seen]
            unique_ids.extend(remaining[:18 - len(unique_ids)])
        elif len(unique_ids) > 20:
            logger.warning(f"{len(unique_ids)} items selected, trimming to 20")
            unique_ids = unique_ids[:20]

        return unique_ids

    def _heuristic_selection(self, news_items: dict, target_count: int = 18) -> List[str]:
        """
        Score-based fallback selection used when the Stage 1 response
        contains no parseable IDs: longer descriptions, high-value
        keywords in the title, and official company blogs score higher.
        """
        high_value_keywords = [
            'breakthrough', 'release', 'launch', 'funding',
            'acquisition', 'open-source', 'open source',
        ]
        premium_sources = [
            'OpenAI Blog', 'Google AI Blog', 'DeepMind Blog', 'Meta AI Blog',
        ]

        scored_items = []
        for news_id, item in news_items.items():
            score = 0.0
            # More detailed news scores higher
            score += len(item.get('description', '') or '') * 0.001
            # High-value keywords in the title
            title_lower = (item.get('title', '') or '').lower()
            score += sum(10 for kw in high_value_keywords if kw in title_lower)
            # Official sources are more authoritative
            if item.get('source') in premium_sources:
                score += 20
            scored_items.append((score, news_id))

        # Highest scores first; ties keep the original order
        scored_items.sort(key=lambda pair: pair[0], reverse=True)
        selected = [news_id for _, news_id in scored_items[:target_count]]

        int_count = sum(1 for i in selected if i.startswith('INT-'))
        dom_count = sum(1 for i in selected if i.startswith('DOM-'))
        logger.info(f"Heuristic selection: {int_count} international, {dom_count} domestic")
        return selected

    # Source quality tiers for pre-scoring: tier1 = official company blogs,
    # tier2 = major tech media and research feeds, tier3 = other majors.
    # A tier entry matches a source name exactly or as a prefix, so the
    # single entry 'arXiv' covers 'arXiv AI', 'arXiv NLP', and so on.
    _SOURCE_TIERS = (
        (0.3, ('OpenAI Blog', 'Google AI Blog', 'DeepMind Blog', 'Meta AI Blog', 'Microsoft AI Blog')),
        (0.2, ('TechCrunch AI', 'VentureBeat AI', 'MIT Technology Review', 'arXiv')),
        (0.1, ('The Verge AI', 'Ars Technica AI', 'Wired AI')),
    )

    def _score_news_item(self, item: Dict) -> float:
        """
        Score a news item's quality from 0 to 1 before Stage 1 selection.

        Dimensions: keyword weight (max 0.3), source tier (max 0.3),
        content quality (max 0.2), and freshness (max 0.2). Rule-based
        and cheap, so low-quality items can be dropped before any LLM
        tokens are spent on them.

        Args:
            item: News item with title, description, source, published

        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0

        # 1. Keyword weight (title only)
        title_lower = (item.get('title') or '').lower()
        high_value_keywords = {
            'breakthrough', 'release', 'launch', 'funding', 'acquisition',
            'open-source', 'announce', 'introduce', 'unveil',
        }
        tech_keywords = {
            'gpt', 'claude', 'llm', 'transformer', 'agent', 'multimodal',
            'reasoning', 'benchmark', 'sota', 'api',
        }
        kw_score = sum(0.05 for kw in high_value_keywords if kw in title_lower)
        kw_score += sum(0.03 for kw in tech_keywords if kw in title_lower)
        score += min(kw_score, 0.3)

        # 2. Source tier (exact name match; the 'arXiv' entry covers all
        #    arXiv feeds such as 'arXiv AI' and 'arXiv NLP' by prefix)
        source = item.get('source') or ''
        for tier_score, names in self._SOURCE_TIERS:
            if any(
                source == name or (name == 'arXiv' and source.startswith('arXiv'))
                for name in names
            ):
                score += tier_score
                break

        # 3. Content quality: description depth and concrete numbers
        description = item.get('description') or ''
        desc_len = len(description)
        if desc_len > 500:
            score += 0.1
        elif desc_len > 300:
            score += 0.05
        if re.search(r'\d+%|\d+x|\$\d+[MB]|\d+\s*(billion|million)', description.lower()):
            score += 0.1

        # 4. Freshness based on the published timestamp
        score += self._freshness_score(item.get('published') or '')

        return min(score, 1.0)

    @staticmethod
    def _freshness_score(published: str) -> float:
        """
        Map an RSS/Atom published timestamp to a freshness score:
        under 6h -> 0.2, under 24h -> 0.15, under 3d -> 0.1,
        under 7d -> 0.05, older or unparseable -> 0.
        """
        if not published:
            return 0.0
        published_dt = None
        try:
            published_dt = parsedate_to_datetime(published)
        except (TypeError, ValueError):
            published_dt = None
        if published_dt is None:
            try:
                published_dt = datetime.fromisoformat(published.replace('Z', '+00:00'))
            except ValueError:
                return 0.0
        if published_dt.tzinfo is None:
            published_dt = published_dt.replace(tzinfo=timezone.utc)

        age_hours = (datetime.now(timezone.utc) - published_dt).total_seconds() / 3600
        if age_hours <= 6:
            return 0.2
        if age_hours <= 24:
            return 0.15
        if age_hours <= 72:
            return 0.1
        if age_hours <= 168:
            return 0.05
        return 0.0

    def _filter_by_quality(
        self,
        news_items: List[Dict],
        min_score: float = 0.3,
        min_keep: int = 25
    ) -> List[Dict]:
        """
        Drop low-quality items before Stage 1 and sort the rest by score.

        The score is attached to each kept item as 'quality_score'.
        Safeguard: if filtering would leave fewer than min_keep items,
        the top-scoring min_keep items are kept instead, so Stage 1
        always has enough candidates even on a slow news day.

        Args:
            news_items: Items to filter
            min_score: Minimum quality score to keep an item
            min_keep: Minimum number of items to return

        Returns:
            Kept items, best first
        """
        scored = [(self._score_news_item(item), item) for item in news_items]
        above_threshold = sum(1 for s, _ in scored if s >= min_score)
        kept = [(s, item) for s, item in scored if s >= min_score]

        if len(kept) < min_keep:
            scored.sort(key=lambda pair: pair[0], reverse=True)
            kept = scored[:min_keep]
            logger.info(
                f"Quality filtering safeguard: only {above_threshold} items "
                f"scored >= {min_score}, keeping top {len(kept)} by score"
            )

        kept.sort(key=lambda pair: pair[0], reverse=True)
        for s, item in kept:
            item['quality_score'] = round(s, 3)

        logger.info(
            f"Quality filtering: {len(news_items)} \u2192 {len(kept)} items (threshold: {min_score})"
        )
        logger.debug(
            f"Score distribution: {self._score_distribution([s for s, _ in scored])}"
        )
        return [item for _, item in kept]

    @staticmethod
    def _score_distribution(scores: List[float]) -> Dict[str, int]:
        """Bucket scores into ranges for logging."""
        bins = {'0-0.3': 0, '0.3-0.5': 0, '0.5-0.7': 0, '0.7-1.0': 0}
        for s in scores:
            if s < 0.3:
                bins['0-0.3'] += 1
            elif s < 0.5:
                bins['0.3-0.5'] += 1
            elif s < 0.7:
                bins['0.5-0.7'] += 1
            else:
                bins['0.7-1.0'] += 1
        return bins

    def _call_llm(self, messages, max_tokens: int) -> str:
        """
        Call the LLM provider, retrying transient failures (connection
        errors, timeouts, rate limits, HTTP 429/5xx) with backoff.
        """
        return call_with_retry(
            lambda: self.provider.generate(messages=messages, max_tokens=max_tokens),
            logger=logger,
        )

    def _generate_simplified_digest(self, news_data: Dict) -> str:
        """
        Degraded fallback used when two-stage generation fails: a plain
        list of titles and links (best items first, since the lists were
        already deduplicated and quality-filtered), so subscribers still
        receive something usable.
        """
        digest = "# AI News Digest (Simplified)\n\n"
        digest += "_The full AI-generated digest was unavailable this run; " \
                  "here are the top stories with links._\n\n"

        for category in ('international', 'domestic'):
            if news_data.get(category):
                digest += f"## {category.title()} News\n\n"
                for i, item in enumerate(news_data[category][:20], 1):
                    digest += f"{i}. [{item['title']}]({item['link']})\n"
                    digest += f"   Source: {item['source']}\n\n"

        return digest

    def _two_stage_digest(
        self,
        news_data: Dict,
        max_tokens: int,
        language: str,
        stage1_template: Optional[str],
        stage2_template: Optional[str]
    ) -> tuple:
        """
        Two-stage prompt chaining: Stage 1 selects, Stage 2 summarizes.

        Returns (digest_text, selected_count). Raises on failure; the
        caller decides whether to degrade to the simplified digest.
        """
        # Format news with unique IDs for selection
        formatted_news, news_items = self._format_news_with_ids(news_data)
        total_items = len(news_items)

        logger.info(f"Starting two-stage prompt chaining with {total_items} news items")

        # ============================================================
        # STAGE 1: Selection - Analyze and select 15-20 best items
        # ============================================================
        logger.info(f"Stage 1: Analyzing and selecting high-quality news items...")

        # Use provided template or load from config
        if stage1_template is None:
            from ..config import Config
            config = Config()
            stage1_template = config.stage1_prompt_template

        # Format Stage 1 prompt with placeholders
        selection_prompt = stage1_template.format(
            formatted_news=formatted_news,
            total_items=total_items
        )

        messages = [{"role": "user", "content": selection_prompt}]
        stage_started = time.perf_counter()
        selection_response = self._call_llm(
            messages=messages,
            max_tokens=4000 # give enough tokens for selection
        )
        if self._metrics is not None:
            self._metrics.record_llm_stage(
                1, time.perf_counter() - stage_started,
                getattr(self.provider, 'last_usage', None),
            )

        # Parse selected IDs (three-tier fallback: JSON -> regex -> heuristic)
        selected_ids = self._parse_selection_response(selection_response, news_items)
        if self._metrics is not None:
            self._metrics.record_news_counts(selected=len(selected_ids))

        logger.info(f"Stage 1 completed: Selected {len(selected_ids)} news items")
        logger.debug(f"Selected IDs: {selected_ids}")

        # ============================================================
        # STAGE 2: Summarization - Create detailed summaries
        # ============================================================
        logger.info(f"Stage 2: Creating detailed summaries for selected items...")

        # Format selected news for summarization
        formatted_selected = "# Selected High-Quality AI News Items\n\n"
        for news_id in selected_ids:
            item = news_items[news_id]
            formatted_selected += f"### [{news_id}] {item['title']}\n"
            formatted_selected += f"**Source:** {item['source']}\n"
            if item['description']:
                formatted_selected += f"**Content:** {item['description']}\n"
            formatted_selected += f"**Link:** {item['link']}\n"
            if item['published']:
                formatted_selected += f"**Published:** {item['published']}\n"
            formatted_selected += "\n"

        # Use provided template or load from config
        if stage2_template is None:
            from ..config import Config
            config = Config()
            stage2_template = config.stage2_prompt_template

        # Format Stage 2 prompt with placeholders
        summarization_prompt = stage2_template.format(
            count=len(selected_ids),
            selected_news=formatted_selected
        )

        # Add language instruction if not English
        if language and language.lower() != "en":
            language_name = LANGUAGE_NAMES.get(language.lower(), language.upper())
            summarization_prompt += f"\n\nIMPORTANT: Please respond entirely in {language_name}."

        # Execute Stage 2: Generate detailed summaries
        messages = [{"role": "user", "content": summarization_prompt}]
        stage_started = time.perf_counter()
        response_text = self._call_llm(
            messages=messages,
            max_tokens=max_tokens
        )
        if self._metrics is not None:
            self._metrics.record_llm_stage(
                2, time.perf_counter() - stage_started,
                getattr(self.provider, 'last_usage', None),
            )

        # Add footer with GitHub link
        footer = "\n\n---\n\n*Generated by [AI News Bot](https://github.com/giftedunicorn/ai-news-bot) - Your AI-powered news assistant*"
        response_text += footer

        logger.info("Stage 2 completed: News digest generated successfully")
        logger.info(f"Two-stage prompt chaining completed: {total_items} items → {len(selected_ids)} selected → full digest")
        logger.debug(f"Response length: {len(response_text)} characters")

        return response_text, len(selected_ids)

    def generate_news_digest_from_sources(
        self,
        max_tokens: int = 8000,
        language: str = "en",
        max_items_per_source: int = 5,
        stage1_template: Optional[str] = None,
        stage2_template: Optional[str] = None
    ) -> str:
        """
        Fetch real-time news and generate a digest using two-stage prompt chaining:
        Stage 1: Analyze and select 15-20 high-quality news items
        Stage 2: Create detailed summaries for selected items

        Args:
            max_tokens: Maximum tokens in response
            language: Language code for the response
            max_items_per_source: Maximum items to fetch per source
            stage1_template: Optional Stage 1 prompt template (from config)
            stage2_template: Optional Stage 2 prompt template (from config)

        Returns:
            Generated news digest as string

        Raises:
            Exception: If fetching fails. Generation failures degrade to
                a simplified digest instead of raising.
        """
        try:
            # Fetch real-time news. When incremental mode is enabled and a
            # previous run exists, only items newer than that run are
            # processed (with automatic fallback to a full fetch).
            logger.info("Fetching real-time AI news from sources...")
            incremental_since = self.news_fetcher.incremental_since()
            news_data = self.news_fetcher.fetch_news_since(
                since_time=incremental_since,
                language=language,
                max_items_per_source=max_items_per_source
            )

            fetched_total = len(news_data['international']) + len(news_data['domestic'])

            # Deduplicate: different sources often report the same story
            news_data['international'] = self._deduplicate_news(news_data['international'])
            news_data['domestic'] = self._deduplicate_news(news_data['domestic'])

            if self._metrics is not None:
                self._metrics.record_news_counts(
                    fetched=fetched_total,
                    deduplicated=len(news_data['international']) + len(news_data['domestic']),
                )

            # Pre-score and drop low-quality items before Stage 1 spends
            # LLM tokens analyzing them
            news_data['international'] = self._filter_by_quality(news_data['international'])
            news_data['domestic'] = self._filter_by_quality(news_data['domestic'])

            if not news_data['international'] and not news_data['domestic']:
                error_msg = "No news items fetched from RSS sources. Please check your network connection or RSS feed availability."
                logger.error(error_msg)
                raise Exception(error_msg)

            try:
                digest_text, selected_count = self._two_stage_digest(
                    news_data, max_tokens, language, stage1_template, stage2_template
                )
            except Exception as stage_error:
                logger.error(f"Two-stage generation failed: {stage_error}")
                logger.info("Falling back to simplified digest (titles and links only)")
                digest_text = self._generate_simplified_digest(news_data)
                selected_count = 0

            if self._metrics is not None:
                self._metrics.record_news_counts(final_chars=len(digest_text))

            # A digest was produced (full or simplified): advance the
            # incremental watermark so the next run only sees newer items
            self.news_fetcher.mark_incremental_run({
                'fetched': fetched_total,
                'selected': selected_count,
            })

            return digest_text

        except Exception as e:
            logger.error(f"Failed to generate news digest from sources: {str(e)}", exc_info=True)
            raise
