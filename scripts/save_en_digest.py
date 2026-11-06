#!/usr/bin/env python3
"""Generate the English digest and save it as a dated markdown file.

Usage (from the repository root, with the LLM API key in the environment):
    python3 scripts/save_en_digest.py

Output: site/digests/YYYY-MM-DD.md  (date in Asia/Shanghai, the bot's day)

This script ONLY generates and saves. It sends no email and no other
notification, and it never touches the Chinese digest: the Chinese email
flow lives in main.py and is run separately by the `digest` job.

It reuses the bot's own Config and NewsGenerator, so the English digest
uses the same RSS feeds, the same stage 1 / stage 2 prompts from
config.yaml, and the same provider/model as the daily run. The only
difference is the language argument: "en", passed explicitly.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# The bot's package lives in src/ next to this script's repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("ENABLE_INCREMENTAL", "false")
os.environ.setdefault("ENABLE_RSS_CACHE", "false")


def main():
    from src.config import Config
    from src.logger import setup_logger
    from src.news import NewsGenerator

    config = Config()
    logger = setup_logger("save_en_digest", level=config.log_level,
                          log_format=config.log_format)

    today = datetime.now(ZoneInfo("Asia/Shanghai")).date()
    out_dir = Path("site/digests")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{today.isoformat()}.md"

    logger.info("Generating English digest for %s ...", today.isoformat())
    generator = NewsGenerator(
        provider_name=config.llm_provider,
        api_key=config.llm_api_key,
        model=config.llm_model,
        enable_web_search=config.enable_web_search,
    )
    digest = generator.generate_news_digest_from_sources(
        language="en",
        max_items_per_source=config.max_items_per_source,
        stage1_template=config.stage1_prompt_template,
        stage2_template=config.stage2_prompt_template,
    )
    if not digest or "##" not in digest:
        raise SystemExit("ERROR: generated digest looks empty or malformed; "
                         "not writing a file.")
    out_file.write_text(digest.rstrip() + "\n", encoding="utf-8")
    logger.info("Saved English digest: %s (%d characters)",
                out_file, len(digest))
    print(f"saved {out_file} ({len(digest)} characters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
