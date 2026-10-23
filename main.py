#!/usr/bin/env python3
"""
AI News Bot - Main Application

Generates and distributes daily AI news digests using Anthropic's Claude API.
"""
import sys
from datetime import datetime
from src.config import Config
from src.logger import setup_logger
from src.metrics import MetricsCollector
from src.news import NewsGenerator
from src.notifiers import (
    EmailNotifier,
    ResendNotifier,
    WebhookNotifier,
    SlackNotifier,
    TelegramNotifier,
    DiscordNotifier
)


logger = setup_logger(__name__)


# Exit codes (TODO #10): make the run result machine-readable for CI
EXIT_CODE_SUCCESS = 0           # everything succeeded
EXIT_CODE_PARTIAL_SUCCESS = 0   # >=70% of notifications sent (acceptable)
EXIT_CODE_PARTIAL_FAILURE = 1   # some sent, below the acceptable rate
EXIT_CODE_TOTAL_FAILURE = 2     # nothing could be sent (alert)


def compute_exit_code(total_attempts: int, success_count: int) -> int:
    """Map notification outcomes to an exit code."""
    if total_attempts <= 0:
        # Nothing was configured to be sent; reaching the end is success
        return EXIT_CODE_SUCCESS
    success_rate = success_count / total_attempts
    if success_rate >= 1.0:
        return EXIT_CODE_SUCCESS
    if success_rate >= 0.7:
        return EXIT_CODE_PARTIAL_SUCCESS
    if success_rate > 0:
        return EXIT_CODE_PARTIAL_FAILURE
    return EXIT_CODE_TOTAL_FAILURE


def save_digest_to_file(digest: str, language: str):
    """Back up a digest to .output/ when no notification could deliver it."""
    from pathlib import Path
    output_dir = Path(".output")
    output_dir.mkdir(exist_ok=True)
    filename = f"digest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{language}.md"
    output_file = output_dir / filename
    output_file.write_text(digest, encoding='utf-8')
    logger.info(f"Digest backed up to {output_file}")


def main():
    """Main application entry point"""
    try:
        # Load configuration
        config = Config()

        # Setup logger with config
        logger = setup_logger(
            "ai_news_bot",
            level=config.log_level,
            log_format=config.log_format
        )

        # Get list of languages to process
        languages = config.ai_response_languages
        
        logger.info("=" * 60)
        logger.info("AI News Bot Starting")
        logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"LLM Provider: {config.llm_provider}")
        if config.llm_model:
            logger.info(f"LLM Model: {config.llm_model}")
        logger.info(f"Languages: {', '.join(languages)}")
        logger.info(f"Web Search: {config.enable_web_search}")
        logger.info("=" * 60)

        # Performance metrics for this run (exported at the end)
        metrics = MetricsCollector()

        # Initialize news generator once
        logger.info("Initializing news generator...")
        news_gen = NewsGenerator(
            provider_name=config.llm_provider,
            api_key=config.llm_api_key,
            model=config.llm_model,
            enable_web_search=config.enable_web_search,
            metrics=metrics
        )

        # Get enabled notification methods
        notification_methods = config.notification_methods
        logger.info(f"Enabled notification methods: {notification_methods}")

        # Track overall results
        overall_results = {"sent": [], "failed": []}
        
        # Process each language
        for language in languages:
            logger.info("=" * 60)
            logger.info(f"Processing language: {language.upper()}")
            logger.info("=" * 60)

            try:
                # Generate news digest for this language
                logger.info(f"Generating AI news digest in {language.upper()} from real-time sources...")
                news_digest = news_gen.generate_news_digest_from_sources(
                    language=language,
                    max_items_per_source=config.max_items_per_source,
                    stage1_template=config.stage1_prompt_template,
                    stage2_template=config.stage2_prompt_template
                )

                logger.info(f"News digest generated for {language.upper()} ({len(news_digest)} characters)")
                logger.info("-" * 60)
                logger.info(f"News Digest Preview ({language.upper()}):")
                logger.info("-" * 60)
                # Print first 500 characters as preview
                preview = news_digest[:500] + "..." if len(news_digest) > 500 else news_digest
                logger.info(preview)
                logger.info("-" * 60)

                # Track notification results for this language
                lang_results = {"sent": [], "failed": []}

                # Send email notification if enabled
                if "email" in notification_methods:
                    logger.info(f"Sending email notification for {language.upper()}...")
                    email_notifier = EmailNotifier()
                    if email_notifier.send(news_digest, language=language):
                        lang_results["sent"].append("email")
                        logger.info(f"Email notification sent successfully for {language.upper()}")
                    else:
                        lang_results["failed"].append("email")
                        logger.warning(f"Email notification failed for {language.upper()}")

                # Send Resend (HTTPS email API) notification if enabled
                if "resend" in notification_methods:
                    logger.info(f"Sending Resend email notification for {language.upper()}...")
                    resend_notifier = ResendNotifier()
                    if resend_notifier.send(news_digest, language=language):
                        lang_results["sent"].append("resend")
                        logger.info(f"Resend notification sent successfully for {language.upper()}")
                    else:
                        lang_results["failed"].append("resend")
                        logger.warning(f"Resend notification failed for {language.upper()}")

                # Send webhook notification if enabled
                if "webhook" in notification_methods:
                    logger.info(f"Sending webhook notification for {language.upper()}...")
                    webhook_notifier = WebhookNotifier()
                    if webhook_notifier.send(news_digest, language=language):
                        lang_results["sent"].append("webhook")
                        logger.info(f"Webhook notification sent successfully for {language.upper()}")
                    else:
                        lang_results["failed"].append("webhook")
                        logger.warning(f"Webhook notification failed for {language.upper()}")

                # Send Slack notification if enabled
                if "slack" in notification_methods:
                    logger.info(f"Sending Slack notification for {language.upper()}...")
                    slack_notifier = SlackNotifier()
                    if slack_notifier.send(news_digest, language=language):
                        lang_results["sent"].append("slack")
                        logger.info(f"Slack notification sent successfully for {language.upper()}")
                    else:
                        lang_results["failed"].append("slack")
                        logger.warning(f"Slack notification failed for {language.upper()}")

                # Send Telegram notification if enabled
                if "telegram" in notification_methods:
                    logger.info(f"Sending Telegram notification for {language.upper()}...")
                    telegram_notifier = TelegramNotifier()
                    if telegram_notifier.send(news_digest, language=language):
                        lang_results["sent"].append("telegram")
                        logger.info(f"Telegram notification sent successfully for {language.upper()}")
                    else:
                        lang_results["failed"].append("telegram")
                        logger.warning(f"Telegram notification failed for {language.upper()}")

                # Send Discord notification if enabled
                if "discord" in notification_methods:
                    logger.info(f"Sending Discord notification for {language.upper()}...")
                    discord_notifier = DiscordNotifier()
                    if discord_notifier.send(news_digest, language=language):
                        lang_results["sent"].append("discord")
                        logger.info(f"Discord notification sent successfully for {language.upper()}")
                    else:
                        lang_results["failed"].append("discord")
                        logger.warning(f"Discord notification failed for {language.upper()}")

                # Update overall results
                for method in lang_results["sent"]:
                    metrics.record_notification(method, True)
                    result_key = f"{method} ({language.upper()})"
                    if result_key not in overall_results["sent"]:
                        overall_results["sent"].append(result_key)
                
                for method in lang_results["failed"]:
                    metrics.record_notification(method, False)
                    result_key = f"{method} ({language.upper()})"
                    if result_key not in overall_results["failed"]:
                        overall_results["failed"].append(result_key)

                # If every notification for this language failed, keep a
                # local copy of the digest so the run is not lost
                if notification_methods and not lang_results["sent"]:
                    save_digest_to_file(news_digest, language)

                logger.info(f"Language {language.upper()} completed successfully")

            except Exception as lang_error:
                logger.error(f"Error processing language {language.upper()}: {str(lang_error)}", exc_info=True)
                # Mark all notification methods as failed for this language
                for method in notification_methods:
                    result_key = f"{method} ({language.upper()})"
                    if result_key not in overall_results["failed"]:
                        overall_results["failed"].append(result_key)

        # Export performance metrics for this run
        metrics.export_metrics()

        # Final Summary
        logger.info("=" * 60)
        logger.info("AI News Bot Completed")
        logger.info(f"Processed {len(languages)} language(s): {', '.join(lang.upper() for lang in languages)}")
        logger.info(f"Successfully sent: {', '.join(overall_results['sent']) if overall_results['sent'] else 'None'}")
        if overall_results["failed"]:
            logger.warning(f"Failed to send: {', '.join(overall_results['failed'])}")
        logger.info("=" * 60)

        # Exit code reflects the notification success rate (TODO #10)
        total_attempts = len(languages) * len(notification_methods)
        success_count = len(overall_results["sent"])
        exit_code = compute_exit_code(total_attempts, success_count)
        if total_attempts > 0:
            success_rate = success_count / total_attempts
            if exit_code == EXIT_CODE_TOTAL_FAILURE:
                logger.error("✗ Total failure - no notifications sent")
            elif exit_code == EXIT_CODE_PARTIAL_FAILURE:
                logger.warning(f"⚠ Partial failure ({success_rate:.0%} of notifications sent)")
            elif success_count < total_attempts:
                logger.info(f"⚠ Partial success ({success_rate:.0%} of notifications sent)")
            else:
                logger.info("✓ All notifications sent successfully")
        return exit_code

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
