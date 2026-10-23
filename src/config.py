"""
Configuration management for AI News Bot
"""
import os
import yaml
from typing import Dict, List, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from .logger import setup_logger


logger = setup_logger(__name__)


# Supported language codes and their display names
LANGUAGE_NAMES = {
    "zh": "Chinese (中文)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "ja": "Japanese (日本語)",
    "de": "German (Deutsch)",
    "ko": "Korean (한국어)",
    "pt": "Portuguese (Português)",
    "ru": "Russian (Русский)",
    "ar": "Arabic (العربية)",
    "hi": "Hindi (हिन्दी)",
    "it": "Italian (Italiano)",
    "nl": "Dutch (Nederlands)",
}


class Config:
    """Application configuration manager"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config.yaml file. If None, searches for it in default locations
        """
        # Load environment variables from .env file
        load_dotenv()

        # Find and load YAML config
        self.config_path = self._find_config_file(config_path)
        self.config_data = self._load_yaml_config()

        # Fail fast: validate required configuration before doing any work
        self._validate_config()

        logger.info(f"Configuration loaded from {self.config_path}")

    # Environment variable that holds the API key for each supported provider
    _PROVIDER_KEY_ENV = {
        "claude": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "grok": "XAI_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    def _validate_config(self):
        """
        Validate that all required configuration is present at startup.

        Collects every problem and raises a single ValueError listing them
        all, so the user can fix everything in one pass instead of
        discovering missing items one run at a time.

        Raises:
            ValueError: If any required configuration is missing or invalid
        """
        errors = []

        # 1. LLM provider and API key
        provider = self.llm_provider
        key_env = self._PROVIDER_KEY_ENV.get(provider)
        if key_env is None:
            errors.append(
                f"Unsupported LLM provider '{provider}'. "
                f"Supported providers: {', '.join(sorted(self._PROVIDER_KEY_ENV))}. "
                "Set LLM_PROVIDER in environment or 'llm.provider' in config.yaml."
            )
        elif not self.llm_api_key:
            errors.append(
                f"Missing API key for provider '{provider}'. "
                f"Set {key_env} in environment."
            )

        # 2. Notification methods and their credentials
        methods = self.notification_methods
        if not methods:
            errors.append(
                "No notification methods configured. "
                "Set NOTIFICATION_METHODS in environment (e.g., 'email,slack')."
            )

        if "email" in methods:
            if not all([os.getenv("GMAIL_ADDRESS"),
                        os.getenv("GMAIL_APP_PASSWORD"),
                        os.getenv("EMAIL_TO")]):
                errors.append(
                    "Email notification enabled but missing credentials. "
                    "Required: GMAIL_ADDRESS, GMAIL_APP_PASSWORD, EMAIL_TO"
                )

        if "resend" in methods:
            if not all([os.getenv("RESEND_API_KEY"), os.getenv("EMAIL_TO")]):
                errors.append(
                    "Resend notification enabled but missing credentials. "
                    "Required: RESEND_API_KEY, EMAIL_TO"
                )

        if "slack" in methods and not os.getenv("SLACK_WEBHOOK_URL"):
            errors.append("Slack enabled but SLACK_WEBHOOK_URL not set")

        if "telegram" in methods:
            if not all([os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")]):
                errors.append("Telegram enabled but missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID")

        if "discord" in methods and not os.getenv("DISCORD_WEBHOOK_URL"):
            errors.append("Discord enabled but DISCORD_WEBHOOK_URL not set")

        if "webhook" in methods and not os.getenv("WEBHOOK_URL"):
            errors.append("Webhook enabled but WEBHOOK_URL not set")

        # 3. Languages (ai_response_languages falls back to ['en'], so this
        #    guards against future changes to that behavior)
        if not self.ai_response_languages:
            errors.append("No valid languages configured (AI_RESPONSE_LANGUAGE)")

        # 4. Report everything at once
        if errors:
            error_msg = "\n\u274c Configuration validation failed:\n\n" + "\n".join(f"  \u2022 {e}" for e in errors)
            error_msg += "\n\nPlease fix the configuration and try again.\n"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("\u2713 Configuration validation passed")

    def _find_config_file(self, config_path: Optional[str] = None) -> Path:
        """
        Find the configuration file.

        Args:
            config_path: Explicit path to config file

        Returns:
            Path to configuration file

        Raises:
            FileNotFoundError: If config file cannot be found
        """
        if config_path:
            path = Path(config_path)
            if path.exists():
                return path
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Search in default locations
        search_paths = [
            Path("config.yaml"),
            Path("config.yml"),
            Path(__file__).parent.parent / "config.yaml",
            Path(__file__).parent.parent / "config.yml",
        ]

        for path in search_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            "Config file not found. Searched: " + ", ".join(str(p) for p in search_paths)
        )

    def _load_yaml_config(self) -> Dict[str, Any]:
        """
        Load YAML configuration file.

        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config or {}
        except Exception as e:
            logger.error(f"Failed to load config file: {str(e)}")
            return {}

    @property
    def news_topics(self) -> List[str]:
        """Get list of news topics to cover"""
        return self.config_data.get("news", {}).get("topics", [
            "Latest AI developments and breakthroughs"
        ])

    @property
    def stage1_prompt_template(self) -> str:
        """Get the Stage 1 selection prompt template"""
        default_template = """{formatted_news}

## YOUR TASK - STAGE 1: NEWS SELECTION

You are a senior AI industry analyst. Analyze the {total_items} news items above and select exactly 15-20 of the highest-quality items.

### SELECTION CRITERIA:
- ✅ Groundbreaking research or technical breakthroughs
- ✅ Major product launches or significant updates
- ✅ Important policy changes or regulations
- ✅ Large funding rounds or M&A activities
- ✅ Balanced coverage across categories (LLM, Agents, Research, Products, etc.)
- ✅ Include both international and domestic news when available
- ✅ Prefer primary sources over secondary reporting

### OUTPUT FORMAT:
Return ONLY a JSON array of selected news IDs. No explanations, no markdown, just the JSON array.

Example format:
["INT-1", "INT-5", "DOM-2", "INT-12", ...]

CRITICAL: Select exactly 15-20 items. No more, no less."""

        return self.config_data.get("news", {}).get("stage1_prompt_template", default_template)

    @property
    def stage2_prompt_template(self) -> str:
        """Get the Stage 2 summarization prompt template"""
        default_template = """You are a senior AI industry analyst. Create a comprehensive, in-depth news digest for the {count} pre-selected news items below.

{selected_news}

## OUTPUT STRUCTURE:

Organize news items into relevant categories (use only categories that have news):
1. **Large Language Models & Foundation Models**
2. **AI Agents & Autonomous Systems**
3. **Research & Academic Breakthroughs**
4. **Product Launches & Updates**
5. **AI Infrastructure & Hardware**
6. **Funding & Market Dynamics**
7. **Policy & Regulation**

## CONTENT REQUIREMENTS:

For each news item:
- **Clear Headline**: Informative title
- **Analytical Summary (4-6 sentences)**: What happened, technical details, why it matters, implications
- **Source Attribution**: [Source Name](URL)

## WRITING STYLE:
- Professional, analytical tone
- Include specific metrics and data
- Technical accuracy
- Context and analysis

## QUALITY REQUIREMENTS:
- ✅ Summarize ALL {count} items (no skipping)
- ✅ Each summary exactly 4-6 sentences
- ✅ Include specific numbers and data
- ✅ Balanced coverage across categories
- ✅ All sources as clickable markdown links

## AVOID:
❌ Generic statements
❌ Wrong summary length
❌ Missing links
❌ Skipping items"""

        return self.config_data.get("news", {}).get("stage2_prompt_template", default_template)

    @property
    def log_level(self) -> str:
        """Get logging level"""
        return self.config_data.get("logging", {}).get("level", "INFO")

    @property
    def log_format(self) -> str:
        """Get logging format"""
        return self.config_data.get("logging", {}).get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    @property
    def notification_methods(self) -> List[str]:
        """Get enabled notification methods from environment"""
        methods_str = os.getenv("NOTIFICATION_METHODS", "")
        if not methods_str:
            return []
        return [m.strip().lower() for m in methods_str.split(",")]

    @property
    def ai_response_language(self) -> str:
        """Get the language for AI-generated content (single language, deprecated)"""
        return os.getenv("AI_RESPONSE_LANGUAGE", "en").strip().lower()
    
    @property
    def ai_response_languages(self) -> List[str]:
        """Get the list of languages for AI-generated content (supports comma-separated values)"""
        languages_str = os.getenv("AI_RESPONSE_LANGUAGE", "en").strip().lower()
        # Split by comma and clean up whitespace
        languages = [lang.strip() for lang in languages_str.split(",") if lang.strip()]
        # Validate languages
        valid_languages = []
        for lang in languages:
            if lang == "en" or lang in LANGUAGE_NAMES:
                valid_languages.append(lang)
            else:
                logger.warning(f"Unsupported language code '{lang}', skipping")
        # Return at least 'en' if no valid languages
        return valid_languages if valid_languages else ["en"]

    @property
    def enable_web_search(self) -> bool:
        """Get whether to enable web search for fetching current news"""
        # Check config file first, then environment variable
        config_value = self.config_data.get("news", {}).get("enable_web_search")
        if config_value is not None:
            return bool(config_value)
        env_value = os.getenv("ENABLE_WEB_SEARCH", "false").strip().lower()
        return env_value in ("true", "1", "yes", "on")

    @property
    def max_items_per_source(self) -> int:
        """Maximum news items to fetch per source"""
        return self.config_data.get("news", {}).get("max_items_per_source", 5)

    @property
    def llm_provider(self) -> str:
        """Get the LLM provider to use (claude or deepseek)"""
        # Check environment variable first, then config file
        env_provider = os.getenv("LLM_PROVIDER", "").strip().lower()
        if env_provider:
            return env_provider
        return self.config_data.get("llm", {}).get("provider", "claude").lower()

    @property
    def llm_model(self) -> Optional[str]:
        """Get the specific model to use (if specified)"""
        # Check environment variable first, then config file
        env_model = os.getenv("LLM_MODEL", "").strip()
        if env_model:
            return env_model
        return self.config_data.get("llm", {}).get("model")

    @property
    def llm_api_key(self) -> Optional[str]:
        """Get the API key for the LLM provider"""
        # Check environment variables based on provider
        provider = self.llm_provider
        if provider == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY")
        elif provider == "claude":
            return os.getenv("ANTHROPIC_API_KEY")
        elif provider == "gemini":
            return os.getenv("GOOGLE_API_KEY")
        elif provider == "grok":
            return os.getenv("XAI_API_KEY")
        elif provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        return None

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.

        Args:
            key: Dot-separated key path (e.g., "news.topics")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config_data

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value
