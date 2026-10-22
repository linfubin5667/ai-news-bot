"""
Base LLM Provider - Abstract base class for all LLM providers
"""
import time
from abc import ABC, abstractmethod
from typing import Callable, List, Dict, Any, Optional


_TRANSIENT_NAME_MARKERS = ('connection', 'timeout', 'timedout', 'ratelimit', 'rate_limit')
_TRANSIENT_MESSAGE_MARKERS = (
    'rate limit', 'temporarily', 'overloaded', 'connection error',
    'connection reset', 'timed out', 'timeout', 'try again',
)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def is_transient_llm_error(exc: BaseException) -> bool:
    """
    Heuristic: is this LLM call failure worth retrying?

    True for connection/timeout/rate-limit style errors (including the
    SDK exception types such as APIConnectionError, APITimeoutError and
    RateLimitError, matched by class name), for HTTP 429/5xx responses,
    and for messages that clearly describe a transient condition.
    """
    name = type(exc).__name__.lower()
    if any(marker in name for marker in _TRANSIENT_NAME_MARKERS):
        return True

    response = getattr(exc, 'response', None)
    status = getattr(response, 'status_code', None)
    if status in _RETRYABLE_STATUS_CODES:
        return True
    if status is not None:
        return False  # a definitive HTTP answer (e.g. 400/401) is not transient

    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MESSAGE_MARKERS)


def call_with_retry(
    fn: Callable[[], str],
    max_attempts: int = 3,
    base_delay: float = 2.0,
    sleep_fn: Callable[[float], None] = time.sleep,
    logger=None,
) -> str:
    """
    Call fn(), retrying transient failures with exponential backoff
    (base_delay, then doubling). Non-transient errors raise at once;
    after max_attempts the last error is re-raised.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as e:
            if attempt >= max_attempts or not is_transient_llm_error(e):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            if logger is not None:
                logger.warning(
                    f"LLM call failed with a transient error (attempt {attempt}/{max_attempts}): "
                    f"{type(e).__name__}: {e}; retrying in {delay:.0f}s"
                )
            sleep_fn(delay)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM provider.
        
        Args:
            api_key: API key for the provider
            model: Model name to use
        """
        self.api_key = api_key
        self.model = model
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    @abstractmethod
    def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        max_tokens: int = 2000,
        max_iterations: int = 8,
        **kwargs
    ) -> str:
        """
        Generate a response with tool calling support.
        
        Args:
            messages: List of message dicts
            tools: List of tool definitions
            max_tokens: Maximum tokens in response
            max_iterations: Maximum tool use iterations
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text response after tool interactions
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider"""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model name for this provider"""
        pass
