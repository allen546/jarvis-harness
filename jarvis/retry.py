import asyncio
import random
import functools
from typing import Callable, Any

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            delay = base_delay
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    
                    # Identify transient errors to retry:
                    is_transient = isinstance(exc, (ConnectionError, TimeoutError))
                    
                    if not is_transient:
                        # Dynamic check for openai
                        try:
                            import openai
                            if isinstance(exc, (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError)):
                                is_transient = True
                        except ImportError:
                            pass
                            
                    if not is_transient:
                        # Dynamic check for anthropic
                        try:
                            import anthropic
                            if isinstance(exc, (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError)):
                                is_transient = True
                        except ImportError:
                            pass
                            
                    if not is_transient:
                        exc_name = type(exc).__name__
                        status_code = getattr(exc, "status_code", None)
                        if status_code is None:
                            response = getattr(exc, "response", None)
                            if response is not None:
                                status_code = getattr(response, "status_code", None)
                                
                        if status_code is not None:
                            # Only retry on rate limit (429) or server errors (5xx)
                            is_transient = status_code in (429, 500, 502, 503, 504)
                        else:
                            is_transient = (
                                "Timeout" in exc_name or
                                "Connection" in exc_name or
                                "ConnectTimeout" in exc_name or
                                "ReadTimeout" in exc_name
                            )
                            
                    if not is_transient or attempt == max_retries:
                        raise exc
                    
                    # Exponential backoff with jitter
                    jitter = random.uniform(0, 0.1 * delay)
                    await asyncio.sleep(delay + jitter)
                    delay *= 2
            raise last_exc
        return wrapper
    return decorator
