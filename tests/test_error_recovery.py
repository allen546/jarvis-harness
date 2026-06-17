import pytest
import sys
from unittest.mock import MagicMock
from jarvis.retry import retry_with_backoff

counter = 0

@pytest.mark.asyncio
async def test_retry_eventual_success() -> None:
    global counter
    counter = 0
    
    @retry_with_backoff(max_retries=3, base_delay=0.01)
    async def unstable_api():
        global counter
        counter += 1
        if counter < 3:
            raise ConnectionError("Transient error")
        return "success"
        
    res = await unstable_api()
    assert res == "success"
    assert counter == 3
    
@pytest.mark.asyncio
async def test_retry_ultimate_failure() -> None:
    @retry_with_backoff(max_retries=2, base_delay=0.01)
    async def broken_api():
        raise ValueError("Fatal error")
        
    with pytest.raises(ValueError):
        await broken_api()

@pytest.mark.asyncio
async def test_retry_openai_transient_error() -> None:
    # Try importing openai; if not installed, mock it for the test
    try:
        import openai
        # Create a real exception if possible, or mock one subclassed from the real one
        # openai.APIConnectionError has required args, let's mock the class check
        # Since we use isinstance(exc, (openai.RateLimitError, ...)) in retry.py,
        # we can raise a real openai.APIConnectionError
        try:
            raise openai.APIConnectionError(request=MagicMock())
        except openai.APIConnectionError as e:
            exc_instance = e
    except (ImportError, Exception):
        # Fallback to creating a mock class and registered in sys.modules if needed
        class DummyRateLimitError(Exception):
            pass
        
        mock_openai = MagicMock()
        mock_openai.RateLimitError = DummyRateLimitError
        sys.modules['openai'] = mock_openai
        exc_instance = DummyRateLimitError("Rate limit exceeded")

    global counter
    counter = 0

    @retry_with_backoff(max_retries=2, base_delay=0.01)
    async def openai_api():
        global counter
        counter += 1
        if counter < 2:
            raise exc_instance
        return "ok"

    res = await openai_api()
    assert res == "ok"
    assert counter == 2

@pytest.mark.asyncio
async def test_retry_anthropic_transient_error() -> None:
    try:
        import anthropic
        try:
            raise anthropic.APIConnectionError(request=MagicMock())
        except anthropic.APIConnectionError as e:
            exc_instance = e
    except (ImportError, Exception):
        class DummyRateLimitError(Exception):
            pass
        
        mock_anthropic = MagicMock()
        mock_anthropic.RateLimitError = DummyRateLimitError
        sys.modules['anthropic'] = mock_anthropic
        exc_instance = DummyRateLimitError("Rate limit exceeded")

    global counter
    counter = 0

    @retry_with_backoff(max_retries=2, base_delay=0.01)
    async def anthropic_api():
        global counter
        counter += 1
        if counter < 2:
            raise exc_instance
        return "ok"

    res = await anthropic_api()
    assert res == "ok"
    assert counter == 2

@pytest.mark.asyncio
async def test_retry_http_status_code_error() -> None:
    class DummyResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code

    class DummyHTTPStatusError(Exception):
        def __init__(self, status_code: int, in_response: bool = False):
            if in_response:
                self.response = DummyResponse(status_code)
                self.status_code = None
            else:
                self.status_code = status_code
            super().__init__(f"HTTP {status_code}")

    global counter
    counter = 0

    # Test status_code directly on exception (transient 429)
    @retry_with_backoff(max_retries=2, base_delay=0.01)
    async def http_api():
        global counter
        counter += 1
        if counter < 2:
            raise DummyHTTPStatusError(429)
        return "ok"

    res = await http_api()
    assert res == "ok"
    assert counter == 2

    # Test status_code in exc.response (transient 503)
    counter = 0
    @retry_with_backoff(max_retries=2, base_delay=0.01)
    async def http_api_response():
        global counter
        counter += 1
        if counter < 2:
            raise DummyHTTPStatusError(503, in_response=True)
        return "ok"

    res = await http_api_response()
    assert res == "ok"
    assert counter == 2

    # Verify non-transient status code (HTTP 400) is not retried and attempted exactly once
    counter_400 = 0
    @retry_with_backoff(max_retries=2, base_delay=0.01)
    async def http_api_400():
        nonlocal counter_400
        counter_400 += 1
        raise DummyHTTPStatusError(400)

    with pytest.raises(DummyHTTPStatusError):
        await http_api_400()
    assert counter_400 == 1

    # Verify non-transient status code (HTTP 401) is not retried and attempted exactly once
    counter_401 = 0
    @retry_with_backoff(max_retries=2, base_delay=0.01)
    async def http_api_401():
        nonlocal counter_401
        counter_401 += 1
        raise DummyHTTPStatusError(401)

    with pytest.raises(DummyHTTPStatusError):
        await http_api_401()
    assert counter_401 == 1

