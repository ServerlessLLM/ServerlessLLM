# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #

import asyncio
import aiohttp
from typing import Dict, Any, Optional
from sllm.serve.logger import init_logger

logger = init_logger(__name__)


class HTTPRetryError(Exception):
    """Raised when HTTP request fails after all retries."""
    pass


async def http_request_with_retry(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    timeout: float = 30.0,
    **kwargs
) -> aiohttp.ClientResponse:
    """
    Make HTTP request with exponential backoff retry logic.
    
    Args:
        session: aiohttp ClientSession
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        timeout: Request timeout in seconds
        **kwargs: Additional arguments passed to session.request()
    
    Returns:
        aiohttp.ClientResponse object
        
    Raises:
        HTTPRetryError: If all retries are exhausted
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            kwargs.setdefault('timeout', timeout_obj)
            
            async with session.request(method, url, **kwargs) as response:
                # Consider 5xx status codes as retryable errors
                if response.status >= 500:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Server error: {response.status}"
                    )
                return response
                
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_exception = e
            
            if attempt == max_retries:
                logger.error(f"HTTP request failed after {max_retries + 1} attempts to {url}: {e}")
                break
                
            # Calculate exponential backoff delay
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(f"HTTP request to {url} failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.1f}s: {e}")
            await asyncio.sleep(delay)
    
    raise HTTPRetryError(f"HTTP request to {url} failed after {max_retries + 1} attempts: {last_exception}")


async def post_json_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    payload: Dict[str, Any],
    max_retries: int = 3,
    timeout: float = 30.0,
    **kwargs
) -> Dict[str, Any]:
    """
    POST JSON payload with retry logic and return JSON response.
    
    Args:
        session: aiohttp ClientSession
        url: Request URL
        payload: JSON payload to send
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        **kwargs: Additional arguments passed to http_request_with_retry()
    
    Returns:
        Parsed JSON response as dict
        
    Raises:
        HTTPRetryError: If all retries are exhausted
        ValueError: If response is not valid JSON
    """
    kwargs.setdefault('json', payload)
    
    response = await http_request_with_retry(
        session=session,
        method='POST',
        url=url,
        max_retries=max_retries,
        timeout=timeout,
        **kwargs
    )
    
    try:
        return await response.json()
    except Exception as e:
        raise ValueError(f"Failed to parse JSON response from {url}: {e}")


async def get_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    max_retries: int = 3,
    timeout: float = 30.0,
    **kwargs
) -> aiohttp.ClientResponse:
    """
    GET request with retry logic.
    
    Args:
        session: aiohttp ClientSession
        url: Request URL
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
        **kwargs: Additional arguments passed to http_request_with_retry()
    
    Returns:
        aiohttp.ClientResponse object
        
    Raises:
        HTTPRetryError: If all retries are exhausted
    """
    return await http_request_with_retry(
        session=session,
        method='GET',
        url=url,
        max_retries=max_retries,
        timeout=timeout,
        **kwargs
    )