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

"""
Standardized exception hierarchy for ServerlessLLM.
"""

from typing import Optional, Dict, Any


class ServerlessLLMError(Exception):
    """Base exception for all ServerlessLLM errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class ValidationError(ServerlessLLMError):
    """Raised when input validation fails."""
    pass


class ResourceNotFoundError(ServerlessLLMError):
    """Raised when a requested resource cannot be found."""
    pass


class ResourceConflictError(ServerlessLLMError):
    """Raised when a resource operation conflicts with current state."""
    pass


class InternalServerError(ServerlessLLMError):
    """Raised for internal server errors."""
    pass


class WorkerError(ServerlessLLMError):
    """Raised for worker-related errors."""
    pass


class ModelError(ServerlessLLMError):
    """Raised for model-related errors."""
    pass


class TaskError(ServerlessLLMError):
    """Raised for task processing errors."""
    pass


class RedisError(ServerlessLLMError):
    """Raised for Redis connection/operation errors."""
    pass


class TimeoutError(ServerlessLLMError):
    """Raised when an operation times out."""
    pass


def standardize_error_response(error: Exception) -> Dict[str, Any]:
    """
    Convert any exception to a standardized error response format.
    
    Args:
        error: The exception to convert
        
    Returns:
        Standardized error response dictionary
    """
    if isinstance(error, ServerlessLLMError):
        return {
            "error": {
                "code": error.error_code,
                "message": error.message,
                "details": error.details
            }
        }
    else:
        # Handle non-ServerlessLLM exceptions
        return {
            "error": {
                "code": "InternalError",
                "message": str(error),
                "details": {}
            }
        }


def map_to_http_status(error: Exception) -> int:
    """
    Map exception types to appropriate HTTP status codes.
    
    Args:
        error: The exception to map
        
    Returns:
        HTTP status code
    """
    if isinstance(error, ValidationError):
        return 400
    elif isinstance(error, ResourceNotFoundError):
        return 404
    elif isinstance(error, ResourceConflictError):
        return 409
    elif isinstance(error, TimeoutError):
        return 408
    elif isinstance(error, (WorkerError, ModelError, TaskError)):
        return 503  # Service Unavailable
    elif isinstance(error, RedisError):
        return 503  # Service Unavailable
    elif isinstance(error, InternalServerError):
        return 500
    else:
        return 500  # Default to internal server error