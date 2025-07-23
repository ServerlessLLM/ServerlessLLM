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
Input validation utilities for ServerlessLLM.
"""

import ipaddress
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from sllm.serve.utils import ValidationError


def validate_model_identifier(model_identifier: str) -> tuple[str, str]:
    """
    Validate and parse model identifier in format 'model_name:backend'.

    Args:
        model_identifier: String in format 'model_name:backend'

    Returns:
        Tuple of (model_name, backend)

    Raises:
        ValidationError: If format is invalid
    """
    if not model_identifier or not isinstance(model_identifier, str):
        raise ValidationError("Model identifier must be a non-empty string")

    parts = model_identifier.split(":", 1)
    if len(parts) != 2:
        raise ValidationError(
            "Model identifier must be in format 'model_name:backend'"
        )

    model_name, backend = parts

    if not model_name or not backend:
        raise ValidationError("Both model name and backend must be non-empty")

    # Validate model name
    validate_model_name(model_name)

    # Validate backend
    if backend not in ["vllm", "transformers", "dummy"]:
        raise ValidationError(f"Invalid backend: {backend}")

    return model_name, backend


def validate_model_name(model_name: str) -> None:
    """
    Validate model name to prevent path traversal and injection attacks.

    Args:
        model_name: Model name to validate

    Raises:
        ValidationError: If model name is invalid
    """
    if not model_name or not isinstance(model_name, str):
        raise ValidationError("Model name must be a non-empty string")

    # Check for path traversal sequences
    if ".." in model_name or "/" in model_name or "\\" in model_name:
        raise ValidationError(
            "Model name cannot contain path traversal sequences"
        )

    # Check for valid characters (alphanumeric, hyphens, underscores, dots)
    if not re.match(r"^[a-zA-Z0-9._-]+$", model_name):
        raise ValidationError("Model name contains invalid characters")

    # Check length
    if len(model_name) > 100:
        raise ValidationError("Model name is too long (max 100 characters)")

    # Check for reserved names
    reserved_names = {".", "..", "con", "prn", "aux", "nul"}
    if model_name.lower() in reserved_names:
        raise ValidationError(f"Model name '{model_name}' is reserved")


def validate_node_ip(ip_address: str) -> None:
    """
    Validate IP address format.

    Args:
        ip_address: IP address string to validate

    Raises:
        ValidationError: If IP address is invalid
    """
    if not ip_address or not isinstance(ip_address, str):
        raise ValidationError("IP address must be a non-empty string")

    try:
        ipaddress.ip_address(ip_address)
    except ValueError:
        raise ValidationError(f"Invalid IP address format: {ip_address}")

    # Don't allow localhost/loopback in production
    ip = ipaddress.ip_address(ip_address)
    if ip.is_loopback:
        # Allow localhost for development
        if not os.getenv("SLLM_DEV_MODE", "false").lower() == "true":
            raise ValidationError(
                "Loopback addresses not allowed in production"
            )


def validate_redis_host(host: str) -> None:
    """
    Validate Redis host configuration.

    Args:
        host: Redis host string

    Raises:
        ValidationError: If host is invalid
    """
    if not host or not isinstance(host, str):
        raise ValidationError("Redis host must be a non-empty string")

    # Check if it's an IP address
    try:
        validate_node_ip(host)
        return
    except ValidationError:
        pass  # Not an IP, check if it's a hostname

    # Validate hostname
    if len(host) > 253:
        raise ValidationError("Hostname too long")

    # Check hostname format
    hostname_pattern = r"^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
    if not re.match(hostname_pattern, host):
        raise ValidationError(f"Invalid hostname format: {host}")


def validate_storage_path(storage_path: str) -> None:
    """
    Validate storage path exists and is accessible.

    Args:
        storage_path: Storage path to validate

    Raises:
        ValidationError: If path is invalid or inaccessible
    """
    if not storage_path or not isinstance(storage_path, str):
        raise ValidationError("Storage path must be a non-empty string")

    try:
        path = Path(storage_path)

        # Check if path exists
        if not path.exists():
            raise ValidationError(
                f"Storage path does not exist: {storage_path}"
            )

        # Check if it's a directory
        if not path.is_dir():
            raise ValidationError(
                f"Storage path is not a directory: {storage_path}"
            )

        # Check if writable
        if not os.access(path, os.W_OK):
            raise ValidationError(
                f"Storage path is not writable: {storage_path}"
            )

    except OSError as e:
        raise ValidationError(f"Cannot access storage path: {e}")


def validate_url(url: str) -> None:
    """
    Validate URL format.

    Args:
        url: URL string to validate

    Raises:
        ValidationError: If URL is invalid
    """
    if not url or not isinstance(url, str):
        raise ValidationError("URL must be a non-empty string")

    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValidationError("URL must have scheme and netloc")

        if parsed.scheme not in ["http", "https"]:
            raise ValidationError("URL scheme must be http or https")

    except Exception as e:
        raise ValidationError(f"Invalid URL format: {e}")


def validate_task_id(task_id: str) -> None:
    """
    Validate task ID format.

    Args:
        task_id: Task ID to validate

    Raises:
        ValidationError: If task ID is invalid
    """
    if not task_id or not isinstance(task_id, str):
        raise ValidationError("Task ID must be a non-empty string")

    # Task IDs should be alphanumeric with hyphens
    if not re.match(r"^[a-zA-Z0-9-]+$", task_id):
        raise ValidationError("Task ID contains invalid characters")

    if len(task_id) > 100:
        raise ValidationError("Task ID is too long")


def validate_model_config(model_config: Dict[str, Any]) -> None:
    """
    Validate model configuration dictionary.

    Args:
        model_config: Model configuration to validate

    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(model_config, dict):
        raise ValidationError("Model config must be a dictionary")

    # Required fields
    required_fields = ["model_name", "backend"]
    for field in required_fields:
        if field not in model_config:
            raise ValidationError(f"Missing required field: {field}")

    # Validate individual fields
    validate_model_name(model_config["model_name"])

    backend = model_config["backend"]
    if backend not in ["vllm", "transformers", "dummy"]:
        raise ValidationError(f"Invalid backend: {backend}")

    # Validate backend_config if present
    backend_config = model_config.get("backend_config")
    if backend_config is not None and not isinstance(backend_config, dict):
        raise ValidationError("backend_config must be a dictionary")

    # Validate startup_config if present
    startup_config = model_config.get("startup_config")
    if startup_config is not None and not isinstance(startup_config, dict):
        raise ValidationError("startup_config must be a dictionary")


def sanitize_log_data(data: Any) -> Any:
    """
    Sanitize data for logging to remove sensitive information.

    Args:
        data: Data to sanitize

    Returns:
        Sanitized data safe for logging
    """
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            # Mask sensitive keys
            if any(
                sensitive in key.lower()
                for sensitive in ["password", "token", "key", "secret"]
            ):
                sanitized[key] = "***MASKED***"
            else:
                sanitized[key] = sanitize_log_data(value)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_log_data(item) for item in data]
    elif isinstance(data, str):
        # Mask potential API keys or tokens
        if len(data) > 20 and re.match(r"^[a-zA-Z0-9+/=]+$", data):
            return "***POTENTIAL_TOKEN_MASKED***"
        return data
    else:
        return data
