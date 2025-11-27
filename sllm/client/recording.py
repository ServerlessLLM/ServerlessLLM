"""
ServerlessLLM Recording Client APIs

This module provides client-side functions for interacting with ServerlessLLM's
recording features:
- Batch recording: Track batch statistics during inference
- Expert distribution recording: Track MoE expert activation patterns

Usage:
    from sllm.client.recording import (
        start_batch_recording,
        stop_batch_recording,
        dump_batch_recording,
        start_expert_distribution_recording,
        stop_expert_distribution_recording,
        dump_expert_distribution,
    )
"""

from typing import Any, Dict, Optional

import requests

# ============================================================================
# Batch Recording API Functions
# ============================================================================


def start_batch_recording(server_url: str, model_name: str) -> bool:
    """Start batch recording on the server.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model

    Returns:
        True if successful, False otherwise
    """
    url = f"{server_url}/start_batch_recording"
    payload = {"model": model_name}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"✓ Started batch recording for {model_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to start batch recording: {e}")
        return False


def stop_batch_recording(server_url: str, model_name: str) -> bool:
    """Stop batch recording on the server.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model

    Returns:
        True if successful, False otherwise
    """
    url = f"{server_url}/stop_batch_recording"
    payload = {"model": model_name}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"✓ Stopped batch recording for {model_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to stop batch recording: {e}")
        return False


def get_batch_recording_status(
    server_url: str, model_name: str
) -> Dict[str, Any]:
    """Get current batch recording status.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model

    Returns:
        Status dictionary
    """
    url = f"{server_url}/batch_recording_status"

    try:
        response = requests.get(url, params={"model": model_name})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"✗ Failed to get batch recording status: {e}")
        return {}


def dump_batch_recording(server_url: str, model_name: str) -> Dict[str, Any]:
    """Dump recorded batch statistics.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model

    Returns:
        Dictionary containing batch records
    """
    url = f"{server_url}/dump_batch_recording"
    payload = {"model": model_name}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        # Normalize the response - backend might return 'batches' or 'records'
        if "batches" in data and "records" not in data:
            data["records"] = data["batches"]
        elif "records" not in data:
            data["records"] = []

        print(f"✓ Dumped {len(data.get('records', []))} batch records")
        return data
    except Exception as e:
        print(f"✗ Failed to dump batch recording: {e}")
        return {}


def clear_batch_recording(server_url: str, model_name: str) -> bool:
    """Clear all recorded batch statistics.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model

    Returns:
        True if successful, False otherwise
    """
    url = f"{server_url}/clear_batch_recording"
    payload = {"model": model_name}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"✓ Cleared batch recording for {model_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to clear batch recording: {e}")
        return False


# ============================================================================
# Expert Distribution Recording API Functions
# ============================================================================


def configure_expert_distribution(
    server_url: str,
    model_name: str,
    recording_mode: str = "per_pass",
    enable_metrics: bool = True,
    buffer_size: int = -1,
) -> Dict[str, Any]:
    """Configure expert distribution recording on the server.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model
        recording_mode: One of "per_token", "per_pass", "stat", "stat_approx"
        enable_metrics: Whether to compute and log metrics
        buffer_size: Size of recording buffer (-1 for unlimited)

    Returns:
        Configuration status dictionary
    """
    url = f"{server_url}/configure_expert_distribution"
    payload = {
        "model": model_name,
        "recording_mode": recording_mode,
        "enable_metrics": enable_metrics,
        "buffer_size": buffer_size,
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print(
            f"✓ Configured expert distribution recording (mode={recording_mode})"
        )
        return result
    except Exception as e:
        print(f"✗ Failed to configure expert distribution: {e}")
        return {"status": "error", "message": str(e)}


def start_expert_distribution_recording(
    server_url: str,
    model_name: str,
    recording_mode: str = "per_pass",
) -> bool:
    """Start expert distribution recording on the server.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model
        recording_mode: One of "per_token", "per_pass", "stat", "stat_approx"

    Returns:
        True if successful, False otherwise
    """
    url = f"{server_url}/start_expert_distribution"
    payload = {
        "model": model_name,
        "recording_mode": recording_mode,
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"✓ Started expert distribution recording for {model_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to start expert distribution recording: {e}")
        return False


def stop_expert_distribution_recording(
    server_url: str, model_name: str
) -> bool:
    """Stop expert distribution recording on the server.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model

    Returns:
        True if successful, False otherwise
    """
    url = f"{server_url}/stop_expert_distribution"
    payload = {"model": model_name}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"✓ Stopped expert distribution recording for {model_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to stop expert distribution recording: {e}")
        return False


def dump_expert_distribution(
    server_url: str,
    model_name: str,
    output_path: Optional[str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Dump recorded expert distribution data.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model
        output_path: Optional path to save the recording on the server
        timeout: Request timeout in seconds

    Returns:
        Dictionary containing recorded expert distribution data
    """
    url = f"{server_url}/dump_expert_distribution"
    payload = {"model": model_name}
    if output_path:
        payload["output_path"] = output_path

    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        # Count records from various sources
        worker_data = data.get("worker_data", [])
        local_records = data.get("local_records", [])
        total = len(local_records)
        if isinstance(worker_data, list):
            for wd in worker_data:
                if isinstance(wd, dict):
                    total += len(wd.get("records", []))

        # Only print if there are records
        if total > 0:
            print(f"✓ Dumped expert distribution data ({total} total records)")
        return data
    except Exception as e:
        print(f"✗ Failed to dump expert distribution: {e}")
        return {"status": "error", "message": str(e)}


def get_expert_distribution_status(
    server_url: str, model_name: str
) -> Dict[str, Any]:
    """Get current expert distribution recording status.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model

    Returns:
        Status dictionary
    """
    url = f"{server_url}/expert_distribution_status"

    try:
        response = requests.get(url, params={"model": model_name})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"✗ Failed to get expert distribution status: {e}")
        return {}


def clear_expert_distribution(server_url: str, model_name: str) -> bool:
    """Clear all recorded expert distribution data.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model

    Returns:
        True if successful, False otherwise
    """
    url = f"{server_url}/clear_expert_distribution"
    payload = {"model": model_name}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print(f"✓ Cleared expert distribution recording for {model_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to clear expert distribution: {e}")
        return False


# ============================================================================
# Convenience Functions
# ============================================================================


def start_all_recording(
    server_url: str,
    model_name: str,
    enable_expert_recording: bool = True,
    expert_recording_mode: str = "per_pass",
) -> bool:
    """Start both batch and expert distribution recording.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model
        enable_expert_recording: Whether to enable expert distribution recording
        expert_recording_mode: Expert distribution recording mode

    Returns:
        True if batch recording started successfully
    """
    batch_ok = start_batch_recording(server_url, model_name)

    if enable_expert_recording:
        start_expert_distribution_recording(
            server_url, model_name, expert_recording_mode
        )

    return batch_ok


def stop_all_recording(
    server_url: str,
    model_name: str,
    enable_expert_recording: bool = True,
) -> bool:
    """Stop both batch and expert distribution recording.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model
        enable_expert_recording: Whether expert distribution recording was enabled

    Returns:
        True if batch recording stopped successfully
    """
    batch_ok = stop_batch_recording(server_url, model_name)

    if enable_expert_recording:
        stop_expert_distribution_recording(server_url, model_name)

    return batch_ok


def clear_all_recording(
    server_url: str,
    model_name: str,
    enable_expert_recording: bool = True,
) -> bool:
    """Clear both batch and expert distribution recordings.

    Args:
        server_url: Base URL of ServerlessLLM server
        model_name: Name of the model
        enable_expert_recording: Whether to clear expert distribution recording

    Returns:
        True if batch recording cleared successfully
    """
    batch_ok = clear_batch_recording(server_url, model_name)

    if enable_expert_recording:
        clear_expert_distribution(server_url, model_name)

    return batch_ok
