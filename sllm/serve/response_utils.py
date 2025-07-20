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
Standardized response formatting utilities for ServerlessLLM.
"""

from typing import Any, Dict, Optional
from datetime import datetime


def success_response(
    message: str, 
    data: Optional[Any] = None, 
    status: str = "ok"
) -> Dict[str, Any]:
    """
    Create a standardized success response.
    
    Args:
        message: Success message
        data: Optional response data
        status: Status string (default: "ok")
        
    Returns:
        Standardized success response dictionary
    """
    response = {
        "status": status,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if data is not None:
        response["data"] = data
    
    return response


def operation_response(
    operation: str,
    resource: str,
    resource_id: Optional[str] = None,
    data: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Create a standardized operation response.
    
    Args:
        operation: The operation performed (e.g., "created", "updated", "deleted")
        resource: The resource type (e.g., "model", "worker", "task")
        resource_id: Optional resource identifier
        data: Optional response data
        
    Returns:
        Standardized operation response dictionary
    """
    message = f"{resource.title()} {operation} successfully"
    if resource_id:
        message += f": {resource_id}"
    
    return success_response(message=message, data=data)


def task_response(task_id: str, data: Optional[Any] = None) -> Dict[str, Any]:
    """
    Create a standardized task response.
    
    Args:
        task_id: The task identifier
        data: Optional task result data
        
    Returns:
        Standardized task response dictionary
    """
    return success_response(
        message=f"Task {task_id} completed successfully",
        data={
            "task_id": task_id,
            "result": data
        }
    )


def list_response(
    items: list, 
    resource_type: str, 
    total_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a standardized list response.
    
    Args:
        items: List of items
        resource_type: Type of resources being listed
        total_count: Optional total count (if different from len(items))
        
    Returns:
        Standardized list response dictionary
    """
    count = total_count if total_count is not None else len(items)
    
    return success_response(
        message=f"Retrieved {count} {resource_type}(s)",
        data={
            "items": items,
            "count": count
        }
    )


def health_response(services: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Create a standardized health check response.
    
    Args:
        services: Optional dict of service name -> status
        
    Returns:
        Standardized health response dictionary
    """
    data = {"healthy": True}
    if services:
        data["services"] = services
    
    return success_response(
        message="Service is healthy",
        data=data
    )