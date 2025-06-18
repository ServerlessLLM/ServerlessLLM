# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #

import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import pytest
import aiohttp
import sys
import os
import subprocess


sys.path.insert(0, '../serverlessllm')

from sllm.serve.backends.sglang_backend import (
    SGLangBackend,
    SGLangMode,
    BackendStatus,
)

# 配置 pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


@pytest.fixture
def model_name():
    return "test-model"


@pytest.fixture
def backend_config():
    return {
        "server_url": "http://localhost:8123",
        "use_native_sglang": True,
        "timeout": 30,
        "cache_size": 100,
        "cache_ttl": 1800,
    }


@pytest.fixture
def http_backend_config():
    return {
        "server_url": "http://localhost:8123",
        "use_native_sglang": False,
        "timeout": 30,
    }


@pytest.fixture
def sglang_backend(model_name, backend_config):
    return SGLangBackend(model_name, backend_config)


@pytest.fixture
def http_sglang_backend(model_name, http_backend_config):
    return SGLangBackend(model_name, http_backend_config)


def test_init(sglang_backend, backend_config):
    """Test SGLang Backend initialization"""
   
    assert hasattr(sglang_backend, 'model_name')
    assert sglang_backend.model_name == "test-model"
    assert sglang_backend.status == BackendStatus.UNINITIALIZED
    assert sglang_backend.use_native == True
   
    assert sglang_backend.mode is None


def test_init_http_mode(http_sglang_backend, http_backend_config):
    """Test SGLang Backend initialization in HTTP mode"""
    assert http_sglang_backend.use_native == False
    assert http_sglang_backend.status == BackendStatus.UNINITIALIZED
    # 修正：初始 mode 是 None
    assert http_sglang_backend.mode is None


@pytest.mark.asyncio
async def test_init_backend_basic():
    """Test basic backend initialization"""
    config = {
        "server_url": "http://localhost:8123",
        "use_native_sglang": False, 
        "timeout": 30,
    }
    
    backend = SGLangBackend("test-model", config)
    

    with patch('aiohttp.ClientSession') as mock_session_cls:
        mock_session = Mock()
        mock_response = Mock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()
        
        mock_session.get.return_value = mock_response
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session_cls.return_value = mock_session
        
        await backend.init_backend()
        
        assert backend.status == BackendStatus.RUNNING
       
        assert backend.mode == SGLangMode.SERVER


@pytest.mark.asyncio
async def test_init_backend_native_mode():
    """Test backend initialization in native mode - simplified version"""
    config = {
        "server_url": "http://localhost:8123",
        "use_native_sglang": True, 
        "timeout": 30,
    }
    
    backend = SGLangBackend("test-model", config)
    
   
    assert backend.use_native == True
    assert backend.status == BackendStatus.UNINITIALIZED
    assert backend.mode is None
   
    request_data = {
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    }
    
    response = await backend.generate(request_data)
    assert "error" in response
    assert "not running" in response["error"].lower()
    
    print("✅ Native mode configuration and error handling test passed")


@pytest.mark.asyncio
async def test_generate_without_init(sglang_backend):
    """Test generation without initialization"""
    request_data = {
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    }
    
    response = await sglang_backend.generate(request_data)
    assert "error" in response
    
    assert "not running" in response["error"].lower()


@pytest.mark.asyncio
async def test_generate_http_mode():
    """Test generation in HTTP mode with proper mocking"""
    config = {
        "server_url": "http://localhost:8123",
        "use_native_sglang": False,
        "timeout": 30,
    }
    
    backend = SGLangBackend("test-model", config)
    
 
    mock_response_data = {
        "choices": [{
            "message": {
                "content": "Hello! I'm doing well, thank you for asking."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    }
    
    with patch('aiohttp.ClientSession') as mock_session_cls:
        mock_session = Mock()
        
        # Mock 健康检查
        mock_health_resp = Mock()
        mock_health_resp.status = 200
        mock_health_resp.__aenter__ = AsyncMock(return_value=mock_health_resp)
        mock_health_resp.__aexit__ = AsyncMock()
        
     
        mock_gen_resp = Mock()
        mock_gen_resp.status = 200
        mock_gen_resp.json = AsyncMock(return_value=mock_response_data)
        mock_gen_resp.__aenter__ = AsyncMock(return_value=mock_gen_resp)
        mock_gen_resp.__aexit__ = AsyncMock()
        
        mock_session.get.return_value = mock_health_resp
        mock_session.post.return_value = mock_gen_resp
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()
        mock_session_cls.return_value = mock_session
        
     
        
        await backend.init_backend()
        
      
        
        assert backend.status == BackendStatus.RUNNING
        assert backend.mode == SGLangMode.SERVER
        
   
        request_data = {
            "messages": [{"role": "user", "content": "Hello!"}],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = await backend.generate(request_data)
        
        assert "error" not in response
        assert "choices" in response
        assert len(response["choices"]) > 0


@pytest.mark.asyncio
async def test_encode_not_supported(sglang_backend):
    """Test that encode operation returns error"""
    request_data = {"input": ["Hello world"]}
    
    result = await sglang_backend.encode(request_data)
    assert "error" in result
    assert "not supported" in result["error"].lower()


def test_get_backend_stats(sglang_backend):
    """Test getting backend statistics"""
    stats = sglang_backend.get_backend_stats()
    
    assert "backend_type" in stats
    assert "mode" in stats
    assert "total_requests" in stats
    assert stats["backend_type"] == "sglang"


def test_health_status(sglang_backend):
    """Test health status reporting"""
    health = sglang_backend.get_health_status()
    
    assert "status" in health
    assert health["status"] in ["healthy", "unhealthy", "initializing"]
   


@pytest.mark.asyncio
async def test_shutdown():
    """Test backend shutdown"""
    config = {
        "server_url": "http://localhost:8123",
        "use_native_sglang": False,
        "timeout": 30,
    }
    
    backend = SGLangBackend("test-model", config)
    
   
    await backend.shutdown()
    
 
    assert backend.status in [BackendStatus.DELETING, BackendStatus.UNINITIALIZED]


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling for various scenarios"""
    config = {
        "server_url": "http://localhost:8123",
        "use_native_sglang": False,
        "timeout": 30,
    }
    
    backend = SGLangBackend("test-model", config)
    
 
    invalid_request = {
        "messages": [],  
        "max_tokens": -1  
    }
    
    response = await backend.generate(invalid_request)
    assert "error" in response


def test_configuration_validation():
    """Test configuration validation"""

    valid_config = {
        "server_url": "http://localhost:8123",
        "use_native_sglang": False,
        "timeout": 30,
    }
    
    backend = SGLangBackend("test-model", valid_config)
    assert backend.use_native == False
    assert backend.server_url == "http://localhost:8123"


def test_enum_values():
    """Test that we're using the correct enum values"""

    assert SGLangMode.SERVER.value == "server"
    assert SGLangMode.NATIVE_ENDPOINT.value == "endpoint"
    
    assert BackendStatus.UNINITIALIZED.value == 1
    assert BackendStatus.RUNNING.value == 2
    assert BackendStatus.STOPPING.value == 3
    assert BackendStatus.DELETING.value == 4



if __name__ == "__main__":
    print("🧪 Running Final SGLang Backend Unit Tests")
    print("=" * 50)
    
  
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "--tb=short"
    ], cwd="../serverlessllm")
    
    sys.exit(result.returncode)