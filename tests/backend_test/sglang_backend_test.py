# #!/usr/bin/env python3
# """
# Simple test script for SGLang Backend
# Used for step-by-step debugging
# """

# import asyncio
# import os
# import sys
# import traceback

# # Add project path
# sys.path.insert(0, "/home/fiona/serverlessllm")


# def test_imports():
#     print("🔍 Testing imports...")
#     try:
#         from sllm.serve.backends.sglang_backend import SGLangBackend

#         print("✅ SGLangBackend imported successfully")

#         import sglang

#         try:
#             print(f"✅ SGLang version: {sglang.__version__}")
#         except:
#             print("✅ SGLang imported (version unknown)")

#         from sglang.srt.entrypoints.engine import Engine as SGLEngine

#         print("✅ SGLEngine imported successfully")

#         return True
#     except Exception as e:
#         print(f"❌ Import failed: {e}")
#         traceback.print_exc()
#         return False


# def test_model_path():
#     """Check if the model path and required files exist"""
#     print("\n🔍 Checking model path...")

#     model_path = "/home/fiona/sllm_models/facebook_writable/opt-1.3b-padded/"

#     if not os.path.exists(model_path):
#         print(f"❌ Model path does not exist: {model_path}")
#         return False

#     required_files = ["config.json", "tensor.data_0", "tensor_index.json"]
#     for file in required_files:
#         file_path = os.path.join(model_path, file)
#         if os.path.exists(file_path):
#             print(f"✅ {file} exists")
#         else:
#             print(f"❌ {file} is missing")
#             return False

#     return True


# async def test_sglang_backend_basic():
#     """Test basic functionality of SGLang Backend"""
#     print("\n🔍 Testing basic functionality of SGLang Backend...")

#     try:
#         from sllm.serve.backends.sglang_backend import SGLangBackend

#         # Create backend config
#         backend_config = {
#             "load_format": "serverless_llm",
#             "tp_size": 1,
#             "mem_fraction_static": 0.4,
#             "enable_flashinfer": False,
#         }

#         print("✅ Backend config created")

#         # Instantiate backend
#         model_name = "/home/fiona/sllm_models/facebook_writable/opt-1.3b-padded"
#         backend = SGLangBackend(model_name, backend_config)
#         print("✅ SGLangBackend instance created")

#         # Initialize backend (may fail)
#         print("🔧 Initializing backend...")
#         await backend.init_backend()
#         print("✅ Backend initialized successfully!")

#         return backend
#     except Exception as e:
#         print(f"❌ SGLangBackend test failed: {e}")
#         traceback.print_exc()
#         return None


# async def main():
#     """Main function"""
#     print("🚀 Simple SGLang Backend Test")
#     print("=" * 50)

#     # Check SLLM Store status
#     print("🔍 Checking SLLM Store status...")
#     import socket

#     # Get port from environment variable
#     port = int(os.getenv("SLLM_STORE_PORT", "8073"))

#     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     result = sock.connect_ex(("localhost", port))
#     sock.close()

#     if result == 0:
#         print(f"✅ SLLM Store is running on port {port}")
#     else:
#         print(f"❌ SLLM Store is not running on port {port}")
#         print(
#             f"Please start it with: sllm-store start --storage-path /home/fiona/sllm_models --mem-pool-size 6GB --port {port}"
#         )
#         return False

#     # Step 1: Test imports
#     if not test_imports():
#         return False

#     # Step 2: Test model path
#     if not test_model_path():
#         return False

#     # Step 3: Test SGLang Backend
#     backend = await test_sglang_backend_basic()
#     if backend is None:
#         return False

#     # Step 4: Run simple inference
#     print("\n🔍 Testing inference...")
#     try:
#         request_data = {
#             "prompt": "Hello, how are you?",
#             "max_new_tokens": 10,
#             "temperature": 0.7,
#         }
#         result = await backend.generate(request_data)
#         print(f"✅ Inference succeeded! Result: {result}")
#         return True
#     except Exception as e:
#         print(f"❌ Inference failed: {e}")
#         traceback.print_exc()
#         return False


# if __name__ == "__main__":
#     success = asyncio.run(main())
#     if success:
#         print("\n🎉 All tests passed!")
#         sys.exit(0)
#     else:
#         print("\n💥 Some tests failed!")
#         sys.exit(1)

from unittest.mock import AsyncMock, patch

import pytest

from sllm.serve.backends.sglang_backend import (
    BackendStatus,
    SGLangBackend,
)


@pytest.fixture
def model_path():
    return "/home/fiona/sllm_models/facebook_writable/opt-1.3b-padded"


@pytest.fixture
def backend_config():
    return {
        "load_format": "serverless_llm",
        "tp_size": 1,
        "mem_fraction_static": 0.4,
        "enable_flashinfer": False,
    }


@pytest.fixture
def async_sglang_engine():
    class MockSGLEngine:
        def __init__(self, *args, **kwargs):
            self.status = BackendStatus.UNINITIALIZED
            self.abort = AsyncMock()
            self._shutdown_called = False

            async def shutdown():
                self.status = BackendStatus.DELETING
                self._shutdown_called = True

            self.shutdown = AsyncMock(side_effect=shutdown)

            self.async_generate = AsyncMock(
                return_value={
                    "id": "test-request-id",
                    "model": "sglang-model",
                    "choices": [{"text": "test output"}],
                }
            )
            self.async_encode = AsyncMock(
                return_value={
                    "data": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                }
            )
            self.get_current_tokens = AsyncMock(return_value=[1, 2, 3])

            class Scheduler:
                def __init__(self):
                    self.running_reqs = [self.MockReq()]

                class MockReq:
                    prompt_token_ids = [1, 2, 3]
                    outputs = []

            self.scheduler = Scheduler()

        async def stop(self):
            self.status = BackendStatus.STOPPING

        async def resume_kv_cache(self, data):
            return [await self.async_generate({"prompt": str(d)}) for d in data]

    return MockSGLEngine()


@pytest.fixture
def sglang_backend(model_path, backend_config, async_sglang_engine):
    with patch(
        "sllm.serve.backends.sglang_backend.SGLEngine",
        return_value=async_sglang_engine,
    ):
        backend = SGLangBackend(model_path, backend_config)
        yield backend


def test_init(sglang_backend, backend_config):
    assert sglang_backend.backend_config == backend_config
    assert sglang_backend.status == BackendStatus.UNINITIALIZED


@pytest.mark.asyncio
async def test_init_backend(sglang_backend):
    await sglang_backend.init_backend()
    assert sglang_backend.status == BackendStatus.RUNNING


@pytest.mark.asyncio
async def test_generate_without_init(model_path, backend_config):
    backend = SGLangBackend(model_path, backend_config)
    request_data = {
        "prompt": "Hello",
        "request_id": "test-request-id",
    }
    response = await backend.generate(request_data)
    assert "error" in response


@pytest.mark.asyncio
async def test_generate(sglang_backend):
    await sglang_backend.init_backend()
    request_data = {
        "prompt": "Hello",
        "request_id": "test-request-id",
    }
    response = await sglang_backend.generate(request_data)
    assert "error" not in response
    assert "model" in response and response["model"] == "sglang-model"
    assert "id" in response and response["id"] == "test-request-id"


@pytest.mark.asyncio
async def test_shutdown(sglang_backend):
    await sglang_backend.init_backend()
    await sglang_backend.generate(
        {"prompt": "Hello", "request_id": "test-request-id"}
    )
    await sglang_backend.shutdown()
    assert sglang_backend.status == BackendStatus.UNINITIALIZED

    assert sglang_backend.engine is None


@pytest.mark.asyncio
async def test_stop(sglang_backend):
    await sglang_backend.init_backend()
    await sglang_backend.generate(
        {"prompt": "Hello", "request_id": "test-request-id"}
    )
    with patch.object(
        sglang_backend, "_get_running_requests", return_value=[]
    ), patch.object(
        sglang_backend, "shutdown", new_callable=AsyncMock
    ) as mock_shutdown:
        await sglang_backend.stop()
        assert (
            sglang_backend.status == BackendStatus.UNINITIALIZED
            or sglang_backend.status == BackendStatus.STOPPING
        )
        assert mock_shutdown.call_count == 1


@pytest.mark.asyncio
async def test_get_current_tokens(sglang_backend):
    await sglang_backend.init_backend()
    await sglang_backend.generate(
        {"prompt": "Hello", "request_id": "test-request-id"}
    )
    tokens = await sglang_backend.get_current_tokens()
    get_current_tokens = AsyncMock(return_value=[[1, 2, 3]])


@pytest.mark.asyncio
async def test_resume_kv_cache(sglang_backend):
    await sglang_backend.init_backend()
    sglang_backend.generate = AsyncMock()
    data = [[1, 2, 3], [4, 5, 6]]
    await sglang_backend.resume_kv_cache(data)
    assert sglang_backend.generate.call_count == len(data)


@pytest.mark.asyncio
async def test_encode(sglang_backend):
    await sglang_backend.init_backend()
    request_data = {"input": ["Hi, How are you?"]}
    result = await sglang_backend.encode(request_data)
    assert "data" in result
    assert len(result["data"]) == 2
