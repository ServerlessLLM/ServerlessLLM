#!/usr/bin/env python3
"""
Simple test script for SGLang Backend
Used for step-by-step debugging
"""

import asyncio
import os
import sys
import traceback

# Add project path
sys.path.insert(0, '/home/fiona/serverlessllm')

def test_imports():
    print("🔍 Testing imports...")
    try:
        from sllm.serve.backends.sglang_backend import SGLangBackend
        print("✅ SGLangBackend imported successfully")

        import sglang
        try:
            print(f"✅ SGLang version: {sglang.__version__}")
        except:
            print("✅ SGLang imported (version unknown)")

        from sglang.srt.entrypoints.engine import Engine as SGLEngine
        print("✅ SGLEngine imported successfully")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_model_path():
    """Check if the model path and required files exist"""
    print("\n🔍 Checking model path...")
    model_path = "/home/fiona/my_models/facebook/opt-1.3b"

    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return False

    required_files = ["config.json", "tensor.data_0", "tensor_index.json"]
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} is missing")
            return False

    return True

async def test_sglang_backend_basic():
    """Test basic functionality of SGLang Backend"""
    print("\n🔍 Testing basic functionality of SGLang Backend...")

    try:
        from sllm.serve.backends.sglang_backend import SGLangBackend

        # Create backend config
        backend_config = {
            'load_format': 'serverless_llm',
            'tp_size': 1,
            'mem_fraction_static': 0.4,
            'enable_flashinfer': False,
        }

        print("✅ Backend config created")

        # Instantiate backend
        model_name = "/home/fiona/my_models/facebook/opt-1.3b"
        backend = SGLangBackend(model_name, backend_config)
        print("✅ SGLangBackend instance created")

        # Initialize backend (may fail)
        print("🔧 Initializing backend...")
        await backend.init_backend()
        print("✅ Backend initialized successfully!")

        return backend
    except Exception as e:
        print(f"❌ SGLangBackend test failed: {e}")
        traceback.print_exc()
        return None

async def main():
    """Main function"""
    print("🚀 Simple SGLang Backend Test")
    print("=" * 50)

    # Check SLLM Store status
    print("🔍 Checking SLLM Store status...")
    import socket
    
    # Get port from environment variable
    port = int(os.getenv("SLLM_STORE_PORT", "8073"))
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()

    if result == 0:
        print(f"✅ SLLM Store is running on port {port}")
    else:
        print(f"❌ SLLM Store is not running on port {port}")
        print(f"Please start it with: sllm-store start --storage-path /home/fiona/my_models --mem-pool-size 6GB --port {port}")
        return False

    # Step 1: Test imports
    if not test_imports():
        return False

    # Step 2: Test model path
    if not test_model_path():
        return False

    # Step 3: Test SGLang Backend
    backend = await test_sglang_backend_basic()
    if backend is None:
        return False

    # Step 4: Run simple inference
    print("\n🔍 Testing inference...")
    try:
        request_data = {
            "prompt": "Hello, I am",
            "max_new_tokens": 10,
            "temperature": 0.7
        }
        result = await backend.generate(request_data)
        print(f"✅ Inference succeeded! Result: {result}")
        return True
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
