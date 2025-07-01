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
import gc
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Union


import torch


# Import SGLang components
try:
    import sglang as sgl
    from sglang.srt.entrypoints.engine import Engine as SGLEngine
except ImportError as e:
    print(f"Warning: SGLang not available: {e}")
    sgl = None
    SGLEngine = None


from sllm.serve.backends.backend_utils import (
    BackendStatus,
    SllmBackend,
)


logger = logging.getLogger("sllm.sglang_backend")




class SGLangBackend(SllmBackend):
    """SGLang backend implementation for SLLM, similar to VLLMBackend structure."""
   
    def __init__(
        self, model: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")


        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.backend_config = backend_config
        self.model_name = model
       
        # SGLang specific configurations - follow VLLM backend pattern
        load_format = backend_config.get("load_format")
       
        if load_format is not None:
            # If load_format is specified, use pretrained_model_name_or_path
            self.load_format = load_format
            self.model_path = backend_config.get("pretrained_model_name_or_path")
        else:
            # Default behavior: use STORAGE_PATH with sglang subdirectory
            storage_path = os.getenv("STORAGE_PATH", "./models")
            self.model_path = os.path.join(storage_path, "sglang", model)
            self.load_format = "serverless_llm"
       
        # Extract SGLang specific configs - let SGLang use its own defaults where possible
        # Only set essential overrides
        self.device = backend_config.get("device", "cuda")
        self.tp_size = backend_config.get("tp_size", 1)
       
        # Only override memory fraction if explicitly set, otherwise let SGLang decide
        self.mem_fraction_static = backend_config.get("mem_fraction_static")
       
        # Network configs - only set if needed for distributed setup
        self.host = backend_config.get("host")
        self.port = backend_config.get("port")
       
        logger.info(f"Initializing SGLang backend for model: {model}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Load format: {self.load_format}")


        self.engine = None


    async def init_backend(self) -> None:
        """Initialize the SGLang backend engine."""
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return
           
            try:
                # Import SGLang engine with correct import
                from sglang.srt.entrypoints.engine import Engine as SGLEngine
                import signal
                import time
               
                logger.info("Creating SGLang engine using SGLEngine...")
               
                # Create SGLang engine with dynamic parameter construction
                engine_kwargs = {}
               
                # Add model path - handle None case for HuggingFace models
                if self.model_path is not None:
                    engine_kwargs["model_path"] = self.model_path
                else:
                    # If model_path is None, use the original model name for HuggingFace models
                    engine_kwargs["model_path"] = self.model_name
               
                # Add load format
                engine_kwargs["load_format"] = self.load_format
               
                # Add device and tensor parallel size
                engine_kwargs["device"] = self.device
                engine_kwargs["tp_size"] = self.tp_size
               
                # Only add optional parameters if they are explicitly set
                if self.mem_fraction_static is not None:
                    engine_kwargs["mem_fraction_static"] = self.mem_fraction_static
               
                if self.host is not None:
                    engine_kwargs["host"] = self.host
                   
                if self.port is not None:
                    engine_kwargs["port"] = self.port
               
                # Add other optional configurations only if present
                optional_configs = [
                    "disable_radix_cache",
                    "enable_mixed_chunk",
                    "context_length",
                    "log_level"
                ]
               
                for config_key in optional_configs:
                    if config_key in self.backend_config:
                        engine_kwargs[config_key] = self.backend_config[config_key]
               
                logger.info(f"SGLang engine parameters: {engine_kwargs}")
               
                # Set up signal handling for graceful shutdown
                def signal_handler(signum, frame):
                    logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
                    if hasattr(self, 'engine') and self.engine is not None:
                        try:
                            self.engine.shutdown()
                        except Exception as e:
                            logger.error(f"Error during engine shutdown: {e}")
                    raise KeyboardInterrupt(f"Received signal {signum}")
               
                # Install signal handlers
                original_sigint = signal.signal(signal.SIGINT, signal_handler)
                original_sigterm = signal.signal(signal.SIGTERM, signal_handler)
               
                try:
                    # Create engine with timeout and retry mechanism
                    logger.info("Initializing SGLang engine (this may take a few moments)...")
                    start_time = time.time()
                   
                    self.engine = SGLEngine(**engine_kwargs)
                   
                    init_time = time.time() - start_time
                    logger.info(f"SGLang engine initialized successfully in {init_time:.2f} seconds")
                   
                    self.status = BackendStatus.RUNNING
                   
                except KeyboardInterrupt:
                    logger.warning("Engine initialization interrupted by user")
                    self.status = BackendStatus.UNINITIALIZED
                    if hasattr(self, 'engine') and self.engine is not None:
                        try:
                            self.engine.shutdown()
                        except:
                            pass
                        self.engine = None
                    raise
               
                finally:
                    # Restore original signal handlers
                    signal.signal(signal.SIGINT, original_sigint)
                    signal.signal(signal.SIGTERM, original_sigterm)
               
            except KeyboardInterrupt:
                logger.warning("SGLang backend initialization interrupted")
                self.status = BackendStatus.UNINITIALIZED
                raise
            except Exception as e:
                logger.error(f"Failed to initialize SGLang engine: {e}")
                self.status = BackendStatus.UNINITIALIZED
                raise


    async def generate(self, request_data: Dict[str, Any]):
        """Generate text using SGLang engine."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}


        assert self.engine is not None


        if request_data is None:
            return {"error": "Request data is missing"}


        model_name: str = request_data.pop("model", "sglang-model")
        messages: List[Dict[str, str]] = request_data.pop("messages", [])
       
        # Construct prompt from messages if no direct prompt provided
        construct_prompt: str = "\n".join(
            [
                f"{message['role']}: {message['content']}"
                for message in messages
                if "content" in message
            ]
        )


        # Get prompt or use constructed one
        prompt: str = request_data.pop("prompt", construct_prompt)
       
        request_id: str = request_data.pop(
            "request_id", f"sglang-{uuid.uuid4()}"
        )


        try:
            # Create SGLang sampling parameters
            sampling_params = {
                "max_new_tokens": request_data.get("max_tokens", request_data.get("max_new_tokens", 100)),
                "temperature": request_data.get("temperature", 0.7),
                "top_p": request_data.get("top_p", 0.9),
                "top_k": request_data.get("top_k", -1),
                "stop": request_data.get("stop", None),
            }
           
            # Use SGLang engine for generation - use async method
            result = await self.engine.async_generate(
                prompt=prompt,
                sampling_params=sampling_params
            )
           
            # Extract text from result
            if isinstance(result, dict):
                if "text" in result:
                    generated_text = result["text"]
                elif "outputs" in result and result["outputs"]:
                    generated_text = result["outputs"][0].get("text", str(result["outputs"][0]))
                else:
                    generated_text = str(result)
            else:
                generated_text = str(result)
                   
            # Format response similar to VLLMBackend's process_output
            return {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{
                    "text": generated_text,
                    "index": 0,
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),  # Rough estimate
                    "completion_tokens": len(generated_text.split()),  # Rough estimate
                    "total_tokens": len(prompt.split()) + len(generated_text.split()),
                }
            }
           
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {"error": f"Generation failed: {str(e)}"}


    async def encode(self, request_data: Dict[str, Any]):
        """Encode text to embeddings (if supported by the model)."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Engine is not running"}
           
        # SGLang doesn't typically support embeddings, return error
        return {"error": "Embeddings not supported by SGLang backend"}


    async def get_current_tokens(self) -> List[List[int]]:
        """Get current KV cache tokens."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []
           
        # Use SGLang engine to get current tokens if available
        try:
            if self.engine and hasattr(self.engine, 'get_kv_cache_tokens'):
                return self.engine.get_kv_cache_tokens()
            elif self.engine and hasattr(self.engine, 'get_current_tokens'):
                return self.engine.get_current_tokens()
            else:
                logger.info("KV cache token extraction not available in current SGLang version")
                return []
        except Exception as e:
            logger.warning(f"Failed to get current tokens: {e}")
       
        return []


    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        """Resume from KV cache state."""
        async with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return
           
        # Use SGLang engine to resume KV cache if available
        try:
            if self.engine and hasattr(self.engine, 'restore_kv_cache'):
                self.engine.restore_kv_cache(request_datas)
            elif self.engine and hasattr(self.engine, 'resume_kv_cache'):
                self.engine.resume_kv_cache(request_datas)
            else:
                logger.info("KV cache resume not supported by current SGLang version")
        except Exception as e:
            logger.warning(f"Failed to resume KV cache: {e}")


    async def fine_tuning(self, request_data: Dict[str, Any]):
        """Fine-tuning support (placeholder)."""
        logger.warning("Fine-tuning not supported in SGLang backend")
        return {"error": "Fine-tuning not supported"}


    async def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
        async with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
       
        # TODO: Wait for ongoing requests to finish
        logger.info("Stopping SGLang backend...")
        await self.shutdown()


    async def shutdown(self):
        """Abort all requests and shutdown the backend."""
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING


        try:
            logger.info("Shutting down SGLang backend...")
           
            # Shutdown SGLang engine with multiple fallback methods
            if self.engine:
                shutdown_success = False
               
                # Try shutdown method first
                if hasattr(self.engine, 'shutdown'):
                    try:
                        if asyncio.iscoroutinefunction(self.engine.shutdown):
                            await self.engine.shutdown()
                        else:
                            self.engine.shutdown()
                        shutdown_success = True
                        logger.info("Engine shutdown using shutdown() method")
                    except Exception as e:
                        logger.warning(f"Engine shutdown() failed: {e}")
               
                # Try stop method as fallback
                if not shutdown_success and hasattr(self.engine, 'stop'):
                    try:
                        if asyncio.iscoroutinefunction(self.engine.stop):
                            await self.engine.stop()
                        else:
                            self.engine.stop()
                        shutdown_success = True
                        logger.info("Engine shutdown using stop() method")
                    except Exception as e:
                        logger.warning(f"Engine stop() failed: {e}")
               
                # Force cleanup if normal shutdown fails
                if not shutdown_success:
                    logger.warning("Normal shutdown failed, forcing cleanup...")
                    try:
                        # Kill child processes if they exist
                        import psutil
                        import os
                        current_process = psutil.Process(os.getpid())
                        children = current_process.children(recursive=True)
                        for child in children:
                            try:
                                child.terminate()
                            except psutil.NoSuchProcess:
                                pass
                       
                        # Wait for children to terminate
                        psutil.wait_procs(children, timeout=3)
                       
                        # Kill any remaining children
                        for child in children:
                            try:
                                if child.is_running():
                                    child.kill()
                            except psutil.NoSuchProcess:
                                pass
                               
                    except ImportError:
                        logger.warning("psutil not available for process cleanup")
                    except Exception as e:
                        logger.warning(f"Process cleanup failed: {e}")
               
                del self.engine
                self.engine = None
               
            # Cleanup GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
               
            logger.info("SGLang engine shutdown complete")
           
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            self.status = BackendStatus.UNINITIALIZED


    async def chat_completion(self, request_data: Dict[str, Any]):
        """Chat completion using SGLang engine."""
        # Convert chat format to generation request
        messages = request_data.get("messages", [])
       
        # Construct prompt from messages
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            prompt_parts.append(f"{role}: {content}")
       
        prompt = "\n".join(prompt_parts)
        if prompt_parts:
            prompt += "\nassistant:"  # Add assistant prompt
       
        # Create generation request
        generation_request = {
            "prompt": prompt,
            "max_tokens": request_data.get("max_tokens", 100),
            "temperature": request_data.get("temperature", 0.7),
            "top_p": request_data.get("top_p", 0.9),
            "stop": request_data.get("stop", []),
        }
       
        # Use generate method
        result = await self.generate(generation_request)
       
        # Convert generation result to chat completion format
        if "error" in result:
            return result
       
        # Update response format for chat completion
        if "choices" in result and result["choices"]:
            result["object"] = "chat.completion"
            result["choices"][0]["message"] = {
                "role": "assistant",
                "content": result["choices"][0]["text"]
            }
            # Remove the text field as it's now in message.content
            del result["choices"][0]["text"]
       
        return result



