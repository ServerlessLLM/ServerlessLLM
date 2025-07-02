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
import signal
import time
import uuid
from typing import Any, Dict, List, Optional

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
    """SGLang backend implementation for SLLM, following VLLM backend pattern."""
   
    def __init__(self, model: str, backend_config: Optional[Dict[str, Any]] = None) -> None:
        if backend_config is None:
            raise ValueError("Backend config is missing")

        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.status_lock = asyncio.Lock()
        self.backend_config = backend_config
        self.model_name = model
        self.engine = None
        

        self._setup_model_config()
        self._setup_engine_config()
        
        logger.info(f"Initializing SGLang backend for model: {model}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Load format: {self.load_format}")

    def _setup_model_config(self) -> None:
        """Setup model path and load format - exactly like VLLM."""
        load_format = self.backend_config.get("load_format")
        
        if load_format is not None:
            self.load_format = load_format
            self.model_path = self.backend_config.get("pretrained_model_name_or_path")
        else:
            storage_path = os.getenv("STORAGE_PATH", "./models")
            self.model_path = os.path.join(storage_path, "sglang", self.model_name)
            self.load_format = "serverless_llm"

    def _setup_engine_config(self) -> None:
        """Setup SGLang engine configuration."""
        self.device = self.backend_config.get("device", "cuda")
        self.tp_size = self.backend_config.get("tp_size", 1)
        self.mem_fraction_static = self.backend_config.get("mem_fraction_static")
        self.host = self.backend_config.get("host")
        self.port = self.backend_config.get("port")

    def _build_engine_kwargs(self) -> Dict[str, Any]:
        """Build SGLang engine parameters."""
        engine_kwargs = {
            "model_path": self.model_path or self.model_name,
            "load_format": self.load_format,
            "device": self.device,
            "tp_size": self.tp_size,
        }
        
        # Add optional parameters only if explicitly set
        optional_params = {
            "mem_fraction_static": self.mem_fraction_static,
            "host": self.host,
            "port": self.port,
        }
        
        for key, value in optional_params.items():
            if value is not None:
                engine_kwargs[key] = value
        
        # Add backend-specific optional configurations
        optional_configs = [
            "disable_radix_cache", "enable_mixed_chunk", "context_length",
            "log_level", "is_embedding"
        ]
        
        for config_key in optional_configs:
            if config_key in self.backend_config:
                engine_kwargs[config_key] = self.backend_config[config_key]
        
        return engine_kwargs

    async def _check_running_status(self) -> bool:
        """Check if backend is running."""
        async with self.status_lock:
            return self.status == BackendStatus.RUNNING

    def _has_scheduler_with_requests(self) -> bool:
        """Check if engine has scheduler with running requests."""
        return (self.engine and 
                hasattr(self.engine, 'scheduler') and 
                hasattr(self.engine.scheduler, 'running_reqs'))

    def _get_running_requests(self) -> List:
        """Get running requests from scheduler."""
        if self._has_scheduler_with_requests():
            return getattr(self.engine.scheduler, 'running_reqs', [])
        return []

    async def init_backend(self) -> None:
        """Initialize the SGLang backend engine."""
        async with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return
           
            try:
                logger.info("Creating SGLang engine...")
                engine_kwargs = self._build_engine_kwargs()
                logger.info(f"SGLang engine parameters: {engine_kwargs}")
               
                # Set up signal handling for graceful shutdown
                def signal_handler(signum, frame):
                    logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
                    if self.engine:
                        try:
                            self.engine.shutdown()
                        except Exception as e:
                            logger.error(f"Error during engine shutdown: {e}")
                    raise KeyboardInterrupt(f"Received signal {signum}")
               
                # Install signal handlers
                original_sigint = signal.signal(signal.SIGINT, signal_handler)
                original_sigterm = signal.signal(signal.SIGTERM, signal_handler)
               
                try:
                    logger.info("Initializing SGLang engine...")
                    start_time = time.time()
                    self.engine = SGLEngine(**engine_kwargs)
                    init_time = time.time() - start_time
                    
                    logger.info(f"SGLang engine initialized successfully in {init_time:.2f} seconds")
                    self.status = BackendStatus.RUNNING
                   
                except KeyboardInterrupt:
                    logger.warning("Engine initialization interrupted by user")
                    self._cleanup_engine()
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

    def _cleanup_engine(self) -> None:
        """Cleanup engine resources."""
        if self.engine:
            try:
                self.engine.shutdown()
            except:
                pass
            self.engine = None
        self.status = BackendStatus.UNINITIALIZED

    async def generate(self, request_data: Dict[str, Any]):
        """Generate text using SGLang engine."""
        if not await self._check_running_status():
            return {"error": "Engine is not running"}

        if not request_data:
            return {"error": "Request data is missing"}

        model_name = request_data.pop("model", "sglang-model")
        messages = request_data.pop("messages", [])
        
        # Construct prompt from messages or use direct prompt
        if messages:
            prompt = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in messages if "content" in msg
            ])
        else:
            prompt = request_data.pop("prompt", "")
            
        request_id = request_data.pop("request_id", f"sglang-{uuid.uuid4()}")

        try:
            # Create SGLang sampling parameters
            sampling_params = {
                "max_new_tokens": request_data.get("max_tokens", request_data.get("max_new_tokens", 100)),
                "temperature": request_data.get("temperature", 0.7),
                "top_p": request_data.get("top_p", 0.9),
                "top_k": request_data.get("top_k", -1),
                "stop": request_data.get("stop", None),
            }
           
            # Generate using SGLang engine
            result = await self.engine.async_generate(prompt=prompt, sampling_params=sampling_params)
           
            # Extract generated text
            generated_text = self._extract_generated_text(result)
                   
            # Format response
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
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(generated_text.split()),
                    "total_tokens": len(prompt.split()) + len(generated_text.split()),
                }
            }
           
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {"error": f"Generation failed: {str(e)}"}

    def _extract_generated_text(self, result) -> str:
        """Extract generated text from SGLang result."""
        if isinstance(result, dict):
            if "text" in result:
                return result["text"]
            elif "outputs" in result and result["outputs"]:
                return result["outputs"][0].get("text", str(result["outputs"][0]))
        return str(result)

    async def encode(self, request_data: Dict[str, Any]):
        """Encode text to embeddings using SGLang engine."""
        if not await self._check_running_status():
            return {"error": "Engine is not running"}
    
        if not request_data:
            return {"error": "Request data is missing"}
    
        try:
            # Extract text from request
            text = request_data.get("input", request_data.get("text", request_data.get("prompt", "")))
            if not text:
                return {"error": "No text provided for encoding"}
            
            image_data = request_data.get("image_data")
            
            # Use SGLang engine's async_encode method
            result = await self.engine.async_encode(prompt=text, image_data=image_data)
            
            # Format response
            if isinstance(result, dict) and "embeddings" in result:
                return {
                    "object": "list",
                    "data": result["embeddings"],
                    "model": request_data.get("model", "sglang-model"),
                    "usage": {
                        "prompt_tokens": result.get("prompt_tokens", 0),
                        "total_tokens": result.get("total_tokens", 0),
                    }
                }
            return result
                
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return {"error": f"Encoding failed: {str(e)}"}

    async def get_current_tokens(self) -> List[List[int]]:
        """Return a list of all ongoing request tokens."""
        if not await self._check_running_status():
            return []
    
        try:
            running_reqs = self._get_running_requests()
            return [
                getattr(req, 'prompt_token_ids', []) + 
                (getattr(req.outputs[0], 'token_ids', []) 
                 if hasattr(req, 'outputs') and req.outputs else [])
                for req in running_reqs
            ]
        except Exception as e:
            logger.warning(f"Failed to get current tokens: {e}")
            return []

    async def resume_kv_cache(self, request_datas: List[List[int]]) -> None:
        """Resume from KV cache state by prefilling token sequences."""
        if not await self._check_running_status() or not request_datas:
            return

        try:
            
            constructed_inputs = []
            
            for request_data in request_datas:
                if not (request_data and isinstance(request_data, list) and len(request_data) > 0):
                    continue
                    
                # Decode tokens to text for SGLang
                if (self.engine and hasattr(self.engine, 'tokenizer') and 
                    hasattr(self.engine.tokenizer, 'decode')):
                    prompt_text = self.engine.tokenizer.decode(request_data, skip_special_tokens=True)
                else:
                    # Fallback: use token representation
                    prompt_text = f"tokens: {' '.join(str(token) for token in request_data[:50])}"
                
                constructed_inputs.append({
                    "prompt": prompt_text,
                    "max_tokens": 1,  # Minimal generation for cache warming
                    "temperature": 0.0,
                    "top_p": 1.0,
                })
        
            if constructed_inputs:
                # Execute all prefill requests concurrently
                tasks = [self.generate(inputs) for inputs in constructed_inputs]
                await asyncio.gather(*tasks, return_exceptions=True)
                logger.info(f"KV cache resumed for {len(constructed_inputs)} token sequences")
            
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

        # Wait for requests to finish
        while self._get_running_requests():
            logger.info("Waiting for all requests to finish")
            await asyncio.sleep(1)
    
        logger.info("All requests finished. Shutting down the backend.")
        await self.shutdown()

    async def shutdown(self):
        """Abort all requests and shutdown the backend."""
        async with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING

        # Abort running requests
        try:
            for req in self._get_running_requests():
                if hasattr(self.engine, 'abort') and hasattr(req, 'request_id'):
                    try:
                        await self.engine.abort(req.request_id)
                    except Exception as e:
                        logger.debug(f"Failed to abort request: {e}")
        except Exception as e:
            logger.debug(f"Failed to abort requests: {e}")
        
        # Shutdown engine
        if self.engine:
            try:
                shutdown_method = (getattr(self.engine, 'shutdown', None) or 
                                 getattr(self.engine, 'stop', None))
                if shutdown_method:
                    if asyncio.iscoroutinefunction(shutdown_method):
                        await shutdown_method()
                    else:
                        shutdown_method()
            except Exception as e:
                logger.warning(f"Engine shutdown failed: {e}")
            
            del self.engine
            self.engine = None
        
        # Cleanup GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.status = BackendStatus.UNINITIALIZED




