"""
Enhanced SGLang Backend for ServerlessLLM with Streaming Support
"""
import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Union

from sllm.serve.backends.backend_utils import BackendStatus, SllmBackend

# Optional imports
try:
    import sglang
    from sglang import (
        assistant,
        flush_cache,
        function,
        gen,
        select,
        set_default_backend,
        system,
        user,
    )
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False

logger = logging.getLogger("sglang_backend")


class SGLangMode(Enum):
    SERVER = "server"            # HTTP API
    NATIVE_ENDPOINT = "endpoint" # RuntimeEndpoint


@dataclass
class RequestMetrics:
    request_id: str
    start_time: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    success: bool = False
    method: str = "unknown"
    stream: bool = False


class OptimizedCache:
    def __init__(self, max_size: int = 500, ttl: float = 1800):
        self.cache: Dict[str, Dict] = {}
        self.access_times: Dict[str, float] = {}
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
        self._counter = 0

    def _key(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate cache key based on prompt and generation parameters"""
        cache_params = {
            "max_tokens": params.get("max_tokens", 128),
            "temperature": round(params.get("temperature", 0.7), 2),
            "top_p": round(params.get("top_p", 1.0), 2)
        }
        data = {"prompt": prompt[:100], **cache_params}  
        return hashlib.md5(str(sorted(data.items())).encode()).hexdigest()[:12]

    def get(self, prompt: str, params: Dict[str, Any]) -> Any:
        """Get cached response if available and not expired"""
        self._counter += 1
        if self._counter % 100 == 0:
            self._cleanup()
        k = self._key(prompt, params)
        entry = self.cache.get(k)
        if entry and time.time() - entry["ts"] < self.ttl:
            self.access_times[k] = time.time()
            self.hits += 1
            return entry["resp"]
        self.misses += 1
        return None

    def put(self, prompt: str, params: Dict[str, Any], resp: str):
        """Cache response with LRU eviction if necessary"""
        if len(self.cache) >= self.max_size:
            oldest = min(self.access_times, key=self.access_times.get)
            self.cache.pop(oldest, None)
            self.access_times.pop(oldest, None)
        k = self._key(prompt, params)
        self.cache[k] = {"resp": resp, "ts": time.time()}
        self.access_times[k] = time.time()

    def _cleanup(self):
        """Remove expired cache entries"""
        now = time.time()
        for k, e in list(self.cache.items()):
            if now - e["ts"] > self.ttl:
                self.cache.pop(k, None)
                self.access_times.pop(k, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        return {
            "size": len(self.cache), 
            "hits": self.hits,
            "misses": self.misses, 
            "hit_rate": self.hits / total if total else 0
        }

    def stats(self) -> Dict[str, Any]:
        """Alias for get_stats()"""
        return self.get_stats()


class SGLangBackend(SllmBackend):
    def __init__(self, model: str, backend_config: Dict[str, Any] = None):
        super().__init__(model)
        self.model_name = model
        cfg = backend_config or {}
        
        if not cfg.get("server_url"):
            raise ValueError("server_url is required in backend_config")
            
        self.server_url: str = cfg.get("server_url", "")
        self.timeout = cfg.get("timeout", 30)
        self.use_native = cfg.get("use_native_sglang", False)

        self.status = BackendStatus.UNINITIALIZED
        self.mode = None
        self.endpoint: RuntimeEndpoint = None
        self.cache = OptimizedCache(cfg.get("cache_size", 500), cfg.get("cache_ttl", 1800))

        self.metrics: Dict[str, RequestMetrics] = {}
        self.total_requests = 0
        self.successful = 0
        self.sglang_functions = {}
        self._register_functions()
        logger.info(f"SGLangBackend init: server_url={self.server_url}, use_native={self.use_native}")

    def _register_functions(self):
        """Register predefined SGLang functions for native mode"""
        if not SGLANG_AVAILABLE:
            return
        
        @function
        def simple_chat(s, msg):
            s += user(msg)
            s += assistant(gen("response", max_tokens=128, stop=["\n\n", "User:", "Assistant:"]))

        @function
        def code_gen(s, task):
            s += system("You are a helpful coding assistant.")
            s += user(task)
            s += assistant("```python\n" + gen("code", max_tokens=256, stop=["```", "\n\n"]) + "\n```")

        self.sglang_functions = {"simple_chat": simple_chat, "code_gen": code_gen}

    async def init_backend(self):
        """Initialize backend with native mode priority fix"""
        if not SGLANG_AVAILABLE:
            logger.error("SGLang not installed")
            raise RuntimeError("Missing sglang package")
        
        # Priority logic: if use_native=True, prioritize native endpoint
        if self.use_native:
            logger.info("🔧 Attempting native endpoint mode...")
            if await self._init_endpoint():
                self.mode = SGLangMode.NATIVE_ENDPOINT
                logger.info("✅ Using native endpoint mode")
            elif await self._test_http():
                self.mode = SGLangMode.SERVER
                logger.info("⚠️  Native failed, falling back to HTTP API mode")
            else:
                raise RuntimeError("Cannot connect to SGLang backend")
        else:
            # If use_native=False, try HTTP first
            if await self._test_http():
                self.mode = SGLangMode.SERVER
                logger.info("Using HTTP API mode")
            elif await self._init_endpoint():
                self.mode = SGLangMode.NATIVE_ENDPOINT
                logger.info("HTTP failed, trying native endpoint mode")
            else:
                raise RuntimeError("Cannot connect to SGLang backend")
        
        self.status = BackendStatus.RUNNING
        logger.info(f"Backend running in {self.mode.value} mode")

    async def _init_endpoint(self) -> bool:
        """Initialize native endpoint with enhanced debugging"""
        try:
            logger.info(f"🔗 Connecting to SGLang Runtime endpoint: {self.server_url}")
            
            # Try to create RuntimeEndpoint
            from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
            self.endpoint = RuntimeEndpoint(base_url=self.server_url)
            
            # Test connection
            logger.info("🧪 Testing RuntimeEndpoint connection...")
            info = await asyncio.to_thread(self.endpoint.get_server_info)
            logger.info(f"📊 Server info retrieved successfully")
            
            # Set default backend
            if set_default_backend:
                set_default_backend(self.endpoint)
                logger.info("✅ Set default SGLang backend")
            
            return True
            
        except ImportError as e:
            logger.warning(f"❌ RuntimeEndpoint import failed: {e}")
            return False
        except Exception as e:
            logger.warning(f"❌ Native endpoint initialization failed: {e}")
            return False

    async def _test_http(self) -> bool:
        """Test HTTP connection"""
        try:
            import aiohttp
            url = self.server_url.rstrip('/') + '/health'
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=5) as r:
                    return r.status == 200
        except Exception as e:
            logger.debug(f"HTTP test failed: {e}")
            return False

    def _extract_generation_params(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and validate generation parameters"""
        return {
            "messages": data.get("messages", []),
            "max_tokens": max(1, min(data.get("max_tokens", 128), 4096)),
            "temperature": max(0.0, min(data.get("temperature", 0.7), 2.0)),
            "top_p": max(0.0, min(data.get("top_p", 1.0), 1.0)),
            "stream": data.get("stream", False)
        }

    async def stream_generate(self, data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Streaming generation support"""
        if self.status != BackendStatus.RUNNING:
            yield "[ERROR] Backend not running"
            return
            
        params = self._extract_generation_params(data)
        prompt = self._format_prompt(params["messages"])
        rid = f"chatcmpl-{uuid.uuid4()}"
        
        self.total_requests += 1
        m = RequestMetrics(
            rid,
            time.time(),
            len(prompt.split()),
            method=self.mode.value if self.mode else "unknown",
            stream=True
        )
        self.metrics[rid] = m
        
        try:
            generated_text = ""
            method = "unknown"
            
            if self.mode == SGLangMode.NATIVE_ENDPOINT and self.endpoint:
                # Native streaming generation - SGLang may not support full streaming API yet
                # Implement non-streaming version but return in streaming format
                try:
                    logger.debug("🚀 Using native endpoint streaming generation...")
                    state = await asyncio.to_thread(self.sglang_functions['simple_chat'].run, msg=prompt)
                    
                    # Extract text
                    if hasattr(state, 'response'):
                        text = str(state.response).strip()
                    elif hasattr(state, '_variables') and 'response' in state._variables:
                        text = str(state._variables['response']).strip()
                    elif hasattr(state, 'resp'):
                        text = str(state.resp).strip()
                    else:
                        raw_text = str(state)
                        if "ASSISTANT:" in raw_text:
                            text = raw_text.split("ASSISTANT:")[-1].split("USER:")[0].strip()
                        else:
                            text = raw_text.strip()
                    
                    method = "native_sglang_stream"
                    
                    # Simulate streaming output (split by words)
                    words = text.split()
                    for i, word in enumerate(words):
                        chunk = word + (" " if i < len(words) - 1 else "")
                        generated_text += chunk
                        yield chunk
                        await asyncio.sleep(0.01)  # Small delay to simulate streaming
                        
                except Exception as e:
                    logger.warning(f"Native streaming generation failed: {e}, falling back to HTTP streaming")
                    async for chunk in self._http_stream_generate(prompt, params):
                        generated_text += chunk
                        yield chunk
                    method = "http_stream_fallback"
            else:
                # HTTP streaming generation
                async for chunk in self._http_stream_generate(prompt, params):
                    generated_text += chunk
                    yield chunk
                method = "http_stream"
            
            # Update metrics
            self.successful += 1
            m.success = True
            m.completion_tokens = len(generated_text.split())
            m.method = method
            
        except Exception as e:
            logger.error(f"Stream generation error: {e}")
            yield f"[ERROR] {str(e)}"
        finally:
            # Clean up metrics (after streaming completion)
            self.metrics.pop(rid, None)

    async def _http_stream_generate(self, prompt: str, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """HTTP streaming generation"""
        try:
            import aiohttp
            import json
            
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": params.get("max_tokens", 128),
                "temperature": params.get("temperature", 0.7),
                "top_p": params.get("top_p", 1.0),
                "stream": True  # Enable streaming
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.server_url.rstrip("/") + "/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        yield f"[ERROR] HTTP {response.status}: {error_text}"
                        return
                    
                    # Handle SSE streaming response
                    buffer = ""
                    async for chunk in response.content.iter_chunked(1024):
                        buffer += chunk.decode('utf-8')
                        
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            
                            if line.startswith('data: '):
                                if line == 'data: [DONE]':
                                    return
                                    
                                try:
                                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                                    if 'choices' in data and len(data['choices']) > 0:
                                        delta = data['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            yield delta['content']
                                except json.JSONDecodeError:
                                    continue
                                    
        except Exception as e:
            logger.error(f"HTTP streaming generation failed: {e}")
            # Fallback to non-streaming HTTP generation
            try:
                full_text = await self._http_generate(prompt, params)
                # Simulate streaming output
                words = full_text.split()
                for i, word in enumerate(words):
                    chunk = word + (" " if i < len(words) - 1 else "")
                    yield chunk
                    await asyncio.sleep(0.01)
            except Exception as fallback_error:
                yield f"[ERROR] Both streaming and non-streaming generation failed: {fallback_error}"

    async def generate(self, data: Dict[str, Any]) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Generate response - supports both streaming and non-streaming"""
        params = self._extract_generation_params(data)
        
        # Check if streaming is requested
        if params.get("stream", False):
            return self.stream_generate(data)
        
        # Non-streaming generation (preserve original logic)
        if self.status != BackendStatus.RUNNING:
            return {"error": "Backend not running"}

        prompt = self._format_prompt(params["messages"])
        rid = f"chatcmpl-{uuid.uuid4()}"
        self.total_requests += 1
        m = RequestMetrics(
            rid,
            time.time(),
            len(prompt.split()),
            method=self.mode.value if self.mode else "unknown"
        )
        self.metrics[rid] = m

        # Check cache (enabled for low temperature)
        if params['temperature'] < 0.5:
            cached = self.cache.get(prompt, params)
            if cached:
                m.success = True
                logger.debug(f"Cache hit for request {rid}")
                return self._format_openai_response({
                    "text": cached, 
                    "request_id": rid, 
                    "cached": True,
                    "prompt_tokens": m.prompt_tokens,
                    "completion_tokens": len(cached.split()),
                    "method": "cache"
                }, rid)

        try:
            text = None
            method = "unknown"
            
            # Fix: Better native mode handling
            if self.mode == SGLangMode.NATIVE_ENDPOINT and self.sglang_functions:
                try:
                    logger.debug(f"🚀 Using native SGLang generation...")
                    state = await asyncio.to_thread(self.sglang_functions['simple_chat'].run, msg=prompt)
                    
                    # Fix: Correctly extract text response
                    if hasattr(state, 'response'):
                        text = str(state.response).strip()
                    elif hasattr(state, '_variables') and 'response' in state._variables:
                        text = str(state._variables['response']).strip()
                    elif hasattr(state, 'resp'):
                        text = str(state.resp).strip()
                    else:
                        # Try to extract content after ASSISTANT
                        raw_text = str(state)
                        if "ASSISTANT:" in raw_text:
                            text = raw_text.split("ASSISTANT:")[-1].split("USER:")[0].strip()
                        else:
                            text = raw_text.strip()
                    
                    # Further clean response
                    if text.startswith("ProgramState(") or not text:
                        logger.warning("Native SGLang returned abnormal format, falling back to HTTP API")
                        text = await self._http_generate(prompt, params)
                        method = "http_api_fallback"
                    else:
                        method = "native_sglang"
                        logger.debug(f"✅ Native generation successful: {text[:50]}...")
                        
                except Exception as e:
                    logger.warning(f"Native SGLang failed: {e}, falling back to HTTP")
                    text = await self._http_generate(prompt, params)
                    method = "http_api_fallback"
            else:
                text = await self._http_generate(prompt, params)
                method = "http_api"
            
            # Validate response
            if not text or text.strip() == "":
                return {"error": "Empty response from SGLang"}
            
            self.successful += 1
            if params['temperature'] < 0.5:
                self.cache.put(prompt, params, text)
                logger.debug(f"Cached response for future requests")
            
            m.success = True
            m.completion_tokens = len(text.split())
            
            return self._format_openai_response({
                "text": text,
                "request_id": rid,
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "method": method,
                "cached": False
            }, rid)
            
        except Exception as e:
            logger.error(f"Generate error: {e}")
            return {"error": str(e)}

    async def _http_generate(self, prompt: str, params: Dict[str, Any]) -> str:
        """HTTP API generation"""
        import aiohttp
        payload = {
            "model": self.model_name, 
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": params.get("max_tokens", 128),
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 1.0)
        }
        async with aiohttp.ClientSession() as s:
            async with s.post(self.server_url.rstrip('/') + '/v1/chat/completions', 
                             json=payload, timeout=self.timeout) as r:
                if r.status != 200:
                    raise Exception(f"HTTP {r.status}: {await r.text()}")
                d = await r.json()
                if 'choices' not in d or len(d['choices']) == 0:
                    raise Exception("Invalid response format")
                return d['choices'][0]['message']['content']

    def _format_prompt(self, msgs: List[Dict[str, str]]) -> str:
        """Format prompt from messages"""
        return '\n'.join(f"{m['role'].title()}: {m['content']}" for m in msgs)

    def _format_openai_response(self, result: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Format OpenAI-compatible response"""
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result.get("text", "")
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
                "total_tokens": result.get("prompt_tokens", 0) + result.get("completion_tokens", 0)
            },
            "sglang_info": {
                "cached": result.get("cached", False),
                "mode": self.mode.value if self.mode else "unknown",
                "generation_method": result.get("method", "unknown"),
                "native_sglang": self.mode == SGLangMode.NATIVE_ENDPOINT,
                "function_used": result.get("function_used"),
                "streaming_supported": True  # New field
            }
        }

    def get_current_tokens(self) -> int:
        """Get current token count"""
        return sum(req.prompt_tokens + req.completion_tokens for req in self.metrics.values())

    async def stop(self, request_id: str = None) -> bool:
        """Stop request(s)"""
        if request_id:
            return self.metrics.pop(request_id, None) is not None
        self.metrics.clear()
        return True

    async def resume_kv_cache(self, data: List[List[int]]) -> None:
        """Resume KV cache - improved version"""
        # Only native mode supports prefill, HTTP mode not supported yet
        if self.mode != SGLangMode.NATIVE_ENDPOINT or not self.endpoint:
            logger.debug("KV cache prefill only supported in native mode")
            return

        if not data:
            logger.debug("No data to prefill")
            return

        try:
            # Concurrently start all prefill operations
            async def do_prefill(input_ids: List[int]):
                # Check if endpoint has prefill method
                if hasattr(self.endpoint, 'prefill'):
                    await asyncio.to_thread(self.endpoint.prefill, input_ids)
                else:
                    logger.warning("SGLang endpoint does not support prefill method")

            # Gather all requests
            await asyncio.gather(*(do_prefill(ids) for ids in data))
            logger.info(f"Successfully prefilled {len(data)} contexts to SGLang cache")
            
        except Exception as e:
            logger.error(f"KV cache prefill failed: {e}")
            # Don't raise exception as this is an optimization feature

    async def encode(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encoding (not supported yet)"""
        return {"error": "Encoding not supported"}

    async def fine_tuning(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Fine-tuning method implementation (SGLang doesn't support fine-tuning yet)"""
        return {"error": "Fine-tuning not supported by SGLang backend"}

    def get_backend_stats(self) -> Dict[str, Any]:
        """Get backend statistics"""
        uptime = time.time() - (min(m.start_time for m in self.metrics.values()) if self.metrics else time.time())
        
        # Count streaming requests
        stream_requests = sum(1 for m in self.metrics.values() if getattr(m, 'stream', False))
        
        return {
            "backend_type": "sglang",
            "status": self.status.value,
            "mode": self.mode.value if self.mode else "unknown",
            "model": self.model_name,
            "uptime": uptime,
            "total_requests": self.total_requests,
            "successful_requests": self.successful,
            "success_rate": self.successful / self.total_requests if self.total_requests > 0 else 0,
            "active_requests": len(self.metrics),
            "stream_requests": stream_requests,
            "cache_stats": self.cache.get_stats(),
            "features": {
                "streaming": True,
                "native_endpoint": self.mode == SGLangMode.NATIVE_ENDPOINT,
                "kv_cache": self.mode == SGLangMode.NATIVE_ENDPOINT,
                "sglang_functions": len(self.sglang_functions)
            }
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            "status": "healthy" if self.status == BackendStatus.RUNNING else "unhealthy",
            "mode": self.mode.value if self.mode else "unknown",
            "uptime": time.time() - (min(m.start_time for m in self.metrics.values()) if self.metrics else time.time()),
            "streaming_available": True,
            "native_endpoint_available": self.mode == SGLangMode.NATIVE_ENDPOINT
        }

    async def shutdown(self):
        """Shutdown backend"""
        await self.stop()
        if SGLANG_AVAILABLE and flush_cache:
            try:
                await asyncio.to_thread(flush_cache)
            except:
                pass
        self.status = BackendStatus.DELETING
        logger.info("SGLangBackend shutdown complete")