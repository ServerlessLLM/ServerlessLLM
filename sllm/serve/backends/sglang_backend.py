"""
Refined SGLang Backend for ServerlessLLM (without mock)
"""
import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from sllm.serve.backends.backend_utils import BackendStatus, SllmBackend

# Optional imports
try:
    import sglang
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
    from sglang import function, gen, select, assistant, system, user
    from sglang import flush_cache, set_default_backend
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
     
        cache_params = {
            "max_tokens": params.get("max_tokens", 128),
            "temperature": round(params.get("temperature", 0.7), 2),  # 四舍五入避免浮点精度问题
            "top_p": round(params.get("top_p", 1.0), 2)
        }
        data = {"prompt": prompt[:100], **cache_params}  # 缩短prompt长度
        return hashlib.md5(str(sorted(data.items())).encode()).hexdigest()[:12]

    def get(self, prompt: str, params: Dict[str, Any]) -> Any:
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
        if len(self.cache) >= self.max_size:
            oldest = min(self.access_times, key=self.access_times.get)
            self.cache.pop(oldest, None)
            self.access_times.pop(oldest, None)
        k = self._key(prompt, params)
        self.cache[k] = {"resp": resp, "ts": time.time()}
        self.access_times[k] = time.time()

    def _cleanup(self):
        now = time.time()
        for k, e in list(self.cache.items()):
            if now - e["ts"] > self.ttl:
                self.cache.pop(k, None)
                self.access_times.pop(k, None)

    def get_stats(self) -> Dict[str, Any]:
      
        total = self.hits + self.misses
        return {
            "size": len(self.cache), 
            "hits": self.hits,
            "misses": self.misses, 
            "hit_rate": self.hits / total if total else 0
        }

    def stats(self) -> Dict[str, Any]:
 
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
        self.use_native = cfg.get("use_native_sglang", False)  # 修复：默认使用HTTP模式

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
      
        if not SGLANG_AVAILABLE:
            logger.error("SGLang not installed")
            raise RuntimeError("Missing sglang package")
        
       
        if await self._test_http():
            self.mode = SGLangMode.SERVER
            logger.info("HTTP API")
        elif self.use_native and await self._init_endpoint():
            self.mode = SGLangMode.NATIVE_ENDPOINT
            logger.info("Native")
        else:
            raise RuntimeError("Cannot connect to SGLang backend")
        
        self.status = BackendStatus.RUNNING
        logger.info(f"Backend running in {self.mode.value} mode")

    async def _init_endpoint(self) -> bool:
        
        try:
            self.endpoint = RuntimeEndpoint(base_url=self.server_url)
            info = await asyncio.to_thread(self.endpoint.get_server_info)
            if set_default_backend:
                set_default_backend(self.endpoint)
            return True
        except Exception as e:
            logger.debug(f"Native endpoint init failed: {e}")
            return False

    async def _test_http(self) -> bool:
  
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
  
        return {
            "messages": data.get("messages", []),
            "max_tokens": max(1, min(data.get("max_tokens", 128), 4096)),
            "temperature": max(0.0, min(data.get("temperature", 0.7), 2.0)),
            "top_p": max(0.0, min(data.get("top_p", 1.0), 1.0))
        }

    async def generate(self, data: Dict[str, Any]) -> Dict[str, Any]:
     
        if self.status != BackendStatus.RUNNING:
            return {"error": "Backend not running"}
        
        params = self._extract_generation_params(data)
        prompt = self._format_prompt(params["messages"])
        rid = f"chatcmpl-{uuid.uuid4()}"
        self.total_requests += 1
        m = RequestMetrics(rid, time.time(), len(prompt.split()), method=self.mode.value if self.mode else "unknown")
        self.metrics[rid] = m

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
            
          
            if self.mode == SGLangMode.NATIVE_ENDPOINT and self.sglang_functions:
                try:
                    state = await asyncio.to_thread(self.sglang_functions['simple_chat'].run, msg=prompt)
                    
                 
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
                    
                   
                    if text.startswith("ProgramState("):
                      
                        logger.warning("Native SGLang返回格式异常，回退到HTTP API")
                        text = await self._http_generate(prompt, params)
                        method = "http_api_fallback"
                    else:
                        method = "native_sglang"
                        
                except Exception as e:
                    logger.warning(f"Native SGLang failed: {e}, falling back to HTTP")
                    text = await self._http_generate(prompt, params)
                    method = "http_api_fallback"
            else:
                text = await self._http_generate(prompt, params)
                method = "http_api"
            
            
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
       
        import aiohttp
        payload = {
            "model": self.model_name, 
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": params.get("max_tokens", 128),
            "temperature": params.get("temperature", 0.7)
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
       
        return '\n'.join(f"{m['role'].title()}: {m['content']}" for m in msgs)

    def _format_openai_response(self, result: Dict[str, Any], request_id: str) -> Dict[str, Any]:
       
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
                "function_used": result.get("function_used")
            }
        }

    def get_current_tokens(self) -> int:
    
        return sum(req.prompt_tokens + req.completion_tokens for req in self.metrics.values())

    async def stop(self, request_id: str = None) -> bool:
     
        if request_id:
            return self.metrics.pop(request_id, None) is not None
        self.metrics.clear()
        return True

    async def resume_kv_cache(self, data: List[List[int]]) -> None:
     
        logger.debug(f"Resume KV cache for {len(data)} requests")

    async def encode(self, data: Dict[str, Any]) -> Dict[str, Any]:
     
        return {"error": "Encoding not supported"}

    async def fine_tuning(self, data: Dict[str, Any]) -> Dict[str, Any]:
       
        return {"error": "Fine-tuning not supported by SGLang backend"}

    def get_backend_stats(self) -> Dict[str, Any]:
        
        uptime = time.time() - (min(m.start_time for m in self.metrics.values()) if self.metrics else time.time())
        
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
            "cache_stats": self.cache.get_stats()
        }

    def get_health_status(self) -> Dict[str, Any]:
       
        return {
            "status": "healthy" if self.status == BackendStatus.RUNNING else "unhealthy",
            "mode": self.mode.value if self.mode else "unknown",
            "uptime": time.time() - (min(m.start_time for m in self.metrics.values()) if self.metrics else time.time())
        }

    async def shutdown(self):
        
        await self.stop()
        if SGLANG_AVAILABLE and flush_cache:
            try:
                await asyncio.to_thread(flush_cache)
            except:
                pass
        self.status = BackendStatus.DELETING
        logger.info("SGLangBackend shutdown complete")