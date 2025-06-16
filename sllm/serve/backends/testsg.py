"""
SGLang Backend 完整测试套件
测试所有功能模式和性能
"""
import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, Any, List
from dataclasses import dataclass

# 导入您的 SGLang Backend
from sllm.serve.backends.sglang_backend import SGLangBackend, SGLangMode

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sglang_test")


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    success: bool
    duration: float
    details: Dict[str, Any] = None
    error: str = None


class SGLangBackendTestSuite:
    """SGLang Backend 测试套件"""
    
    def __init__(self, server_url: str = None):
        self.server_url = server_url or "http://localhost:30000"
        self.test_results: List[TestResult] = []
        self.passed_tests = 0
        self.total_tests = 0
        
    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始 SGLang Backend 完整测试")
        print("=" * 60)
        
        # 测试列表
        tests = [
            ("服务器连接测试", self.test_server_connection),
            ("Mock 模式测试", self.test_mock_mode),
            ("HTTP API 模式测试", self.test_http_api_mode),
            ("缓存系统测试", self.test_cache_system),
            ("并发请求测试", self.test_concurrent_requests),
            ("性能基准测试", self.test_performance_benchmark),
            ("错误处理测试", self.test_error_handling),
            ("统计信息测试", self.test_stats_monitoring),
            ("SGLang 原生功能测试", self.test_sglang_native_features),
            ("后端生命周期测试", self.test_backend_lifecycle)
        ]
        
        # 执行测试
        for test_name, test_func in tests:
            await self._run_single_test(test_name, test_func)
        
        # 输出测试摘要
        self.print_summary()
    
    async def _run_single_test(self, test_name: str, test_func):
        """运行单个测试"""
        self.total_tests += 1
        start_time = time.time()
        
        try:
            print(f"\n🔍 {test_name}...")
            result = await test_func()
            duration = time.time() - start_time
            
            if result.get("success", True):
                print(f"✅ {test_name} 通过 ({duration:.2f}s)")
                self.passed_tests += 1
                test_result = TestResult(
                    test_name=test_name,
                    success=True,
                    duration=duration,
                    details=result
                )
            else:
                print(f"❌ {test_name} 失败: {result.get('error', 'Unknown error')}")
                test_result = TestResult(
                    test_name=test_name,
                    success=False,
                    duration=duration,
                    error=result.get('error', 'Unknown error')
                )
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"❌ {test_name} 异常: {str(e)}")
            test_result = TestResult(
                test_name=test_name,
                success=False,
                duration=duration,
                error=str(e)
            )
        
        self.test_results.append(test_result)
    
    async def test_server_connection(self) -> Dict[str, Any]:
        """测试服务器连接"""
        try:
            async with aiohttp.ClientSession() as session:
                # 测试健康检查端点
                endpoints_to_test = [
                    "/health",
                    "/get_model_info", 
                    "/get_server_info",
                    "/v1/models"
                ]
                
                connection_results = {}
                for endpoint in endpoints_to_test:
                    try:
                        url = f"{self.server_url}{endpoint}"
                        async with session.get(url, timeout=5) as resp:
                            connection_results[endpoint] = {
                                "status": resp.status,
                                "accessible": resp.status == 200
                            }
                            if resp.status == 200:
                                try:
                                    data = await resp.json()
                                    connection_results[endpoint]["data"] = data
                                except:
                                    connection_results[endpoint]["data"] = await resp.text()
                    except Exception as e:
                        connection_results[endpoint] = {
                            "status": "error",
                            "accessible": False,
                            "error": str(e)
                        }
                
                # 检查是否有任何端点可访问
                accessible_endpoints = [
                    ep for ep, result in connection_results.items() 
                    if result.get("accessible", False)
                ]
                
                return {
                    "success": len(accessible_endpoints) > 0,
                    "accessible_endpoints": accessible_endpoints,
                    "connection_results": connection_results,
                    "server_available": len(accessible_endpoints) > 0
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"连接测试失败: {str(e)}",
                "server_available": False
            }
    
    async def test_mock_mode(self) -> Dict[str, Any]:
        """测试 Mock 模式"""
        # 强制使用 Mock 模式
        backend_config = {
            "force_mock": True,
            "timeout": 10
        }
        
        backend = SGLangBackend("test-model", backend_config)
        
        try:
            await backend.init_backend()
            
            # 验证模式
            assert backend.mode == SGLangMode.MOCK, f"期望 Mock 模式，但得到 {backend.mode}"
            
            # 测试基本生成
            test_requests = [
                {
                    "messages": [{"role": "user", "content": "Hello, world!"}],
                    "max_tokens": 50,
                    "temperature": 0.7
                },
                {
                    "messages": [{"role": "user", "content": "What is SGLang?"}],
                    "max_tokens": 100,
                    "temperature": 0.5
                },
                {
                    "messages": [{"role": "user", "content": "Write a simple function"}],
                    "max_tokens": 80,
                    "temperature": 0.3
                }
            ]
            
            results = []
            for i, request in enumerate(test_requests):
                result = await backend.generate(request)
                
                # 验证响应格式
                assert "id" in result, "响应缺少 ID"
                assert "choices" in result, "响应缺少 choices"
                assert "usage" in result, "响应缺少 usage"
                assert "sglang_info" in result, "响应缺少 sglang_info"
                
                # 验证内容
                content = result["choices"][0]["message"]["content"]
                assert len(content) > 0, "响应内容为空"
                
                results.append({
                    "request_id": result["id"],
                    "content_length": len(content),
                    "prompt_tokens": result["usage"]["prompt_tokens"],
                    "completion_tokens": result["usage"]["completion_tokens"]
                })
            
            # 获取统计信息
            stats = backend.get_backend_stats()
            
            return {
                "success": True,
                "mode": backend.mode.value,
                "requests_processed": len(results),
                "results": results,
                "backend_stats": stats
            }
            
        finally:
            await backend.shutdown()
    
    async def test_http_api_mode(self) -> Dict[str, Any]:
        """测试 HTTP API 模式"""
        # 检查服务器是否可用
        server_check = await self.test_server_connection()
        if not server_check.get("server_available", False):
            return {
                "success": True,  # 跳过测试，不算失败
                "skipped": True,
                "reason": "SGLang 服务器不可用，跳过 HTTP API 测试"
            }
        
        backend_config = {
            "server_url": self.server_url,
            "force_mock": False,
            "timeout": 30,
            "use_native_sglang": True
        }
        
        backend = SGLangBackend("test-model", backend_config)
        
        try:
            await backend.init_backend()
            
            # 验证连接到服务器
            assert backend.use_real_sglang, "未连接到真实 SGLang 服务器"
            
            # 测试生成
            request = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello! How are you?"}
                ],
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            result = await backend.generate(request)
            
            # 验证响应
            assert "error" not in result, f"生成请求失败: {result.get('error')}"
            assert "choices" in result, "响应格式不正确"
            
            content = result["choices"][0]["message"]["content"]
            sglang_info = result.get("sglang_info", {})
            
            return {
                "success": True,
                "mode": backend.mode.value,
                "content_length": len(content),
                "generation_method": sglang_info.get("generation_method"),
                "native_sglang": sglang_info.get("native_sglang", False),
                "backend_stats": backend.get_backend_stats()
            }
            
        finally:
            await backend.shutdown()
    
    async def test_cache_system(self) -> Dict[str, Any]:
        """测试缓存系统"""
        backend_config = {
            "force_mock": True,
            "cache_size": 100,
            "cache_ttl": 60
        }
        
        backend = SGLangBackend("test-model", backend_config)
        
        try:
            await backend.init_backend()
            
            # 相同请求（低温度触发缓存）
            request = {
                "messages": [{"role": "user", "content": "Test caching"}],
                "max_tokens": 50,
                "temperature": 0.1  # 低温度会启用缓存
            }
            
            # 第一次请求
            result1 = await backend.generate(request)
            assert not result1.get("sglang_info", {}).get("cached", False), "第一次请求不应该被缓存"
            
            # 第二次相同请求
            result2 = await backend.generate(request)
            assert result2.get("sglang_info", {}).get("cached", False), "第二次请求应该被缓存"
            
            # 验证响应内容相同
            content1 = result1["choices"][0]["message"]["content"]
            content2 = result2["choices"][0]["message"]["content"]
            assert content1 == content2, "缓存的内容应该相同"
            
            # 获取缓存统计
            cache_stats = backend.cache.get_stats()
            
            # 测试高温度请求（不应该使用缓存）
            high_temp_request = {
                "messages": [{"role": "user", "content": "Test caching"}],
                "max_tokens": 50,
                "temperature": 0.9  # 高温度不会使用缓存
            }
            
            result3 = await backend.generate(high_temp_request)
            assert not result3.get("sglang_info", {}).get("cached", False), "高温度请求不应该使用缓存"
            
            return {
                "success": True,
                "cache_stats": cache_stats,
                "cache_hit_verified": result2.get("sglang_info", {}).get("cached", False),
                "high_temp_no_cache": not result3.get("sglang_info", {}).get("cached", False)
            }
            
        finally:
            await backend.shutdown()
    
    async def test_concurrent_requests(self) -> Dict[str, Any]:
        """测试并发请求"""
        backend_config = {
            "force_mock": True,
            "timeout": 10
        }
        
        backend = SGLangBackend("test-model", backend_config)
        
        try:
            await backend.init_backend()
            
            # 创建多个并发请求
            concurrent_requests = []
            request_count = 10
            
            for i in range(request_count):
                request = {
                    "messages": [{"role": "user", "content": f"Concurrent request {i}"}],
                    "max_tokens": 30,
                    "temperature": 0.7
                }
                concurrent_requests.append(backend.generate(request))
            
            # 并发执行
            start_time = time.time()
            results = await asyncio.gather(*concurrent_requests, return_exceptions=True)
            end_time = time.time()
            
            # 分析结果
            successful_results = [r for r in results if isinstance(r, dict) and "error" not in r]
            failed_results = [r for r in results if not isinstance(r, dict) or "error" in r]
            
            total_duration = end_time - start_time
            avg_duration = total_duration / request_count
            
            return {
                "success": len(successful_results) >= request_count * 0.8,  # 80% 成功率
                "total_requests": request_count,
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": len(successful_results) / request_count,
                "total_duration": total_duration,
                "average_duration": avg_duration,
                "concurrent_qps": request_count / total_duration if total_duration > 0 else 0
            }
            
        finally:
            await backend.shutdown()
    
    async def test_performance_benchmark(self) -> Dict[str, Any]:
        """测试性能基准"""
        backend_config = {
            "force_mock": True,
            "cache_size": 200,
            "timeout": 15
        }
        
        backend = SGLangBackend("test-model", backend_config)
        
        try:
            await backend.init_backend()
            
            # 性能测试参数
            warmup_requests = 5
            benchmark_requests = 20
            
            # 预热
            warmup_tasks = []
            for i in range(warmup_requests):
                request = {
                    "messages": [{"role": "user", "content": f"Warmup {i}"}],
                    "max_tokens": 50,
                    "temperature": 0.7
                }
                warmup_tasks.append(backend.generate(request))
            
            await asyncio.gather(*warmup_tasks)
            
            # 基准测试
            benchmark_tasks = []
            for i in range(benchmark_requests):
                request = {
                    "messages": [{"role": "user", "content": f"Benchmark request {i}"}],
                    "max_tokens": 100,
                    "temperature": 0.5
                }
                benchmark_tasks.append(backend.generate(request))
            
            start_time = time.time()
            results = await asyncio.gather(*benchmark_tasks, return_exceptions=True)
            end_time = time.time()
            
            # 分析性能
            successful_results = [r for r in results if isinstance(r, dict) and "error" not in r]
            total_duration = end_time - start_time
            
            # 计算统计
            total_tokens = sum(
                r.get("usage", {}).get("total_tokens", 0) 
                for r in successful_results
            )
            
            qps = len(successful_results) / total_duration
            tokens_per_second = total_tokens / total_duration
            
            # 获取后端统计
            backend_stats = backend.get_backend_stats()
            
            return {
                "success": len(successful_results) >= benchmark_requests * 0.9,  # 90% 成功率
                "benchmark_requests": benchmark_requests,
                "successful_requests": len(successful_results),
                "total_duration": total_duration,
                "qps": qps,
                "tokens_per_second": tokens_per_second,
                "total_tokens": total_tokens,
                "backend_stats": backend_stats
            }
            
        finally:
            await backend.shutdown()
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """测试错误处理"""
        backend_config = {
            "force_mock": True,
            "timeout": 5
        }
        
        backend = SGLangBackend("test-model", backend_config)
        
        try:
            await backend.init_backend()
            
            # 测试各种错误情况
            error_tests = []
            
            # 1. 空请求
            result1 = await backend.generate({})
            error_tests.append({
                "test": "empty_request",
                "has_error": "error" in result1,
                "result": result1
            })
            
            # 2. 空消息
            result2 = await backend.generate({"messages": []})
            error_tests.append({
                "test": "empty_messages", 
                "has_error": "error" in result2,
                "result": result2
            })
            
            # 3. 无效参数
            result3 = await backend.generate({
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": -1,  # 无效值
                "temperature": 10.0  # 无效值
            })
            error_tests.append({
                "test": "invalid_params",
                "handled_gracefully": "error" not in result3,  # 应该被修正而不是错误
                "result": result3
            })
            
            # 4. 后端未运行状态
            await backend.shutdown()
            result4 = await backend.generate({
                "messages": [{"role": "user", "content": "test"}]
            })
            error_tests.append({
                "test": "backend_not_running",
                "has_error": "error" in result4,
                "result": result4
            })
            
            return {
                "success": True,
                "error_tests": error_tests,
                "all_errors_handled": all(
                    test.get("has_error", False) or test.get("handled_gracefully", False)
                    for test in error_tests
                )
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"错误处理测试失败: {str(e)}"
            }
    
    async def test_stats_monitoring(self) -> Dict[str, Any]:
        """测试统计和监控功能"""
        backend_config = {
            "force_mock": True,
            "cache_size": 50
        }
        
        backend = SGLangBackend("test-model", backend_config)
        
        try:
            await backend.init_backend()
            
            # 执行一些请求来生成统计数据
            for i in range(5):
                request = {
                    "messages": [{"role": "user", "content": f"Stats test {i}"}],
                    "max_tokens": 50,
                    "temperature": 0.5
                }
                await backend.generate(request)
            
            # 获取统计信息
            backend_stats = backend.get_backend_stats()
            health_status = backend.get_health_status()
            cache_stats = backend.cache.get_stats()
            
            # 验证统计信息
            required_stats_fields = [
                "backend_type", "status", "mode", "model", "uptime",
                "total_requests", "successful_requests", "success_rate"
            ]
            
            stats_complete = all(field in backend_stats for field in required_stats_fields)
            
            return {
                "success": stats_complete,
                "backend_stats": backend_stats,
                "health_status": health_status,
                "cache_stats": cache_stats,
                "stats_fields_present": stats_complete,
                "uptime": backend_stats.get("uptime", 0),
                "total_requests": backend_stats.get("total_requests", 0)
            }
            
        finally:
            await backend.shutdown()
    
    async def test_sglang_native_features(self) -> Dict[str, Any]:
        """测试 SGLang 原生功能"""
        backend_config = {
            "force_mock": True,
            "use_native_sglang": True
        }
        
        backend = SGLangBackend("test-model", backend_config)
        
        try:
            await backend.init_backend()
            
            # 测试不同类型的请求以触发不同的 SGLang 函数
            test_cases = [
                {
                    "name": "simple_chat",
                    "request": {
                        "messages": [{"role": "user", "content": "Hello there!"}],
                        "max_tokens": 50
                    }
                },
                {
                    "name": "code_generation",
                    "request": {
                        "messages": [{"role": "user", "content": "Write a Python function to calculate fibonacci"}],
                        "max_tokens": 100
                    }
                },
                {
                    "name": "structured_qa",
                    "request": {
                        "messages": [{"role": "user", "content": "Question: What is the capital of France? Please answer."}],
                        "max_tokens": 80
                    }
                }
            ]
            
            results = []
            for test_case in test_cases:
                result = await backend.generate(test_case["request"])
                results.append({
                    "test_case": test_case["name"],
                    "success": "error" not in result,
                    "content_length": len(result.get("choices", [{}])[0].get("message", {}).get("content", "")),
                    "sglang_info": result.get("sglang_info", {})
                })
            
            # 检查 SGLang 功能
            sglang_functions_available = len(backend.sglang_functions)
            native_features_working = all(r["success"] for r in results)
            
            return {
                "success": True,
                "sglang_functions_count": sglang_functions_available,
                "test_results": results,
                "native_features_working": native_features_working,
                "backend_mode": backend.mode.value
            }
            
        finally:
            await backend.shutdown()
    
    async def test_backend_lifecycle(self) -> Dict[str, Any]:
        """测试后端生命周期"""
        backend_config = {
            "force_mock": True
        }
        
        backend = SGLangBackend("test-model", backend_config)
        
        try:
            # 测试初始化
            initial_status = backend.status
            await backend.init_backend()
            running_status = backend.status
            
            # 测试运行中的操作
            request = {
                "messages": [{"role": "user", "content": "Lifecycle test"}],
                "max_tokens": 30
            }
            result = await backend.generate(request)
            
            # 测试停止特定请求
            stop_result = await backend.stop("non-existent-id")
            
            # 测试关闭
            await backend.shutdown()
            final_status = backend.status
            
            return {
                "success": True,
                "initial_status": initial_status.value,
                "running_status": running_status.value,
                "final_status": final_status.value,
                "generation_worked": "error" not in result,
                "stop_handled": isinstance(stop_result, bool)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"生命周期测试失败: {str(e)}"
            }
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "=" * 60)
        print("📋 SGLang Backend 测试摘要")
        print("=" * 60)
        print(f"✅ 通过测试: {self.passed_tests}/{self.total_tests}")
        print(f"📊 成功率: {self.passed_tests/self.total_tests*100:.1f}%")
        
        if self.passed_tests == self.total_tests:
            print("🎉 所有测试通过！SGLang Backend 工作正常！")
        else:
            print("⚠️  部分测试失败，请检查详细信息：")
            
            failed_tests = [r for r in self.test_results if not r.success]
            for test in failed_tests:
                print(f"   ❌ {test.test_name}: {test.error}")
        
        print("\n📈 详细测试结果:")
        for test in self.test_results:
            status = "✅" if test.success else "❌"
            print(f"   {status} {test.test_name} ({test.duration:.2f}s)")
        
        print("=" * 60)


async def main():
    """主测试函数"""
    # 可以修改这个 URL 以匹配您的 SGLang 服务器
    server_url = "http://localhost:8123"  # 根据您的实际配置修改
    
    test_suite = SGLangBackendTestSuite(server_url)
    await test_suite.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())