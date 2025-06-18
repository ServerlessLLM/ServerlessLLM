import os
import re
import ast
from pathlib import Path

def find_vllm_backend_usage():
    """查找 vLLM Backend 在项目中的使用情况"""
    
    serverless_root = Path("/home/fiona/serverlessllm")
    vllm_usages = []
    
    # 搜索模式
    patterns = [
        r'VllmBackend',
        r'vllm_backend',
        r'backend.*vllm',
        r'vllm.*backend',
        r'from.*vllm_backend',
        r'import.*VllmBackend',
    ]
    
    # 排除目录
    exclude_dirs = {'.git', '__pycache__', '.vscode', '.pytest_cache', 'node_modules'}
    
    print("🔍 搜索 vLLM Backend 使用情况...")
    print("=" * 60)
    
    for file_path in serverless_root.rglob("*.py"):
        if any(exclude_dir in str(file_path) for exclude_dir in exclude_dirs):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    for pattern in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            rel_path = file_path.relative_to(serverless_root)
                            vllm_usages.append({
                                'file': str(rel_path),
                                'line': i,
                                'content': line.strip(),
                                'pattern': pattern
                            })
                            
        except Exception as e:
            continue
    
    # 按文件分组显示结果
    files_with_usage = {}
    for usage in vllm_usages:
        file_path = usage['file']
        if file_path not in files_with_usage:
            files_with_usage[file_path] = []
        files_with_usage[file_path].append(usage)
    
    print(f"📊 找到 {len(vllm_usages)} 个 vLLM Backend 引用，分布在 {len(files_with_usage)} 个文件中")
    print()
    
    for file_path, usages in sorted(files_with_usage.items()):
        print(f"📁 {file_path}")
        for usage in usages:
            print(f"   🔹 第{usage['line']}行: {usage['content']}")
        print()
    
    return files_with_usage

def analyze_backend_registry():
    """分析 Backend 注册机制"""
    print("\n🔧 分析 Backend 注册机制...")
    print("=" * 60)
    
    # 查找 backend registry 相关文件
    backend_files = [
        "/home/fiona/serverlessllm/sllm/serve/backends/__init__.py",
        "/home/fiona/serverlessllm/sllm/serve/backends/backend_utils.py"
    ]
    
    for file_path in backend_files:
        if os.path.exists(file_path):
            print(f"\n📄 分析文件: {file_path}")
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    
                # 查找导入语句
                import_lines = [line.strip() for line in content.split('\n') 
                              if 'import' in line and ('Backend' in line or 'backend' in line)]
                
                if import_lines:
                    print("📥 导入语句:")
                    for line in import_lines:
                        print(f"   {line}")
                
                # 查找 __all__ 定义
                all_lines = [line.strip() for line in content.split('\n') 
                           if '__all__' in line]
                
                if all_lines:
                    print("📝 __all__ 定义:")
                    for line in all_lines:
                        print(f"   {line}")
                        
            except Exception as e:
                print(f"❌ 读取文件失败: {e}")

def find_backend_factory_pattern():
    """查找 Backend 工厂模式或创建模式"""
    print("\n🏭 查找 Backend 创建模式...")
    print("=" * 60)
    
    serverless_root = Path("/home/fiona/serverlessllm")
    
    factory_patterns = [
        r'create.*backend',
        r'get.*backend',
        r'Backend.*create',
        r'backend.*factory',
        r'BACKEND_REGISTRY',
        r'backend.*registry',
    ]
    
    for file_path in serverless_root.rglob("*.py"):
        if '.git' in str(file_path) or '__pycache__' in str(file_path):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    for pattern in factory_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            rel_path = file_path.relative_to(serverless_root)
                            print(f"🔍 {rel_path}:{i}")
                            print(f"   {line.strip()}")
                            
                            # 显示上下文
                            context_start = max(0, i-3)
                            context_end = min(len(lines), i+2)
                            print("   上下文:")
                            for j in range(context_start, context_end):
                                marker = "➤ " if j == i-1 else "   "
                                print(f"   {marker}{j+1}: {lines[j]}")
                            print()
                            
        except Exception as e:
            continue

def analyze_config_usage():
    """分析配置文件中如何指定 backend"""
    print("\n⚙️ 分析配置文件中的 Backend 使用...")
    print("=" * 60)
    
    serverless_root = Path("/home/fiona/serverlessllm")
    
    config_patterns = [
        r'"backend"',
        r"'backend'",
        r'backend.*:',
        r'backend.*=',
    ]
    
    config_files = list(serverless_root.rglob("*.json")) + \
                   list(serverless_root.rglob("*.yaml")) + \
                   list(serverless_root.rglob("*.yml")) + \
                   list(serverless_root.rglob("config*.py"))
    
    for file_path in config_files:
        if '.git' in str(file_path):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
                found_backend_config = False
                for i, line in enumerate(lines, 1):
                    for pattern in config_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            if not found_backend_config:
                                rel_path = file_path.relative_to(serverless_root)
                                print(f"📁 {rel_path}")
                                found_backend_config = True
                            print(f"   第{i}行: {line.strip()}")
                
                if found_backend_config:
                    print()
                    
        except Exception as e:
            continue

def main():
    """主函数"""
    print("🔍 ServerlessLLM vLLM Backend 使用情况分析")
    print("=" * 70)
    
    # 1. 查找 vLLM Backend 使用情况
    vllm_usages = find_vllm_backend_usage()
    
    # 2. 分析 Backend 注册机制
    analyze_backend_registry()
    
    # 3. 查找 Backend 工厂模式
    find_backend_factory_pattern()
    
    # 4. 分析配置文件
    analyze_config_usage()
    
    print("\n📋 分析总结:")
    print("=" * 70)
    print("基于以上分析，SGLang Backend 需要集成到以下位置:")
    print("1. 📁 sllm/serve/backends/__init__.py - 添加导入和注册")
    print("2. 🏭 Backend 工厂/注册表 - 使 SGLang 可被发现")
    print("3. ⚙️ 配置文件模板 - 添加 SGLang 配置示例")
    print("4. 📚 文档 - 更新支持的 Backend 列表")

if __name__ == "__main__":
    main()