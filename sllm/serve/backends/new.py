import ast
import os
import re
from pathlib import Path


def find_vllm_backend_usage():
    """Find vLLM Backend usage throughout the project"""

    serverless_root = Path("/home/fiona/serverlessllm")
    vllm_usages = []

    # Search patterns
    patterns = [
        r"VllmBackend",
        r"vllm_backend",
        r"backend.*vllm",
        r"vllm.*backend",
        r"from.*vllm_backend",
        r"import.*VllmBackend",
    ]

    # Exclude directories
    exclude_dirs = {
        ".git",
        "__pycache__",
        ".vscode",
        ".pytest_cache",
        "node_modules",
    }

    print("🔍 Searching for vLLM Backend usage...")
    print("=" * 60)

    for file_path in serverless_root.rglob("*.py"):
        if any(exclude_dir in str(file_path) for exclude_dir in exclude_dirs):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

                for i, line in enumerate(lines, 1):
                    for pattern in patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            rel_path = file_path.relative_to(serverless_root)
                            vllm_usages.append(
                                {
                                    "file": str(rel_path),
                                    "line": i,
                                    "content": line.strip(),
                                    "pattern": pattern,
                                }
                            )

        except Exception as e:
            continue

    # Group results by file
    files_with_usage = {}
    for usage in vllm_usages:
        file_path = usage["file"]
        if file_path not in files_with_usage:
            files_with_usage[file_path] = []
        files_with_usage[file_path].append(usage)

    print(
        f"📊 Found {len(vllm_usages)} vLLM Backend references across {len(files_with_usage)} files"
    )
    print()

    for file_path, usages in sorted(files_with_usage.items()):
        print(f"📁 {file_path}")
        for usage in usages:
            print(f"   🔹 Line {usage['line']}: {usage['content']}")
        print()

    return files_with_usage


def analyze_backend_registry():
    """Analyze Backend registration mechanism"""
    print("\n🔧 Analyzing Backend registration mechanism...")
    print("=" * 60)

    # Look for backend registry related files
    backend_files = [
        "/home/fiona/serverlessllm/sllm/serve/backends/__init__.py",
        "/home/fiona/serverlessllm/sllm/serve/backends/backend_utils.py",
    ]

    for file_path in backend_files:
        if os.path.exists(file_path):
            print(f"\n📄 Analyzing file: {file_path}")
            try:
                with open(file_path, "r") as f:
                    content = f.read()

                # Find import statements
                import_lines = [
                    line.strip()
                    for line in content.split("\n")
                    if "import" in line
                    and ("Backend" in line or "backend" in line)
                ]

                if import_lines:
                    print("📥 Import statements:")
                    for line in import_lines:
                        print(f"   {line}")

                # Find __all__ definitions
                all_lines = [
                    line.strip()
                    for line in content.split("\n")
                    if "__all__" in line
                ]

                if all_lines:
                    print("📝 __all__ definitions:")
                    for line in all_lines:
                        print(f"   {line}")

            except Exception as e:
                print(f"❌ Failed to read file: {e}")


def find_backend_factory_pattern():
    """Find Backend factory patterns or creation patterns"""
    print("\n🏭 Searching for Backend creation patterns...")
    print("=" * 60)

    serverless_root = Path("/home/fiona/serverlessllm")

    factory_patterns = [
        r"create.*backend",
        r"get.*backend",
        r"Backend.*create",
        r"backend.*factory",
        r"BACKEND_REGISTRY",
        r"backend.*registry",
    ]

    for file_path in serverless_root.rglob("*.py"):
        if ".git" in str(file_path) or "__pycache__" in str(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

                for i, line in enumerate(lines, 1):
                    for pattern in factory_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            rel_path = file_path.relative_to(serverless_root)
                            print(f"🔍 {rel_path}:{i}")
                            print(f"   {line.strip()}")

                            # Show context
                            context_start = max(0, i - 3)
                            context_end = min(len(lines), i + 2)
                            print("   Context:")
                            for j in range(context_start, context_end):
                                marker = "➤ " if j == i - 1 else "   "
                                print(f"   {marker}{j+1}: {lines[j]}")
                            print()

        except Exception as e:
            continue


def analyze_config_usage():
    """Analyze how backends are specified in configuration files"""
    print("\n⚙️ Analyzing Backend usage in configuration files...")
    print("=" * 60)

    serverless_root = Path("/home/fiona/serverlessllm")

    config_patterns = [
        r'"backend"',
        r"'backend'",
        r"backend.*:",
        r"backend.*=",
    ]

    config_files = (
        list(serverless_root.rglob("*.json"))
        + list(serverless_root.rglob("*.yaml"))
        + list(serverless_root.rglob("*.yml"))
        + list(serverless_root.rglob("config*.py"))
    )

    for file_path in config_files:
        if ".git" in str(file_path):
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

                found_backend_config = False
                for i, line in enumerate(lines, 1):
                    for pattern in config_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            if not found_backend_config:
                                rel_path = file_path.relative_to(
                                    serverless_root
                                )
                                print(f"📁 {rel_path}")
                                found_backend_config = True
                            print(f"   Line {i}: {line.strip()}")

                if found_backend_config:
                    print()

        except Exception as e:
            continue


def analyze_test_integration():
    """Analyze how backends are tested and integrated"""
    print("\n🧪 Analyzing Backend test integration...")
    print("=" * 60)

    serverless_root = Path("/home/fiona/serverlessllm")

    test_patterns = [
        r"backend.*test",
        r"test.*backend",
        r"Backend.*Test",
        r"TestBackend",
    ]

    test_dirs = ["tests", "test"]

    for test_dir in test_dirs:
        test_path = serverless_root / test_dir
        if test_path.exists():
            print(f"📁 Found test directory: {test_path}")

            for file_path in test_path.rglob("*.py"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        lines = content.split("\n")

                        found_backend_test = False
                        for i, line in enumerate(lines, 1):
                            for pattern in test_patterns:
                                if re.search(pattern, line, re.IGNORECASE):
                                    if not found_backend_test:
                                        rel_path = file_path.relative_to(
                                            serverless_root
                                        )
                                        print(f"🧪 {rel_path}")
                                        found_backend_test = True
                                    print(f"   Line {i}: {line.strip()}")

                        if found_backend_test:
                            print()

                except Exception as e:
                    continue


def main():
    """Main analysis function"""
    print("🔍 ServerlessLLM vLLM Backend Usage Analysis")
    print("=" * 70)

    # 1. Find vLLM Backend usage
    vllm_usages = find_vllm_backend_usage()

    # 2. Analyze Backend registration mechanism
    analyze_backend_registry()

    # 3. Find Backend factory patterns
    find_backend_factory_pattern()

    # 4. Analyze configuration files
    analyze_config_usage()

    # 5. Analyze test integration
    analyze_test_integration()

    print("\n📋 Analysis Summary:")
    print("=" * 70)
    print(
        "Based on the analysis above, SGLang Backend needs to be integrated at:"
    )
    print("1. 📁 sllm/serve/backends/__init__.py - Add import and registration")
    print("2. 🏭 Backend factory/registry - Make SGLang discoverable")
    print("3. ⚙️ Configuration templates - Add SGLang config examples")
    print("4. 🧪 Test suites - Add SGLang to backend tests")
    print("5. 📚 Documentation - Update supported backend lists")

    print("\n🚀 Next Steps:")
    print("1. Run this analysis to understand current vLLM integration")
    print("2. Follow the integration plan to add SGLang Backend")
    print("3. Test the integration with unit and integration tests")
    print("4. Update documentation and examples")


if __name__ == "__main__":
    main()
