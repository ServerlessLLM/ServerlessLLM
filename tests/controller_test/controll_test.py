import pytest
import ray
import re
import time 
import json
import asyncio
import os
from threading import Thread



import subprocess
import threading

# pytest_plugins = ('pytest_asyncio',)

@pytest.fixture(scope="session")
def setup_environment():
    # 启动 Ray cluster
    ray.init(ignore_reinit_error=True)
    # 获取当前 Conda 环境的 bin 目录
    conda_bin_dir = os.path.join(os.environ.get('CONDA_PREFIX', ''), 'bin')
    
    # 更新环境变量以包括 Conda 环境的路径
    env = os.environ.copy()
    env['PATH'] = conda_bin_dir + os.pathsep + env['PATH']
    # 启动第一个外部程序
    server_proc = subprocess.Popen(
        ['sllm-store-server'],
        stderr=subprocess.PIPE,
        text=True
    )

    time.sleep(10)
    # 等待第一个外部程序启动完成
    for line in server_proc.stderr:
        if "Server listening on" in line:
            break

    # 启动第二个外部程序
    serve_proc = subprocess.Popen(
        ['sllm-serve', 'start'],
        stderr=subprocess.PIPE,
        text=True
    )

    time.sleep(10)
    # 等待第二个外部程序启动完成
    for line in serve_proc.stderr:
        if "Uvicorn running on" in line:
            break

    

    # 在测试完成后关闭外部程序和 Ray cluster
    yield

   

    # 终止外部程序
    server_proc.terminate()
    serve_proc.terminate()

    # 确保进程结束
    server_proc.wait()
    serve_proc.wait()
    # 关闭 Ray cluster
    ray.shutdown()

def test_something(setup_environment):
    # 这里可以进行测试
    pass