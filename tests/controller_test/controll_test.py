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
    # start ray cluster
    ray.init(ignore_reinit_error=True)
    conda_bin_dir = os.path.join(os.environ.get('CONDA_PREFIX', ''), 'bin')
    env = os.environ.copy()
    env['PATH'] = conda_bin_dir + os.pathsep + env['PATH']
    # start sllm store server
    server_proc = subprocess.Popen(
        ['sllm-store-server'],
        stderr=subprocess.PIPE,
        text=True
    )

    time.sleep(10)
    for line in server_proc.stderr:
        if "Server listening on" in line:
            break

    # start sllm serve
    serve_proc = subprocess.Popen(
        ['sllm-serve', 'start'],
        stderr=subprocess.PIPE,
        text=True
    )

    time.sleep(10)
    for line in serve_proc.stderr:
        if "Uvicorn running on" in line:
            break

    

    yield


    server_proc.terminate()
    serve_proc.terminate()
    server_proc.wait()
    serve_proc.wait()
    ray.shutdown()

def test_something(setup_environment):
    # 这里可以进行测试
    pass