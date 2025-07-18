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
import os
import tempfile
import time

import GPUtil
import psutil

from sllm.serve.logger import init_logger

logger = init_logger(__name__)


def get_dynamic_metrics() -> dict:
    """
    Collects lightweight, real-time hardware metrics that change frequently.
    This function is designed to be fast and run in every heartbeat.
    """
    metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "gpu_info": _get_live_gpu_info(),
    }
    return metrics


def _get_live_gpu_info() -> dict:
    """
    Retrieves real-time information about each NVIDIA GPU (load, memory usage).
    Returns raw numbers for easier processing by the head node.
    """
    gpus_info = {}
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return {"status": "No GPUs found."}

        for gpu in gpus:
            gpus_info[gpu.id] = {
                "load": gpu.load * 100,
                "memory_free": gpu.memoryFree,
                "memory_used": gpu.memoryUsed,
            }
    except Exception as e:
        logger.error(
            f"Failed to retrieve live GPU information: {e}", exc_info=True
        )
        gpus_info = {"status": "error", "message": str(e)}
    return gpus_info


def benchmark_static_hardware(test_network: bool = False) -> dict:
    """
    Performs all slow, one-time hardware benchmarks.
    This should only be run once when the worker process starts.
    """
    logger.info("Running one-time static hardware benchmarks...")

    write_bw, read_bw = _benchmark_disk_bandwidth()

    # TODO: add actual profiling logic
    static_info = {
        "pcie_bandwidth": 25000000000,
        "disk_total_space": _get_disk_size(),
        "disk_write_bandwidth": write_bw,
        "disk_read_bandwidth": read_bw,
        "static_gpu_info": _get_static_gpu_info(),
    }

    if test_network:
        logger.warning(
            "Running network bandwidth test. This can take a minute..."
        )
    logger.info("Static hardware benchmarks complete.")
    return static_info


def _get_static_gpu_info() -> dict:
    """Retrieves static information about each NVIDIA GPU (name, total memory)."""
    gpus_info = {}
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return {"status": "No GPUs found."}

        for gpu in gpus:
            gpus_info[gpu.id] = {
                "name": gpu.name,
                "total_memory": gpu.memoryTotal,
            }
    except Exception as e:
        logger.error(
            f"Failed to retrieve static GPU information: {e}", exc_info=True
        )
        gpus_info = {"status": "error", "message": str(e)}
    return gpus_info


def _get_disk_size() -> int:
    """Retrieves total size of the primary disk partition in bytes."""
    try:
        usage = psutil.disk_usage("/")
        return usage.total
    except Exception as e:
        logger.error(f"Failed to retrieve disk info: {e}")
        return 0


def _benchmark_disk_bandwidth(
    file_size_mb: int = 100, num_iterations: int = 3
) -> tuple:
    """Measures disk read and write bandwidth."""
    write_results = []
    read_results = []
    try:
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, "sllm_disk_benchmark.tmp")
        size_bytes = file_size_mb * 1024 * 1024

        logger.info(f"Benchmarking disk R/W with a {file_size_mb}MB file...")

        for _ in range(num_iterations):
            start_time = time.time()
            with open(temp_file, "wb") as f:
                f.write(os.urandom(size_bytes))
            write_time = time.time() - start_time
            write_results.append(size_bytes / write_time)

            start_time = time.time()
            with open(temp_file, "rb") as f:
                _ = f.read()
            read_time = time.time() - start_time
            read_results.append(size_bytes / read_time)

        os.remove(temp_file)

        avg_write = sum(write_results) / len(write_results)
        avg_read = sum(read_results) / len(read_results)
        logger.info(
            f"Disk benchmark results: Write={avg_write/1e6:.2f} MB/s, Read={avg_read/1e6:.2f} MB/s"
        )
        return avg_write, avg_read
    except Exception as e:
        logger.error(f"Disk bandwidth benchmark failed: {e}", exc_info=True)
        return 0.0, 0.0


def _get_network_bandwidth(speedtest_module) -> tuple:
    """Measures network bandwidth using speedtest-cli."""
    try:
        st = speedtest_module.Speedtest()
        st.get_best_server()
        download_bps = st.download()
        upload_bps = st.upload()
        return upload_bps, download_bps
    except Exception as e:
        logger.error(f"Network bandwidth test failed: {e}", exc_info=True)
        return 0.0, 0.0
