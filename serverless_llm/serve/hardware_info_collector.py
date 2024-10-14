import ray
import psutil
import GPUtil
import speedtest
import time
import os
import tempfile
import subprocess
import re
import platform
import pynvml
import json
from typing import Dict
from tqdm import tqdm

@ray.remote
class HardwareInfoCollector:
    """
    A Ray actor class responsible for collecting hardware information.
    """

    def __init__(self):
        """
        Initialize the HardwareInfoCollector actor.
        """
        self.collector = HardwareInfoCollectorMethods()

    def collect_all_info(self):
        """
        Collect all hardware information and return as a dictionary.
        """
        hardware_info = {}
        hardware_info["host_memory"] = self.collector.get_memory_info()
        hardware_info["host_bandwidth"] = self.collector.benchmark_memory_bandwidth()
        hardware_info["disk_size"] = self.collector.get_disk_info()
        write_bw, read_bw = self.collector.benchmark_disk_bandwidth()
        hardware_info["disk_write_bandwidth"] = write_bw
        hardware_info["disk_read_bandwidth"] = read_bw
        upload_bw, download_bw = self.collector.get_network_bandwidth()
        hardware_info["network_upload_bandwidth"] = upload_bw
        hardware_info["network_download_bandwidth"] = download_bw
        hardware_info["GPUs_info"] = self.collector.get_gpu_info()
        return hardware_info

class HardwareInfoCollectorMethods:
    """
    A class encapsulating methods to collect various hardware metrics.
    """

    def get_memory_info(self):
        """
        Retrieves total system memory.
        Returns:
            str: Total memory in GB (e.g., "32GB").
        """
        try:
            mem = psutil.virtual_memory()
            total_memory_gb = mem.total / (1024 ** 3)
            return f"{total_memory_gb:.2f}GB"
        except Exception as e:
            print(f"Failed to retrieve memory info: {e}")
            return "N/A"

    def benchmark_memory_bandwidth(self, num_iterations=5):
        """
        Estimates memory bandwidth by performing memory operations multiple times.
        Args:
            num_iterations (int): Number of iterations to run.
        Returns:
            str: Average memory bandwidth in GB/s (e.g., "24GB/s").
        """
        bandwidth_results = []
        try:
            size = 100 * 1024 * 1024  # 100 MB
            for _ in range(num_iterations):
                data = bytearray(os.urandom(size))
                start_time = time.time()
                # Simulate memory operations
                for _ in range(10):
                    data = bytearray(data)
                end_time = time.time()
                elapsed = end_time - start_time
                # Calculate bandwidth in MB/s
                bandwidth_mb_s = (size * 10) / elapsed / (1024 ** 2)
                # Convert to GB/s
                bandwidth_gb_s = bandwidth_mb_s / 1024
                bandwidth_results.append(bandwidth_gb_s)
            # Calculate average bandwidth
            average_bandwidth = sum(bandwidth_results) / len(bandwidth_results)
            return f"{average_bandwidth:.2f}GB/s"
        except Exception as e:
            print(f"Memory bandwidth benchmark failed: {e}")
            return "N/A"

    def get_disk_info(self):
        """
        Retrieves total size of the primary disk partition.
        Returns:
            str: Disk size in GB (e.g., "128GB").
        """
        try:
            partitions = psutil.disk_partitions()
            # Assume primary partition is mounted at '/'
            for partition in partitions:
                if partition.mountpoint == '/':
                    usage = psutil.disk_usage(partition.mountpoint)
                    total_size_gb = usage.total / (1024 ** 3)
                    return f"{total_size_gb:.2f}GB"
            # If '/' not found, return the first partition's size
            if partitions:
                usage = psutil.disk_usage(partitions[0].mountpoint)
                total_size_gb = usage.total / (1024 ** 3)
                return f"{total_size_gb:.2f}GB"
            return "N/A"
        except Exception as e:
            print(f"Failed to retrieve disk info: {e}")
            return "N/A"

    def benchmark_disk_bandwidth(self, num_iterations=5):
        """
        Measures disk read and write bandwidth by writing and reading a temporary file multiple times.
        Args:
            num_iterations (int): Number of iterations to run.
        Returns:
            tuple: Average write bandwidth (str) and average read bandwidth (str) in GB/s.
        """
        write_results = []
        read_results = []
        try:
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, "disk_bandwidth_test.tmp")
            size = 100 * 1024 * 1024  # 100 MB

            for _ in range(num_iterations):
                # Write Test
                start_time = time.time()
                with open(temp_file, 'wb') as f:
                    f.write(os.urandom(size))
                end_time = time.time()
                write_time = end_time - start_time
                write_bandwidth_mb_s = size / write_time / (1024 ** 2)
                write_bandwidth_gb_s = write_bandwidth_mb_s / 1024
                write_results.append(write_bandwidth_gb_s)

                # Read Test
                start_time = time.time()
                with open(temp_file, 'rb') as f:
                    _ = f.read()
                end_time = time.time()
                read_time = end_time - start_time
                read_bandwidth_mb_s = size / read_time / (1024 ** 2)
                read_bandwidth_gb_s = read_bandwidth_mb_s / 1024
                read_results.append(read_bandwidth_gb_s)

            # Clean up
            os.remove(temp_file)

            # Calculate averages
            average_write = sum(write_results) / len(write_results)
            average_read = sum(read_results) / len(read_results)
            return f"{average_write:.2f}GB/s", f"{average_read:.2f}GB/s"
        except Exception as e:
            print(f"Disk bandwidth benchmark failed: {e}")
            return "N/A", "N/A"

    def get_network_bandwidth(self, num_iterations=3):
        """
        Measures network upload and download bandwidth by performing speed tests multiple times.
        Args:
            num_iterations (int): Number of iterations to run.
        Returns:
            tuple: Average upload bandwidth (str) and average download bandwidth (str) in Gbps.
        """
        upload_results = []
        download_results = []
        try:
            for _ in range(num_iterations):
                st = speedtest.Speedtest()
                st.get_best_server()
                download_speed_mbps = st.download() / 1_000_000  # Convert to Mbps
                upload_speed_mbps = st.upload(pre_allocate=False) / 1_000_000  # Convert to Mbps
                # Convert to Gbps
                download_speed_gbps = download_speed_mbps / 1000
                upload_speed_gbps = upload_speed_mbps / 1000
                upload_results.append(upload_speed_gbps)
                download_results.append(download_speed_gbps)
                # Wait to avoid server overload
                time.sleep(5)
            # Calculate averages
            average_upload = sum(upload_results) / len(upload_results)
            average_download = sum(download_results) / len(download_results)
            return f"{average_upload:.2f}Gbps", f"{average_download:.2f}Gbps"
        except Exception as e:
            print(f"Network bandwidth test failed: {e}")
            return "N/A", "N/A"

    def get_gpu_info(self):
        """
        Retrieves information about each NVIDIA GPU.
        Returns:
            dict: Dictionary containing GPU information.
        """
        gpus_info = {}
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                gpus_info = "No GPUs found."
            else:
                for gpu in gpus:
                    gpu_id = gpu.id + 1  # To start GPU numbering from 1
                    gpus_info[gpu_id] = {
                        # "info": gpu.__dict__,
                        "Name": gpu.name,
                        "Load": f"{gpu.load * 100:.1f}%",
                        "Free_Memory": f"{gpu.memoryFree}MB",
                        "Used_Memory": f"{gpu.memoryUsed}MB",
                        "Total_Memory": f"{gpu.memoryTotal}MB"
                    }
        except Exception as e:
            print(f"Failed to retrieve GPU information: {e}")
            gpus_info = "N/A"
        return gpus_info
    
    
if __name__ == "__main__":
    ray.init()
    collector = HardwareInfoCollector.remote()
    hardware_info = ray.get(collector.collect_all_info.remote())
    print(json.dumps(hardware_info, indent=4))
    ray.shutdown()