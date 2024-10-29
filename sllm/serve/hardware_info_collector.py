import ray
import psutil
import GPUtil
import speedtest
import time
import os
import tempfile
import json
from typing import Dict, Set
from tqdm import tqdm
from sllm.serve.logger import init_logger


logger = init_logger(__name__)

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
        hardware_info["host_size"] = self.collector.get_memory_info()
        hardware_info["host_bandwidth"] = self.collector.benchmark_memory_bandwidth()
        hardware_info["disk_size"] = self.collector.get_disk_info()
        write_bw, read_bw = self.collector.benchmark_disk_bandwidth()
        hardware_info["disk_bandwidth"] = (write_bw + read_bw) / 2 # Average of write and read bandwidth
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
            str: Total memory in B
        """
        try:
            mem = psutil.virtual_memory()
            return mem.total
        except Exception as e:
            logger.error(f"Failed to retrieve memory info: {e}")
            return "N/A"

    def benchmark_memory_bandwidth(self, num_iterations=5):
        """
        Estimates memory bandwidth by performing memory operations multiple times.
        Args:
            num_iterations (int): Number of iterations to run.
        Returns:
            str: Average memory bandwidth in B/s
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
                # Calculate bandwidth in B/s
                bandwidth_b_s = size * 10 / elapsed
                bandwidth_results.append(bandwidth_b_s)
            # Calculate average bandwidth
            average_bandwidth = sum(bandwidth_results) / len(bandwidth_results)
            return average_bandwidth
        except Exception as e:
            logger.error(f"Memory bandwidth benchmark failed: {e}")
            return "N/A"

    def get_disk_info(self):
        """
        Retrieves total size of the primary disk partition.
        Returns:
            str: Disk size in B
        """
        try:
            partitions = psutil.disk_partitions()
            # Assume primary partition is mounted at '/'
            for partition in partitions:
                if partition.mountpoint == '/':
                    usage = psutil.disk_usage(partition.mountpoint)
                    return usage.total
            # If '/' not found, return the first partition's size
            if partitions:
                usage = psutil.disk_usage(partitions[0].mountpoint)
                return usage.total
            return "N/A"
        except Exception as e:
            logger.error(f"Failed to retrieve disk info: {e}")
            return "N/A"

    def benchmark_disk_bandwidth(self, num_iterations=5):
        """
        Measures disk read and write bandwidth by writing and reading a temporary file multiple times.
        Args:
            num_iterations (int): Number of iterations to run.
        Returns:
            tuple: Average write bandwidth (str) and average read bandwidth (str) in B/s
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
                write_bandwidth_b_s = size / write_time
                write_results.append(write_bandwidth_b_s)

                # Read Test
                start_time = time.time()
                with open(temp_file, 'rb') as f:
                    _ = f.read()
                end_time = time.time()
                read_time = end_time - start_time
                read_bandwidth_b_s = size / read_time
                read_results.append(read_bandwidth_b_s)

            # Clean up
            os.remove(temp_file)

            # Calculate averages
            average_write = sum(write_results) / len(write_results)
            average_read = sum(read_results) / len(read_results)
            return average_write, average_read
        except Exception as e:
            logger.error(f"Disk bandwidth benchmark failed: {e}")
            return "N/A", "N/A"

    def get_network_bandwidth(self, num_iterations=1):
        """
        Measures network upload and download bandwidth by performing speed tests multiple times.
        Args:
            num_iterations (int): Number of iterations to run.
        Returns:
            tuple: Average upload bandwidth (str) and average download bandwidth (str) in bps
        """
        upload_results = []
        download_results = []
        try:
            for _ in tqdm(range(num_iterations), desc="Testing network bandwidth"):
                st = speedtest.Speedtest()
                st.get_best_server()
                download_speed_mbps = st.download()
                upload_speed_mbps = st.upload(pre_allocate=False)

                # Convert to bps
                download_results.append(download_speed_mbps * (1024**2) / 8)
                upload_results.append(upload_speed_mbps * (1024**2) / 8)

                # Wait to avoid server overload
                time.sleep(5)
            # Calculate averages
            average_upload = sum(upload_results) / len(upload_results)
            average_download = sum(download_results) / len(download_results)
            return average_upload, average_download
        
        except Exception as e:
            logger.error(f"Network bandwidth test failed: {e}")
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
            logger.error(f"Failed to retrieve GPU information: {e}")
            gpus_info = "N/A"
        return gpus_info
    
    
def get_worker_nodes() -> Dict[str, dict]:
    """
    Retrieves a dictionary of worker nodes with their custom resource labels.
    
    Returns:
        Dict[str, dict]: A dictionary mapping custom resource labels to node information.
    """
    nodes = ray.nodes()
    worker_nodes = {}
    for node in nodes:
        if node['Alive'] and node['Resources'].get('CPU') and node['NodeManagerAddress'] != ray.get_runtime_context().get_node_id():
            # Extract custom resource labels (e.g., 'worker_id_1')
            custom_resources = [res for res in node['Resources'] if res.startswith('worker_id')]
            if custom_resources:
                resource_label = custom_resources[0]
                worker_nodes[resource_label] = node
    return worker_nodes

def store_results(benchmark_results: Dict[str, dict], filename: str = "benchmark_results.json"):
    """
    Stores the benchmark results in a JSON file.
    
    Args:
        benchmark_results (Dict[str, dict]): The benchmark results mapped by node IP.
        filename (str): The filename for the JSON output.
    """
    with open(filename, 'w') as f:
        json.dump(benchmark_results, f, indent=4)
    logger.info(f"Benchmark results stored in {filename}")

def main():
    processed_nodes: Set[str] = set()
    benchmark_results: Dict[str, dict] = {}
    
    logger.info("Starting Ray cluster monitoring for benchmark execution with custom resources...")
    
    try:
        while True:
            worker_nodes = get_worker_nodes()
            new_nodes = set(worker_nodes.keys()) - processed_nodes
            
            if new_nodes:
                logger.info(f"Detected new nodes with resources: {new_nodes}")
                collectors = []
                
                for resource_label in new_nodes:
                    # Dispatch the benchmark task to the specific node using its custom resource
                    collector = HardwareInfoCollector.options(resources={resource_label: 1}).remote()
                    collectors.append((resource_label, collector))
                
                # Collect results as they complete
                for resource_label, collector in collectors:
                    try:
                        result = ray.get(collector.collect_all_info.remote(), timeout=60)  # Adjust timeout as needed
                        benchmark_results[resource_label] = result
                        processed_nodes.add(resource_label)
                        logger.info(f"Stored benchmark result for node {resource_label}")
                    except ray.exceptions.GetTimeoutError:
                        logger.error(f"Benchmark task on resource {resource_label} timed out.")
            
            else:
                logger.info("No new nodes detected.")
                
            store_results(benchmark_results)
            # Wait before the next check
            time.sleep(10)  # Adjust the polling interval as needed
    
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user.")
    
    finally:
        # Store the benchmark results
        logger.info("Final Benchmark Results:")
        for resource_label, result in benchmark_results.items():
            logger.info(f"Node {resource_label}: {result}")
    
if __name__ == "__main__":
    ray.init()
    main()