# ServerlessLLM Docker Compose Quickstart

To quickly set up a local ServerlessLLM cluster using Docker Compose, follow these steps:
```bash
export MODEL_FOLDER=/path/to/models
docker compose up -d
```

Note: Make sure you have Docker installed on your system and NVIDIA GPUs available. For detailed instructions, refer to the [Docker Quickstart Guide](hhttps://serverlessllm.github.io/docs/stable/getting_started).

## Customizing Configuration Options

ServerlessLLM now supports flexible runtime configurations for both the head and worker nodes through customizable command-line arguments. You can adjust these settings in the `docker-compose.yml` file to optimize resource allocation and adapt the deployment to your specific needs.

### Examples:

#### 1. Adjusting Memory Pool Size

To use a memory pool size of 16GB, modify the `command` entry for each `sllm_worker_#` service in `docker-compose.yml` as follows:

```yaml
command: ["-mem-pool-size", "16", "-registration-required", "true"]
```

This command line option will set a memory pool size of 16GB for each worker node.

#### 2. Specifying GPU Devices

To specify the GPU devices to be used by the worker nodes, modify the `device_ids` entry for each `sllm_worker_#` service in `docker-compose.yml` as follows:

```yaml
device_ids: ["0", "1"]
```

This configuration will assign GPUs 0 and 1 to the worker nodes.