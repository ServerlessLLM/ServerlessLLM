# ServerlessLLM Docker Compose Quickstart

To quickly set up a local ServerlessLLM cluster using Docker Compose, follow these steps:
```bash
docker compose up -d --build
```

Note: Make sure you have Docker installed on your system and NVIDIA GPUs available. For detailed instructions, refer to the [Docker Quickstart Guide](https://serverlessllm.github.io/docs/stable/getting_started/docker_quickstart).

## Customizing Configuration Options

ServerlessLLM now supports flexible runtime configurations for both the head and worker nodes through customizable command-line arguments. You can adjust these settings in the `docker-compose.yml` file to optimize resource allocation and adapt the deployment to your specific needs.

### Example: Adjusting Memory Pool Size

To use a memory pool size of 16GB, modify the `command` entry for each `sllm_worker_#` service in `docker-compose.yml` as follows:

```yaml
command: ["-mem_pool_size", "16", "-registration_required", "true"]
```

This command line option will set a memory pool size of 16GB for each worker node.