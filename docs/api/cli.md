---
sidebar_position: 2
---

# CLI API

## ServerlessLLM CLI Documentation

## Overview
`sllm` is the official command-line interface for interacting with ServerlessLLM. It is implemented using the [Click](https://click.palletsprojects.com/) framework to provide a flexible and extensible interface for managing model start, deploy, delete, and system status.

### Installation

```bash
# Create a new environment
conda create -n sllm python=3.10 -y
conda activate sllm

# Install ServerlessLLM
pip install serverless-llm
```


The CLI organizes commands into clearly scoped modules. This document outlines each available command along with its usage and configuration options.

---

## Getting Started

Before using the `sllm` commands, you need to start the ServerlessLLM cluster. Follow the guides below to set up your cluster:

- [Single Machine Deployment](../getting_started.md)
- [Single Machine Deployment (From Scratch)](../deployment/single_machine.md)
- [Multi-Machine Deployment](../deployment/multi_machine.md)
- [SLURM Cluster Deployment](../deployment/slurm_cluster.md)

After setting up the ServerlessLLM cluster, you can use the commands listed below to manage and interact with your models.

---

## Available Commands

To see all available CLI commands, run:

```bash
sllm --help
```

**Example output:**
```text
Usage: sllm [OPTIONS] COMMAND [ARGS]...

  Unified CLI for ServerlessLLM.

Options:
  --help  Show this message and exit.

Commands:
  delete  Delete deployments, or remove only the LoRA adapters.
  deploy  Deploy a model using a config file or model name.
  logs    View instance logs.
  start   Start the SLLM head node (control plane).
  status  Show cluster status.
```

---

## Example Workflow

### 1. Start the Cluster

```bash
sllm start --pylet-endpoint http://localhost:8000
```
**Example output:**
```text
INFO:     ServerlessLLM v1-beta Head Node
INFO:     Pylet endpoint: http://localhost:8000
INFO:     Database path: /var/lib/sllm/state.db
INFO:     Storage path: /models
INFO:     Starting head node (v1-beta)...
INFO:     Head node started on 0.0.0.0:8343
```

---

### 2. Deploy a Model

```bash
sllm deploy --model facebook/opt-1.3b --backend vllm
```
**Example output:**
```text
Deployment created: facebook/opt-1.3b:vllm
  View status: sllm status facebook/opt-1.3b:vllm
```

---

### 3. Check Deployment Status

```bash
sllm status
```
**Example output:**
```text
DEPLOYMENT              STATUS  REPLICAS
facebook/opt-1.3b:vllm  READY   1/1
```

---

### 4. Delete a Model

```bash
sllm delete facebook/opt-1.3b --backend vllm
```
**Example output:**
```text
[SUCCESS] Deployment 'facebook/opt-1.3b:vllm' deletion initiated.
```

---

## Command Reference

### sllm start

Start the SLLM head node (control plane). In v1-beta, this initializes the API Gateway, Router, Autoscaler, and Reconciler components that work with Pylet for cluster management.

**Usage:**
```bash
sllm start [OPTIONS]
```

**Options:**
- `--host <ip_address>`
  Host IP for the API Gateway (default: `0.0.0.0`).
- `--port <port>`
  Port for the API Gateway (default: `8343`).
- `--pylet-endpoint <url>`
  Pylet head endpoint URL (default: `http://localhost:8000` or `PYLET_ENDPOINT` env var).
- `--database-path <path>`
  SQLite database path (default: `/var/lib/sllm/state.db` or `SLLM_DATABASE_PATH` env var).
- `--storage-path <path>`
  Model storage path (default: `/models` or `STORAGE_PATH` env var).

**Examples:**
```bash
sllm start
sllm start --port 8080
sllm start --pylet-endpoint http://pylet-head:8000 --storage-path /data/models
```

---

### sllm deploy

Deploy a model using a configuration file or model name, with options to overwrite default configurations.

**Usage:**
```bash
sllm deploy [OPTIONS]
```

**Options:**
- `--model <model_name>`
  Model name to deploy (must be a HuggingFace pretrained model name).
- `--config <config_path>`
  Path to the JSON configuration file.
- `--backend <backend_name>`
  Backend framework (e.g., `vllm`, `transformers`, `sglang`).
- `--num-gpus <number>`
  Number of GPUs to allocate.
- `--target <number>`
  Target number of requests per second.
- `--min-instances <number>`
  Minimum number of instances.
- `--max-instances <number>`
  Maximum number of instances.
- `--enable-lora`
  Enable LoRA support for the model.
- `--lora-adapters <adapters>`
  List of LoRA adapters in `name=path` format (e.g., `"adapter1=/path/to/adapter1 adapter2=/path/to/adapter2"`).
- `--precision <precision>`
  Model precision for quantization (e.g., `int8`, `fp4`, `nf4`).

**Examples:**
```bash
sllm deploy --model facebook/opt-1.3b --backend vllm
sllm deploy --config /path/to/config.json
sllm deploy --model facebook/opt-1.3b --backend transformers
sllm deploy --model facebook/opt-1.3b --backend vllm --num-gpus 2 --target 5 --min-instances 1 --max-instances 5
sllm deploy --model facebook/opt-1.3b --backend vllm --enable-lora --lora-adapters "adapter1=/path/to/adapter1"
```

#### Example Configuration File (`config.json`)

```json
{
    "model": "facebook/opt-1.3b",
    "backend": "transformers",
    "num_gpus": 1,
    "auto_scaling_config": {
        "metric": "concurrency",
        "target": 1,
        "min_instances": 0,
        "max_instances": 10,
        "keep_alive": 0
    },
    "backend_config": {
        "pretrained_model_name_or_path": "facebook/opt-1.3b",
        "device_map": "auto",
        "torch_dtype": "float16",
        "hf_model_class": "AutoModelForCausalLM",
        "enable_lora": true,
        "lora_adapters": {
            "demo_lora1": "crumb/FLAN-OPT-1.3b-LoRA",
            "demo_lora2": "GrantC/alpaca-opt-1.3b-lora"
        }
    }
}
```

##### Example Quantization Configuration (`config.json`)
`quantization_config` can be obtained from any configuration used in `transformers` via the `.to_json_file(filename)` function:

```python
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
quantization_config.to_json_file("quantization_config.json")

```
Then copy it into `config.json`:

```json
{
    "model": "",
    "backend": "transformers",
    "num_gpus": 1,
    "auto_scaling_config": {
        "metric": "concurrency",
        "target": 1,
        "min_instances": 0,
        "max_instances": 10,
        "keep_alive": 0
    },
    "backend_config": {
        "pretrained_model_name_or_path": "",
        "device_map": "auto",
        "torch_dtype": "float16",
        "hf_model_class": "AutoModelForCausalLM",
        "quantization_config": {
            "_load_in_4bit": false,
            "_load_in_8bit": true,
            "bnb_4bit_compute_dtype": "float32",
            "bnb_4bit_quant_storage": "uint8",
            "bnb_4bit_quant_type": "fp4",
            "bnb_4bit_use_double_quant": false,
            "llm_int8_enable_fp32_cpu_offload": false,
            "llm_int8_has_fp16_weight": false,
            "llm_int8_skip_modules": null,
            "llm_int8_threshold": 6.0,
            "load_in_4bit": false,
            "load_in_8bit": true,
            "quant_method": "bitsandbytes"
        }
    }
}
```

Below is a description of all the fields in config.json.

| Field | Description |
| ----- | ----------- |
| model | HuggingFace model name, used to identify model instance. |
| backend | Inference engine, supports `transformers` and `vllm`. |
| num_gpus | Number of GPUs used to deploy a model instance. |
| auto_scaling_config | Config about auto scaling. |
| auto_scaling_config.metric | Metric used to decide whether to scale up or down. |
| auto_scaling_config.target | Target value of the metric. |
| auto_scaling_config.min_instances | Minimum number of model instances. |
| auto_scaling_config.max_instances | Maximum number of model instances. |
| auto_scaling_config.keep_alive | How long a model instance stays alive after inference ends. |
| backend_config | Config about inference backend. |
| backend_config.pretrained_model_name_or_path | HuggingFace model name or local path. |
| backend_config.device_map | Device map config used to load the model. |
| backend_config.torch_dtype | Torch dtype of the model. |
| backend_config.hf_model_class | HuggingFace model class. |
| backend_config.enable_lora | Set to true to enable loading LoRA adapters during inference. |
| backend_config.lora_adapters| A dictionary of LoRA adapters in the format `{name: path}`, where each path is a local or Hugging Face-hosted LoRA adapter directory. |
| backend_config.quantization_config| A dictionary specifying the desired `BitsAndBytesConfig`. Can be obtained by saving a `BitsAndBytesConfig` to JSON via `BitsAndBytesConfig.to_json_file(filename). Defaults to None.|

---

### sllm delete

Delete deployments, or remove only the LoRA adapters from a deployment.

**Usage:**
```bash
sllm delete [MODELS...] --backend <backend_name> [OPTIONS]
```

**Arguments:**
- `MODELS...`
  One or more space-separated model names to delete.

**Options:**
- `--backend <backend_name>` (required)
  Backend framework (e.g., `vllm`, `sglang`). Required to identify the deployment.
- `--lora-adapters <adapters>`
  LoRA adapters to delete (instead of deleting the entire deployment).

**Examples:**
```bash
sllm delete facebook/opt-1.3b --backend vllm
sllm delete facebook/opt-1.3b facebook/opt-2.7b --backend vllm
sllm delete facebook/opt-1.3b --backend vllm --lora-adapters "adapter1 adapter2"
```

**Example output:**
```text
[SUCCESS] Deployment 'facebook/opt-1.3b:vllm' deletion initiated.
[SUCCESS] Deployment 'facebook/opt-2.7b:vllm' deletion initiated.
```

---

### sllm status

Show cluster status. Without arguments, shows all deployments. With a deployment ID, shows detailed information for that deployment including instances. With `--nodes`, shows cluster nodes.

**Usage:**
```bash
sllm status [DEPLOYMENT_ID] [OPTIONS]
```

**Arguments:**
- `DEPLOYMENT_ID` (optional)
  Deployment ID to show detailed info for (e.g., `facebook/opt-1.3b:vllm`).

**Options:**
- `--nodes`
  Show cluster nodes instead of deployments.

**Examples:**
```bash
sllm status
sllm status facebook/opt-1.3b:vllm
sllm status --nodes
```

**Example output (all deployments):**
```text
DEPLOYMENT              STATUS  REPLICAS
facebook/opt-1.3b:vllm  READY   1/1
meta/llama2:vllm        READY   2/2
```

**Example output (single deployment):**
```text
Deployment: facebook/opt-1.3b:vllm
Status:     READY
Replicas:   1/1

Instances:
  ID                    NODE      ENDPOINT                  STATUS
  opt-1.3b-abc123       worker-0  http://10.0.0.2:8000      RUNNING
```

**Example output (nodes):**
```text
NODE      STATUS  GPUS
worker-0  READY   2/4
worker-1  READY   4/4
```

---

### sllm logs

View logs for a specific instance. Use the instance ID from `sllm status <deployment_id>` output.

**Usage:**
```bash
sllm logs INSTANCE_ID [OPTIONS]
```

**Arguments:**
- `INSTANCE_ID` (required)
  The instance ID to view logs for.

**Options:**
- `-f`, `--follow`
  Follow log output (similar to `tail -f`).

**Examples:**
```bash
sllm logs opt-1.3b-abc123
sllm logs opt-1.3b-abc123 -f
```

---

## Notes

- All commands should be run as `sllm ...` after installation.
- For advanced configuration, refer to the [Example Configuration File](#example-configuration-file-configjson) section.
- For more details, see the official documentation and guides linked above.
