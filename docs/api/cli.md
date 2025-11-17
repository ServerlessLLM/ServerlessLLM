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
  delete   Delete deployed models by name.
  deploy   Deploy a model using a config file or model name.
  start    Start the head node of the SLLM cluster.
  status   Show all deployed models.
```

---

## Example Workflow

### 1. Start the Cluster

```bash
sllm start
```
**Example output:**
```text
[ℹ] Starting services using docker-compose.yml...
[+] Running 3/3
 ✔ sllm_api         Started
 ✔ sllm_worker_0    Started
 ✔ sllm_worker_1    Started
```

---

### 2. Deploy a Model

```bash
sllm deploy --model facebook/opt-1.3b
```
**Example output:**
```text
[✓] Successfully deployed model: facebook/opt-1.3b with 1 GPU(s).
```

---

### 3. Check Deployment Status

```bash
sllm status
```
**Example output:**
```text
[✓] Deployed Models:
- facebook/opt-1.3b
```

---

### 4. Delete a Model

```bash
sllm delete facebook/opt-1.3b
```
**Example output:**
```text
[✓] Deleted model: facebook/opt-1.3b
```

---

## Command Reference

### sllm start

Start the head node of the SLLM cluster. This command initializes Docker services (or other configured backends) that manage the API and worker nodes.

**Usage:**
```bash
sllm start
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
  Overwrite the backend in the configuration.
- `--num-gpus <number>`
  Number of GPUs to allocate.
- `--target <number>`
  Target concurrency.
- `--min-instances <number>`
  Minimum number of instances.
- `--max-instances <number>`
  Maximum number of instances.

**Examples:**
```bash
sllm deploy --model facebook/opt-1.3b
sllm deploy --config /path/to/config.json
sllm deploy --model facebook/opt-1.3b --backend transformers
sllm deploy --model facebook/opt-1.3b --backend sglang
sllm deploy --model facebook/opt-1.3b --num-gpus 2 --target 5 --min-instances 1 --max-instances 5
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

##### Example SGLang Configuration (`config.json`)

```json
{
    "model": "facebook/opt-1.3b",
    "backend": "sglang",
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
        "torch_dtype": "float16",
        "tp_size": 1,
        "trust_remote_code": false,
        "disable_cuda_graph": false,
        "log_level": "error"
    }
}
```

**Note:** When using the SGLang backend with ServerlessLLM's checkpoint store, the `load_format` will automatically be set to `"serverless_llm"` if not specified. Make sure to save your model using the `save_sglang_model.py` script first.

Below is a description of all the fields in config.json.

| Field | Description |
| ----- | ----------- |
| model | HuggingFace model name, used to identify model instance. |
| backend | Inference engine, supports `transformers`, `vllm`, and `sglang`. |
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

Delete deployed models by name.

**Usage:**
```bash
sllm delete [MODELS...]
```
**Arguments:**
- `MODELS...`
  One or more space-separated model names to delete.

**Example:**
```bash
sllm delete facebook/opt-1.3b facebook/opt-2.7b meta/llama2
```
**Example output:**
```text
[✓] Deleted model: facebook/opt-1.3b
[✓] Deleted model: facebook/opt-2.7b
[✓] Deleted model: meta/llama2
```

---

### sllm status

Check the current status of all deployed models. This command displays the list of models currently running in the ServerlessLLM cluster, including their state, GPU usage, and endpoint.

**Usage:**
```bash
sllm status
```
**Example output:**
```text
[✓] Deployed Models:
- facebook/opt-1.3b    Running
- meta/llama2          Running

---

## Notes

- All commands should be run as `sllm ...` after installation.
- For advanced configuration, refer to the [Example Configuration File](#example-configuration-file-configjson) section.
- For more details, see the official documentation and guides linked above.



#### Example
```bash
sllm status
```

#### Example
```bash
sllm-cli status
```
