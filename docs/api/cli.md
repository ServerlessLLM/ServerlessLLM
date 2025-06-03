<<<<<<< HEAD:docs/stable/cli/cli_api.md
# ServerlessLLM CLI Documentation

## Overview
=======
---
sidebar_position: 2
---

# CLI API

## ServerlessLLM CLI Documentation

### Overview
`sllm-cli` is a command-line interface (CLI) tool designed to manage and interact with ServerlessLLM models. This document provides an overview of the available commands and their usage.

### Installation

```bash
# Create a new environment
conda create -n sllm python=3.10 -y
conda activate sllm

# Install ServerlessLLM
pip install serverless-llm
```
>>>>>>> upstream/main:docs/api/cli.md

`sllm` is the official command-line interface for interacting with ServerlessLLM. It is implemented using the [Click](https://click.palletsprojects.com/) framework to provide a flexible and extensible interface for managing model start, deploy, delete, and system status.

The CLI organizes commands into clearly scoped modules. This document outlines each available command along with its usage and configuration options.

---

## Getting Started

Before using the `sllm` commands, you need to start the ServerlessLLM cluster. Follow the guides below to set up your cluster:

- [Single Machine Deployment](../stable/getting_started.md)
- [Single Machine Deployment (From Scratch)](../stable/deployment/single_machine.md)
- [Multi-Machine Deployment](../stable/deployment/multi_machine.md)
- [SLURM Cluster Deployment](../stable/deployment/slurm_cluster.md)

After setting up the ServerlessLLM cluster, you can use the commands listed below to manage and interact with your models.

---

<<<<<<< HEAD:docs/stable/cli/cli_api.md
## Available Commands
=======
1. **Deploy a Model**
    > Deploy a model using the model name, which must be a HuggingFace pretrained model name. i.e. `facebook/opt-1.3b` instead of `opt-1.3b`.
    ```bash
    sllm-cli deploy --model facebook/opt-1.3b
    ```
>>>>>>> upstream/main:docs/api/cli.md

To see all available CLI commands, run:

<<<<<<< HEAD:docs/stable/cli/cli_api.md
=======
3. **Delete a Model**
    ```bash
    sllm-cli delete facebook/opt-1.3b
    ```

### sllm-cli deploy
Deploy a model using a configuration file or model name, with options to overwrite default configurations. The configuration file requires minimal specifications, as sensible defaults are provided for advanced configuration options.

This command also supports [PEFT LoRA (Low-Rank Adaptation)](https://huggingface.co/docs/peft/main/en/index), allowing you to deploy adapters on top of a base model, either via CLI flags or directly in the configuration file.

For more details on the advanced configuration options and their default values, please refer to the [Example Configuration File](#example-configuration-file-configjson) section.

##### Usage
>>>>>>> upstream/main:docs/api/cli.md
```bash
sllm --help
```

**Example output:**
```text
Usage: sllm [OPTIONS] COMMAND [ARGS]...

  Unified CLI for ServerlessLLM.

Options:
  --help  Show this message and exit.

<<<<<<< HEAD:docs/stable/cli/cli_api.md
Commands:
  delete   Delete deployed models by name.
  deploy   Deploy a model using a config file or model name.
  start    Start the head node of the SLLM cluster.
  status   Show all deployed models.
=======
- `--num-gpus <number>`
  - Overwrite the number of GPUs in the default configuration.

- `--target <number>`
  - Overwrite the target concurrency in the default configuration.

- `--min-instances <number>`
  - Overwrite the minimum instances in the default configuration.

- `--max-instances <number>`
  - Overwrite the maximum instances in the default configuration.

- `--enable-lora`
  - Enable LoRA adapter support for the transformers backend. Overwrite `enable_lora` in the default configuration.

- `--lora-adapters`
  - Add one or more LoRA adapters in the format `<name>=<path>`. Overwrite any existing `lora_adapters` in the default configuration.

##### Examples
Deploy using a model name with default configuration:
```bash
sllm-cli deploy --model facebook/opt-1.3b
>>>>>>> upstream/main:docs/api/cli.md
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

<<<<<<< HEAD:docs/stable/cli/cli_api.md
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
sllm deploy --model facebook/opt-1.3b --num-gpus 2 --target 5 --min-instances 1 --max-instances 5
```

#### Example Configuration File (`config.json`)

=======
Deploy a base model with multiple LoRA adapters:
```bash
sllm-cli deploy --model facebook/opt-1.3b --backend transformers --enable-lora --lora-adapters demo_lora1=crumb/FLAN-OPT-1.3b-LoRA demo_lora2=GrantC/alpaca-opt-1.3b-lora
```

##### Example Configuration File (`config.json`)
This file can be incomplete, and missing sections will be filled in by the default configuration:
>>>>>>> upstream/main:docs/api/cli.md
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

**Field Descriptions:**

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

<<<<<<< HEAD:docs/stable/cli/cli_api.md
---

### sllm delete

Delete deployed models by name.
=======
### sllm-cli delete
Delete deployed models by name, or delete specific LoRA adapters associated with a base model.

This command supports:
  - Removing deployed models
  - Removing specific LoRA adapters while preserving the base model
>>>>>>> upstream/main:docs/api/cli.md

**Usage:**
```bash
<<<<<<< HEAD:docs/stable/cli/cli_api.md
sllm delete [MODELS...]
=======
sllm-cli delete [MODELS] [OPTIONS]
>>>>>>> upstream/main:docs/api/cli.md
```
**Arguments:**
- `MODELS...`  
  One or more space-separated model names to delete.

<<<<<<< HEAD:docs/stable/cli/cli_api.md
**Example:**
=======
##### Arguments
- `MODELS`
  - Space-separated list of model names to delete.

##### Options
- `--lora-adapters <adapter_names>`
  - Space-separated list of LoRA adapter names to delete from the given model. If provided, the base model will not be deleted — only the specified adapters will be removed.

##### Example
Delete multiple base models (and all their adapters):
>>>>>>> upstream/main:docs/api/cli.md
```bash
sllm delete facebook/opt-1.3b facebook/opt-2.7b meta/llama2
```
**Example output:**
```text
[✓] Deleted model: facebook/opt-1.3b
[✓] Deleted model: facebook/opt-2.7b
[✓] Deleted model: meta/llama2
```
Delete specific LoRA adapters from a base model, keeping the base model:
```bash
sllm-cli delete facebook/opt-1.3b --lora-adapters demo_lora1 demo_lora2
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

<<<<<<< HEAD:docs/stable/cli/cli_api.md
=======
##### Example Request File (`request.json`)
```json
{
    "model": "facebook/opt-1.3b",
    "messages": [
        {
            "role": "user",
            "content": "Please introduce yourself."
        }
    ],
    "temperature": 0.3,
    "max_tokens": 50
}
```

### sllm-cli encode (embedding)
Get the embedding using the deployed model.

##### Usage
```bash
sllm-cli encode [OPTIONS] <input_path>
```

##### Options
- `-t`, `--threads <num_threads>`
  - Number of parallel encoding processes. Default is 1.

##### Arguments
- `input_path`
  - Path to the JSON input file.

##### Example
```bash
sllm-cli encode --threads 4 /path/to/request.json
```

##### Example Request File (`request.json`)
```json
{
    "model": "intfloat/e5-mistral-7b-instruct",
    "task_instruct": "Given a question, retrieve passages that answer the question",
    "query": [
      "Hi, how are you?"
    ]
}
```

### sllm-cli replay
Replay requests based on workload and dataset.

##### Usage
```bash
sllm-cli replay [OPTIONS]
```

##### Options
- `--workload <workload_path>`
  - Path to the JSON workload file.

- `--dataset <dataset_path>`
  - Path to the JSON dataset file.

- `--output <output_path>`
  - Path to the output JSON file for latency results. Default is `latency_results.json`.

##### Example
```bash
sllm-cli replay --workload /path/to/workload.json --dataset /path/to/dataset.json --output /path/to/output.json
```

#### sllm-cli update
Update a deployed model using a configuration file or model name.

##### Usage
```bash
sllm-cli update [OPTIONS]
```

##### Options
- `--model <model_name>`
  - Model name to update with default configuration.

- `--config <config_path>`
  - Path to the JSON configuration file.

##### Example
```bash
sllm-cli update --model facebook/opt-1.3b
sllm-cli update --config /path/to/config.json
```

### sllm-cli fine-tuning
Fine-tune the deployed model.

##### Usage
```bash
sllm-cli fine-tuning [OPTIONS]
```

##### Options
- `--base-model <model_name>`
  - Base model name to be fine-tuned
- `--config <config_path>`
  - Path to the JSON configuration file.

##### Example
```bash
sllm-cli fine-tuning --base-model <model_name>
sllm-cli fine-tuning --base-model <model_name> --config <path_to_ft_config_file>
```

##### Example Configuration File (`ft_config.json`)
```json
{
    "model": "facebook/opt-125m",
    "ft_backend": "peft",
    "dataset_config": {
        "dataset_source": "hf_hub",
        "hf_dataset_name": "fka/awesome-chatgpt-prompts",
        "tokenization_field": "prompt",
        "split": "train",
        "data_files": "",
        "extension_type": ""
    },
    "lora_config": {
        "r": 4,
        "lora_alpha": 1,
        "lora_dropout": 0.05,
        "bias": "lora_only",
        "task_type": "CAUSAL_LM"
    },
    "training_config": {
        "auto_find_batch_size": true,
        "num_train_epochs": 2,
        "learning_rate": 0.0001,
        "use_cpu": false
    }
}
```

Below is a description of all the fields in ft_config.json.

| Field                                | Description                                                                                                                                                                                |
|--------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model                                | This should be a deployed model name, used to identify the backend instance.                                                                                                                     |
| ft_backend                           | fine-tuning engine, only support `peft` now.                                                                                                                                               |
| dataset_config                       | Configuration for the fine-tuning dataset                                                                                                                                                       |
| dataset_config.dataset_source        | dataset is from `hf_hub` (huggingface_hub) or `local` file                                                                                                                                 |
| dataset_config.hf_dataset_name       | dataset name on huggingface_hub                                                                                                                                                            |
| dataset_config.tokenization_field    | the field to tokenize                                                                                                                                                                      |
| dataset_config.split                 | Partitioning of the dataset (`train`, `validation` and `test`), You can also split the selected dataset, e.g. take only the top 10% of the training data: train[:10%]                                                                                                                             |
| dataset_config.data_files            | data files will be loaded from local                                                                                                                                                       |
| dataset_config.extension_type        | extension type of data files (`csv`, `json`, `parquet`, `arrow`)                                                                                                                           |
| lora_config                          | Configuration for LoRA fine-tuning                                                                                                                                                                          |
| lora_config.r                        | `r` defines how many parameters will be trained.                                                                                                                                           |
| lora_config.lora_alpha               | A multiplier controlling the overall strength of connections within a neural network, typically set at 1                                                                                   |
| lora_config.target_modules           | a list of the target_modules available on the [Hugging Face Documentation][1] |
| lora_config.lora_dropout             | used to avoid overfitting                                                                                                                                                                  |
| lora_config.bias                     | use `none` or `lora_only`                                                                                                                                                                  |
| lora_config.task_type                | Indicates the task the model is begin trained for                                                                                                                                          |
| training_config                      | Configuration for training parameters                                                                                                                                                           |
| training_config.auto_find_batch_size | Find a correct batch size that fits the size of Data.                                                                                                                                      |
| training_config.num_train_epochs     | Total number of training rounds                                                                                                                                                            |
| training_config.learning_rate        | learning rate                                                                                                                                                                              |
| training_config.optim                | select an optimiser                                                                                                                                                                        |
| training_config.use_cpu              | whether to use CPU for training                                                                                                                                                                        |

[1]: https://github.com/huggingface/peft/blob/39ef2546d5d9b8f5f8a7016ec10657887a867041/src/peft/utils/other.py#L220

### sllm-cli status
Check the information of deployed models

#### Usage
```bash
sllm-cli status
```

#### Example
```bash
sllm-cli status
```
>>>>>>> upstream/main:docs/api/cli.md
