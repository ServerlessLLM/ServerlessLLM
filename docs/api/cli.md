---
sidebar_position: 2
---

# CLI API

## ServerlessLLM CLI Documentation

### Overview
`sllm-cli` is a command-line interface (CLI) tool designed for managing and interacting with ServerlessLLM models. This document provides an overview of the available commands and their usage.

### Installation

```bash
# Create a new environment
conda create -n sllm python=3.10 -y
conda activate sllm

# Install ServerlessLLM
pip install serverless-llm
```

### Getting Started

Before using the `sllm-cli` commands, you need to start the ServerlessLLM cluster. Follow the guides below to set up your cluster:

- [Single Machine Deployment](../stable/gettting_started.md)
- [Single Machine Deployment (From Scratch)](../stable/deployment/single_machine.md)
- [Multi-Machine Deployment](../stable/deployment/multi_machine.md)
- [SLURM Cluster Deployment](../stable/deployment/slurm_cluster.md)

After setting up the ServerlessLLM cluster, you can use the commands listed below to manage and interact with your models.

### Example Workflow

1. **Deploy a Model**
    > Deploy a model using the model name, which must be a HuggingFace pretrained model name. i.e. "facebook/opt-1.3b" instead of "opt-1.3b".
    ```bash
    sllm-cli deploy --model facebook/opt-1.3b
    ```

2. **Generate Output**
    ```bash
    echo '{
      "model": "facebook/opt-1.3b",
      "messages": [
        {
          "role": "user",
          "content": "Please introduce yourself."
        }
      ],
      "temperature": 0.7,
      "max_tokens": 50
    }' > input.json
    sllm-cli generate input.json
    ```

3. **Delete a Model**
    ```bash
    sllm-cli delete facebook/opt-1.3b
    ```

### sllm-cli deploy
Deploy a model using a configuration file or model name, with options to overwrite default configurations. The configuration file requires minimal specifications, as sensible defaults are provided for advanced configuration options.

For more details on the advanced configuration options and their default values, please refer to the [Example Configuration File](#example-configuration-file-configjson) section.

##### Usage
```bash
sllm-cli deploy [OPTIONS]
```

##### Options
- `--model <model_name>`
  - Model name to deploy with default configuration. The model name must be a Hugging Face pretrained model name. You can find the list of available models [here](https://huggingface.co/models).

- `--config <config_path>`
  - Path to the JSON configuration file. The configuration file can be incomplete, and missing sections will be filled in by the default configuration.

- `--backend <backend_name>`
  - Overwrite the backend in the default configuration.

- `--num-gpus <number>`
  - Overwrite the number of GPUs in the default configuration.

- `--target <number>`
  - Overwrite the target concurrency in the default configuration.

- `--min-instances <number>`
  - Overwrite the minimum instances in the default configuration.

- `--max-instances <number>`
  - Overwrite the maximum instances in the default configuration.

##### Examples
Deploy using a model name with default configuration:
```bash
sllm-cli deploy --model facebook/opt-1.3b
```

Deploy using a configuration file:
```bash
sllm-cli deploy --config /path/to/config.json
```

Deploy using a model name and overwrite the backend:
```bash
sllm-cli deploy --model facebook/opt-1.3b --backend transformers
```

Deploy using a model name and overwrite multiple configurations:
```bash
sllm-cli deploy --model facebook/opt-1.3b --num-gpus 2 --target 5 --min-instances 1 --max-instances 5
```

##### Example Configuration File (`config.json`)
This file can be incomplete, and missing sections will be filled in by the default configuration:
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
        "hf_model_class": "AutoModelForCausalLM"
    }
}
```

Below is a description of all the fields in config.json.

| Field | Description |
| ----- | ----------- |
| model | This should be a HuggingFace model name, used to identify model instance. |
| backend | Inference engine, support `transformers` and `vllm` now. |
| num_gpus | Number of GPUs used to deploy a model instance. |
| auto_scaling_config | Config about auto scaling. |
| auto_scaling_config.metric | Metric used to decide whether to scale up or down. |
| auto_scaling_config.target | Target value of the metric. |
| auto_scaling_config.min_instances | The minimum value for model instances. |
| auto_scaling_config.max_instances | The maximum value for model instances. |
| auto_scaling_config.keep_alive | How long a model instance lasts after inference ends. For example, if keep_alive is set to 30, it will wait 30 seconds after the inference ends to see if there is another request. |
| backend_config | Config about inference backend. |
| backend_config.pretrained_model_name_or_path | The path to load the model, this can be a HuggingFace model name or a local path. |
| backend_config.device_map | Device map config used to load the model, `auto` is suitable for most scenarios. |
| backend_config.torch_dtype | Torch dtype of the model. |
| backend_config.hf_model_class | HuggingFace model class. |

### sllm-cli delete
Delete deployed models by name.

##### Usage
```bash
sllm-cli delete [MODELS]
```

##### Arguments
- `MODELS`
  - Space-separated list of model names to delete.

##### Example
```bash
sllm-cli delete facebook/opt-1.3b facebook/opt-2.7b meta/llama2
```

### sllm-cli generate
Generate outputs using the deployed model.

##### Usage
```bash
sllm-cli generate [OPTIONS] <input_path>
```

##### Options
- `-t`, `--threads <num_threads>`
  - Number of parallel generation processes. Default is 1.

##### Arguments
- `input_path`
  - Path to the JSON input file.

##### Example
```bash
sllm-cli generate --threads 4 /path/to/request.json
```

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
      "Hi, How are you?"
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
Fine-tuning the deployed model.

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
    "model": "bigscience/bloomz-560m",
    "ft_backend": "peft",
    "dataset_config": {
        "dataset_source": "hf_hub",
        "hf_dataset_name": "fka/awesome-chatgpt-prompts",
        "tokenization_field": "prompt",
        "split": "train[:10%]",
        "data_files": "",
        "extension_type": ""
    },
    "lora_config": {
        "r": 4,
        "lora_alpha": 1,
        "target_modules": ["query_key_value"],
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
| dataset_config                       | Config about the fine-tuning dataset                                                                                                                                                       |
| dataset_config.dataset_source        | dataset is from `hf_hub` (huggingface_hub) or `local` file                                                                                                                                 |
| dataset_config.hf_dataset_name       | dataset name on huggingface_hub                                                                                                                                                            |
| dataset_config.tokenization_field    | the field to tokenize                                                                                                                                                                      |
| dataset_config.split                 | Partitioning of the dataset (`train`, `validation` and `test`), You can also split the selected dataset, e.g. take only the top 10% of the training data: train[:10%]                                                                                                                             |
| dataset_config.data_files            | data files will be loaded from local                                                                                                                                                       |
| dataset_config.extension_type        | extension type of data files (`csv`, `json`, `parquet`, `arrow`)                                                                                                                           |
| lora_config                          | Config about lora                                                                                                                                                                          |
| lora_config.r                        | `r` defines how many parameters will be trained.                                                                                                                                           |
| lora_config.lora_alpha               | A multiplier controlling the overall strength of connections within a neural network, typically set at 1                                                                                   |
| lora_config.target_modules           | a list of the target_modules available on the [Hugging Face Documentation](https://github.com/huggingface/peft/blob/39ef2546d5d9b8f5f8a7016ec10657887a867041/src/peft/utils/other.py#L220) |
| lora_config.lora_dropout             | used to avoid overfitting                                                                                                                                                                  |
| lora_config.bias                     | use `none` or `lora_only`                                                                                                                                                                  |
| lora_config.task_type                | Indicates the task the model is begin trained for                                                                                                                                          |
| training_config                      | Config about training parameters                                                                                                                                                           |
| training_config.auto_find_batch_size | Find a correct batch size that fits the size of Data.                                                                                                                                      |
| training_config.num_train_epochs     | Total number of training rounds                                                                                                                                                            |
| training_config.learning_rate        | learning rate                                                                                                                                                                              |
| training_config.optim                | select an optimiser                                                                                                                                                                        |
| training_config.use_cpu              | if use cpu to train                                                                                                                                                                        |

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