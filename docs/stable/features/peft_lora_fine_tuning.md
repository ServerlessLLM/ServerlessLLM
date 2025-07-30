---
sidebar_position: 3
---
# PEFT LoRA Fine-tuning

This feature introduces a dedicated fine-tuning backend (`ft_backend`) for handling LoRA (Low-Rank Adaptation) fine-tuning jobs in ServerlessLLM. This implementation provides isolated fine-tuning instances with specialized resource management and lifecycle control.

## Architecture

### Workflow

1. **Job Submission**: Fine-tuning jobs are submitted via REST API
2. **Resource Allocation**: Dedicated fine-tuning instances are created with isolated resources
3. **Training Execution**: LoRA fine-tuning is performed using PEFT library
4. **Status Tracking**: Job status is continuously updated and can be queried
5. **Cleanup**: Instances are automatically cleaned up after completion

## Prerequisites

Before using the fine-tuning feature, ensure you have:

1. **Base Model**: A base model must be saved using the transformers backend
2. **Docker Setup**: ServerlessLLM cluster running via Docker Compose
3. **Storage**: Adequate storage space for fine-tuned adapters

## Usage

### Step 1: Save Base Model

First, save a base model using the transformers backend:

```bash
sllm-store save --model facebook/opt-125m --backend transformers
```

### Step 2: Submit Fine-tuning Job

Submit a fine-tuning job using the REST API:

```bash
curl -X POST $LLM_SERVER_URL/fine-tuning \
  -H "Content-Type: application/json" \
  -d @/path/to/fine_tuning_config.json
```

#### Fine-tuning Configuration

Create a configuration file (`default_ft_config.json`) with the following structure:

```json
{
    "model": "facebook/opt-125m",
    "ft_backend": "peft_lora",
    "num_gpus": 1,
    "num_cpus": 1,
    "timeout": 3600,
    "backend_config": {
        "output_dir": "facebook/adapters/opt-125m_adapter_test",
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
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        },
        "training_config": {
            "auto_find_batch_size": true,
            "save_strategy": "no",
            "num_train_epochs": 2,
            "learning_rate": 0.0001,
            "use_cpu": false
        }
    }
}
```

#### Configuration Parameters

**Job Configuration:**
- `model`: Base model name
- `ft_backend`: Fine-tuning backend type (currently supports "peft_lora")
- `num_cpus`: Number of CPU cores required
- `num_gpus`: Number of GPUs required
- `timeout`: Maximum execution time in seconds
- `priority`: Job priority (higher values = higher priority)

**Backend Configuration:**
- `pretrained_model_name_or_path`: HuggingFace model identifier
- `device_map`: Device mapping strategy ("auto", "cpu", "cuda")
- `torch_dtype`: PyTorch data type ("float16", "float32", "bfloat16")
- `hf_model_class`: HuggingFace model class

**Dataset Configuration:**
- `dataset_source`: Source type ("hf_hub" or "local")
- `hf_dataset_name`: HuggingFace dataset name (for hf_hub)
- `data_files`: Local file paths (for local)
- `extension_type`: File extension type (for local)
- `tokenization_field`: Field name for tokenization
- `split`: Dataset split to use

**LoRA Configuration:**
- `r`: LoRA rank
- `lora_alpha`: LoRA alpha parameter
- `target_modules`: Target modules for LoRA adaptation
- `lora_dropout`: Dropout rate
- `bias`: Bias handling strategy
- `task_type`: Task type for PEFT

**Training Configuration:**
- `num_train_epochs`: Number of training epochs
- `per_device_train_batch_size`: Batch size per device
- `gradient_accumulation_steps`: Gradient accumulation steps
- `learning_rate`: Learning rate
- `warmup_steps`: Number of warmup steps
- `logging_steps`: Logging frequency
- `save_steps`: Model saving frequency
- `eval_steps`: Evaluation frequency

### Step 3: Expected Response

Upon successful job submission, you'll receive a response with the job ID:

```json
{
  "job_id": "job-123"
}
```

### Step 4: Monitor Job Status

Check the status of your fine-tuning job:

```bash
curl -X GET "$LLM_SERVER_URL/get-job-status?job_id=job-123"
```

#### Status Response

```json
{
  "fine-tuning job 65bb4144-87cc-47ff-ac10-11e926a41dc2 status": {
    "config": {
      "model": "facebook/opt-125m",
      "ft_backend": "peft_lora",
      "num_gpus": 1,
      "num_cpus": 1,
      "timeout": 3600,
      "backend_config": {
        "output_dir": "facebook/adapters/opt-125m_adapter_test",
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
          "lora_alpha": 32,
          "lora_dropout": 0.05,
          "bias": "none",
          "task_type": "CAUSAL_LM"
        },
        "training_config": {
          "auto_find_batch_size": true,
          "save_strategy": "no",
          "num_train_epochs": 2,
          "learning_rate": 0.0001,
          "use_cpu": false
        }
      }
    },
    "status": "completed",
    "created_time": "2025-07-29T18:30:03.050847",
    "updated_time": "2025-07-29T18:30:03.050852",
    "priority": 0
  }
}
```

**Possible Status Values:**
- `pending`: Job is waiting for resources
- `running`: Job is currently executing
- `completed`: Job completed successfully
- `failed`: Job failed with an error
- `cancelled`: Job was cancelled by user

### Step 5: Cancel Job (Optional)

If needed, you can cancel a running job:

```bash
curl -X POST "$LLM_SERVER_URL/cancel-job" \
  -H "Content-Type: application/json" \
  -d '{"job_id": "job-123"}'
```

## Job Management

### Resource Allocation

Fine-tuning jobs are allocated resources based on the specified requirements:

- **CPU**: Number of CPU cores specified in `num_cpus`
- **GPU**: Number of GPUs specified in `num_gpus`
- **Memory**: Automatically managed based on model size and batch size

### Priority System

Jobs are processed based on priority and creation time:

1. **Higher Priority**: Jobs with higher priority values are processed first
2. **FIFO**: Jobs with the same priority are processed in order of creation
3. **Resource Availability**: Jobs wait until sufficient resources are available

### Timeout Handling

Jobs have configurable timeout limits:

- **Default Timeout**: 3600 seconds (1 hour)
- **Configurable**: Set via `timeout` parameter in job configuration
- **Automatic Cleanup**: Jobs are automatically marked as failed if they exceed the timeout

## Output and Storage

### LoRA Adapter Storage

Fine-tuned LoRA adapters are automatically saved to the `output_dir` path you config in the `default_ft_config.json`, like:

```
{STORAGE_PATH}/transformers/facebook/adapters/opt-125m_adapter_test
```

### Adapter Contents

The saved adapter includes:

- **LoRA Weights**: Fine-tuned LoRA parameters
- **Configuration**: LoRA configuration file
- **Metadata**: Training metadata and statistics

## Integration with Serving

### Using Fine-tuned Adapters

After successful fine-tuning, the LoRA adapter can be used for inference:

```bash
# Deploy model with fine-tuned adapter
sllm deploy --model facebook/opt-125m --backend transformers --enable-lora --lora-adapters "my_adapter=ft_facebook/opt-125m_adapter"

# Use the adapter for inference
curl $LLM_SERVER_URL/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "facebook/opt-125m",
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "lora_adapter_name": "my_adapter"
}'
```

For more details about PEFT LoRA Serving, please see the [documentation](./peft_lora_serving.md)
## Troubleshooting

### Common Issues

1. **Job Stuck in Pending**: Check resource availability and job priority
2. **Dataset Loading Failures**: Verify dataset configuration and accessibility
3. **Training Failures**: Check GPU memory and batch size settings
4. **Timeout Errors**: Increase timeout or optimize training configuration

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/fine-tuning` | POST | Submit a fine-tuning job |
| `/get-job-status` | GET | Get job status |
| `/cancel-job` | POST | Cancel a running job |

### Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 404 | Job not found |
| 500 | Internal Server Error |

## Examples

### Complete Fine-tuning Workflow

```bash
# 1. Save base model
sllm-store save --model facebook/opt-125m --backend transformers

# 2. Start the ServerlessLLM cluster with docker compose
cd examples/docker
docker compose up -d --build

# 3. Submit fine-tuning job
cd .. && cd ..
curl -X POST $LLM_SERVER_URL/fine-tuning \
  -H "Content-Type: application/json" \
  -d @sllm/default_ft_config.json

# 4. Monitor job status
curl -X GET "$LLM_SERVER_URL/get-job-status?job_id=job-123"

# 5. Deploy base model with fine-tuned adapter
sllm deploy --model facebook/opt-125m --backend transformers --enable-lora --lora-adapters "my_adapter=ft_facebook/opt-125m_adapter"

# 5. Use for inference
curl $LLM_SERVER_URL/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "facebook/opt-125m",
    "messages": [{"role": "user", "content": "Hello"}],
    "lora_adapter_name": "my_adapter"
}'
```
