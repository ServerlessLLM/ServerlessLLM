---
sidebar_position: 2
---
# PEFT LoRA Serving

This example illustrates the process of deploying and serving a base large language model enhanced with LoRA (Low-Rank Adaptation) adapters in a ServerlessLLM cluster. It demonstrates how to start the cluster, deploy a base model with multiple LoRA adapters, perform inference using different adapters, and update or remove the adapters dynamically.

## Pre-requisites

To run this example, we will use Docker Compose to set up a ServerlessLLM cluster. Before proceeding, please ensure you have read the [Quickstart Guide](../getting_started.md).

We will use the following example base model & LoRA adapter
- Base model: `facebook/opt-125m`
- LoRA adapter: `peft-internal-testing/opt-125m-dummy-lora`

## Usage

Start a local Docker-based ray cluster using Docker Compose.

### Step 1: Configuration

Set the Model Directory. Create a directory on your host machine where models will be stored and set the `MODEL_FOLDER` environment variable to point to this directory:

```bash
export MODEL_FOLDER=/path/to/your/models
```

Replace `/path/to/your/models` with the actual path where you want to store the models.

### Step 2: Start the Services

Start the ServerlessLLM services using Docker Compose:

```bash
docker compose up -d
```

This command will start the Ray head node and two worker nodes defined in the `docker-compose.yml` file.

:::tip
Use the following command to monitor the logs of the head node:

```bash
docker logs -f sllm_head
```
:::

### Step 3: Deploy Models with LoRA Adapters
1. Configuration
```bash
conda activate sllm
export LLM_SERVER_URL=http://127.0.0.1:8343
```
2. Deploy models with specified lora adapters.
```bash
sllm-cli deploy --model facebook/opt-125m --backend transformers --enable-lora --lora-adapters demo_lora1=peft-internal-testing/opt-125m-dummy-lora demo_lora2=monsterapi/opt125M_alpaca
```
3. Verify the deployment.
```bash
curl $LLM_SERVER_URL/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "facebook/opt-125m",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ],
        "lora_adapter_name": "demo_lora1"
    }'
```
If no lora adapters specified, the system will use the base model to do inference
```bash
curl $LLM_SERVER_URL/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "facebook/opt-125m",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ]
    }'
```
### Step 4: Update LoRA Adapters
If you wish to switch to a different set of LoRA adapters, you can still use `sllm-cli deploy` command with updated adapter configurations. ServerlessLLM will automatically reload the new adapters without restarting the backend.
```bash
sllm-cli deploy --model facebook/opt-125m --backend transformers --enable-lora --lora-adapters demo-lora1=edbeeching/opt-125m-lora demo-lora2=Hagatiana/opt-125m-lora
```

### Step 5: Clean Up

Delete the lora adapters by running the following command (this command will only delete lora adapters, the base model won't be deleted):
```bash
sllm-cli delete facebook/opt-125m --lora-adapters demo-lora1 demo-lora2
```
If you need to stop and remove the containers, you can use the following commands:
```bash
docker compose down
```