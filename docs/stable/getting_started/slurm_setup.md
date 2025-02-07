---
sidebar_position: 4
---

# SLURM-based cluster setup guide

This guide will help you get started with running ServerlessLLM on SLURM cluster, connecting them to the head node, and starting the `sllm-store` on the worker node. Additionally, this guide will also show how to quickly setup with Docker Compose. Please make sure you have installed the ServerlessLLM following the [installation guide](./installation.md) on all machines.

## Job Nodes Setup
Let's start a head on the main job node (```JobNode01```) and add the worker on other job node (```JobNode02```). The head and the worker should be on different job nodes to avoid resource contention. The ```sllm-store``` should be started on the job node that runs worker (```JobNode02```), for passing the model weights, and the ```sllm-serve``` should be started on the main job node (```JobNode01```), finally you can use ```sllm-cli``` to manage the models on the login node.

Note: ```JobNode02``` requires GPU, but ```JobNode01``` does not.
- **Head**: JobNode01
- **Worker**: JobNode02
- **sllm-store**: JobNode02
- **sllm-serve**: JobNode01
- **sllm-cli**: Login Node

### Step 1: Start the Head Node

### Step 2: Start the Worker Node & Store

### Step 3: Start the Serve on the Head Node

### Step 4: Use sllm-cli to manage models

### Step 5: Query the Model Using OpenAI API Client

### Step 6: Stop Jobs

