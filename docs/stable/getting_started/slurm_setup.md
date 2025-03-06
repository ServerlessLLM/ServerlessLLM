---
sidebar_position: 4
---

# SLURM-based cluster setup guide

This guide will help you get started with running ServerlessLLM on SLURM cluster, connecting them to the head node, and starting the `sllm-store` on the worker node. Additionally, this guide will also show how to quickly setup with Docker Compose. Please make sure you have installed the ServerlessLLM following the [installation guide](./installation.md) on all machines.

## Some Tips about Installation
- Both installation and build require an internet connection. please make sure the port 443 on the job node you want to install is accessible.
- If 'not enough disk space' is reported when `pip install` on the login node, you can submit it to a job node for execution
  ```shell
  #!/bin/bash
  #SBATCH --partition=Teach-Standard
  #SBATCH --job-name=ray-head
  #SBATCH --output=sllm_pip.out
  #SBATCH --error=sllm_pip.err
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=4
  #SBATCH --gpus-per-task=0

  # Identify which conda you are using, here is an example that conda is in /opt/conda
  source /opt/conda/bin/activate

  conda create -n sllm python=3.10 -y
  conda activate sllm
  pip install serverless-llm
  pip install serverless-llm-store

  conda deactivate sllm

  conda create -n sllm-worker python=3.10 -y
  conda activate sllm-worker
  pip install serverless-llm[worker]
  pip install serverless-llm-store
  ```
- Importantly, we recommend using [installing with pip](https://serverlessllm.github.io/docs/stable/getting_started/installation#installing-with-pip) if there is no CUDA driver on the node you want to install. If you want to [install from source](https://serverlessllm.github.io/docs/stable/getting_started/installation#installing-from-source), please make sure CUDA driver available on the node you want to install. Here are some commands to check it.
   ```shell
   module avail cuda # if you can see some CUDA options, it means CUDA is available, then load the cuda module
   module load cuda-12.x # load specific CUDA version
   # or
   nvidia-smi # if you can see the GPU information, it means CUDA is available
   # or
   which nvcc # if you can see the CUDA compiler's path, it means CUDA is available
   ```
   However, we **strongly recommend that you read the documentation for the HPC you are using** to find out how to check if the CUDA driver is available.

## Job Nodes Setup
Let's start a head on the main job node (`JobNode01`) and add the worker on other job node (`JobNode02`). The head and the worker should be on different job nodes to avoid resource contention. The `sllm-store` should be started on the job node that runs worker (`JobNode02`), for passing the model weights, and the `sllm-serve` should be started on the main job node (`JobNode01`), finally you can use `sllm-cli` to manage the models on the login node.

Note: `JobNode02` requires GPU, but `JobNode01` does not.
- **Head**: JobNode01
- **Worker**: JobNode02
- **sllm-store**: JobNode02
- **sllm-serve**: JobNode01
- **sllm-cli**: Login Node

## Find nodes with sufficient computing power
Consult the cluster documentation/administrator or run the following commands in the cluster to find a node with sufficient computing power (Compute Capacity > 7.0) ([Click here to check if your node has sufficient computing power](https://developer.nvidia.com/cuda-gpus#compute)).
```shell
sinfo -O partition,nodelist,gres
```
**Expected Output**
```shell
PARTITION           NODELIST            GRES
Partition1          JobNode[01,03]      gpu:gtx_1060:8
Partition2          JobNode[04-17]      gpu:a6000:2,gpu:gtx_
```

## Step 1: Start the Head Node
1. **Identify an idle node**

    Since the head node does not require a gpu, you can find a low-computing capacity node to deploy the head node.
    ```shell
    $ sinfo -p compute
    PARTITION AVAIL  NODES  STATE  TIMELIMIT  NODELIST
    compute    up       10  idle   infinite   JobNode[01-10]
    compute    up        5  alloc  infinite   JobNode[11-15]
    compute    up        2  down   infinite   JobNode[16-17]
    ```
2. **Activate the `sllm` environment and start the head node:**

    Here is the example script, named `start_head_node.sh`.
    ```shell
    #!/bin/bash
    #SBATCH --partition=your-partition    # Specify the partition
    #SBATCH --nodelist=JobNode01          # Specify an idle node
    #SBATCH --job-name=ray-head
    #SBATCH --output=sllm_head.out
    #SBATCH --error=sllm_head.err
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=12
    #SBATCH --gpus-per-task=0

    cd /path/to/ServerlessLLM

    source /opt/conda/bin/activate # make sure conda will be loaded correctly
    conda activate sllm

    ray start --head --port=6379 --num-cpus=12 --num-gpus=0 --resources='{"control_node": 1}' --block
    ```
   - Replace `your-partition`, `JobNode01` and `/path/to/ServerlessLLM`

3. **Submit the script**

    Use ```sbatch start_head_node.sh``` to submit the script to certain idle node.

4. **Expected output**

    In `sllm_head.out`, you will see the following output:

    ```shell
    Local node IP: <HEAD_NODE_IP>
    --------------------
    Ray runtime started.
    --------------------
    ```
   **Remember the IP address**, denoted ```<HEAD_NODE_IP>```, you will need it in following steps.

5. **Find an available port for serve**
  - Some HPCs have a firewall that blocks port 8343. You can use `nc -zv <HEAD_NODE_IP> 8343` to check if the port is accessible.
  - If it is not accessible, find an available port and replace `available_port` in the following script.
  - Here is an example script, named `find_port.sh`

   ```shell
   #!/bin/bash
   #SBATCH --partition=your-partition
   #SBATCH --nodelist=JobNode01
   #SBATCH --job-name=find_port
   #SBATCH --output=find_port.log
   #SBATCH --time=00:05:00
   #SBATCH --mem=1G

   echo "Finding available port on $(hostname)"

   python -c "
   import socket
   with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
       s.bind(('', 0))
       print(f'Available port: {s.getsockname()[1]}')
   "
   ```
   Use `sbatch find_port.sh` to submit the script to JobNode01, and in `find_port.log`, you will see the following output:
   ```
   Finding available port on JobNode01
   Available port: <avail_port>
   ```
   Remember this <avail_port>, you will use it in Step 4

## Step 2: Start the Worker Node & Store
We will start the worker node and store in the same script. Because the server loads the model weights onto the GPU and uses shared GPU memory to pass the pointer to the client. If you submit another script with ```#SBATCH --gpres=gpu:1```, it will be possibly set to use a different GPU, as specified by different ```CUDA_VISIBLE_DEVICES``` settings. Thus, they cannot pass the model weights.
1. **Activate the ```sllm-worker``` environment and start the worker node.**

   Here is the example script, named```start_worker_node.sh```.
   ```shell
   #!/bin/sh
   #SBATCH --partition=your_partition
   #SBATCH --nodelist=JobNode02           # Note JobNode02 should have sufficient compute capacity
   #SBATCH --gres=gpu:a6000:1             # Specify device on JobNode02
   #SBATCH --job-name=sllm-worker-store
   #SBATCH --output=sllm_worker.out
   #SBATCH --error=sllm_worker.err
   #SBATCH --gres=gpu:1                   # Request 1 GPU
   #SBATCH --cpus-per-task=4              # Request 4 CPU cores
   #SBATCH --mem=16G                      # Request 16GB of RAM

   cd /path/to/ServerlessLLM

   conda activate sllm-worker

   HEAD_NODE_IP=<HEAD_NODE_IP>

   export CUDA_HOME=/opt/cuda-12.5.0 # replace with your CUDA path
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

   ray start --address=$HEAD_NODE_IP:6379 --num-cpus=4 --num-gpus=1 \
   --resources='{"worker_node": 1, "worker_id_0": 1}' --block &

   sllm-store start &

   wait
   ```
   - Read the HPC's documentation to find out which partition you can use. Replace ```your_partition``` in the script with that partition name.
   - Replace ```/path/to/ServerlessLLM``` with the path to the ServerlessLLM installation directory.
   - Replace ```<HEAD_NODE_IP>``` with the IP address of the head node.
   - Replace ```/opt/cuda-12.5.0``` with the path to your CUDA path.

2. **Find the CUDA path**
   - Some slurm-based HPCs have a module system, you can use ```module avail cuda``` to find the CUDA module.
   - If it does not work, read the HPC's documentation carefully to find the CUDA path. For example, my doc said CUDA is in ```\opt```. Then you can use ```srun``` command to start an interactive session on the node, such as ```srun --pty -t 00:30:00 -p your_partition --gres=gpu:1 /bin/bash```. A pseudo-terminal will be started for you to find the path.
   - Find it and replace ```/opt/cuda-12.5.0``` with the path to your CUDA path.
3. **Submit the script on the other node**

    Use ```sbatch start_worker_node.sh``` to submit the script to certain idle node (here we assume it is ```JobNode02```). In addition, We recommend that you place the head and worker on different nodes so that the Serve can start smoothly later, rather than queuing up for resource allocation.
4. **Expected output**

   In ```sllm_worker.out```, you will see the following output:

   - The worker node expected output:
      ```shell
       Local node IP: xxx.xxx.xx.xx
       --------------------
       Ray runtime started.
       --------------------
      ```
   - The store expected output:
      ```shell
      I20241030 11:52:54.719007 1321560 checkpoint_store.cpp:41] Number of GPUs: 1
      I20241030 11:52:54.773468 1321560 checkpoint_store.cpp:43] I/O threads: 4, chunk size: 32MB
      I20241030 11:52:54.773548 1321560 checkpoint_store.cpp:45] Storage path: "./models/"
      I20241030 11:52:55.060559 1321560 checkpoint_store.cpp:71] GPU 0 UUID: 52b01995-4fa9-c8c3-a2f2-a1fda7e46cb2
      I20241030 11:52:55.060798 1321560 pinned_memory_pool.cpp:29] Creating PinnedMemoryPool with 128 buffers of 33554432 bytes
      I20241030 11:52:57.258795 1321560 checkpoint_store.cpp:83] Memory pool created with 4GB
      I20241030 11:52:57.262835 1321560 server.cpp:306] Server listening on 0.0.0.0:8073
      ```
## Step 3: Start the Serve on the Head Node
1. **Activate the ```sllm``` environment and start the serve.**

   Here is the example script, named```start_serve.sh```.
   ```shell
   #!/bin/sh
   #SBATCH --partition=your_partition
   #SBATCH --nodelist=JobNode01           # This node should be the same as head
   #SBATCH --output=serve.log

   cd /path/to/ServerlessLLM

   conda activate sllm

   sllm-serve start --host <HEAD_NODE_IP>
   # sllm-serve start --host <HEAD_NODE_IP> --port <avail_port> # if you have changed the port
   ```
   - Replace `your_partition` in the script as before.
   - Replace `/path/to/ServerlessLLM` as before.
   - Replace `<avail_port>` you have found in Step 1 (if port 8343 is not available).
2. **Submit the script on the head node**

    Use ```sbatch start_serve.sh``` to submit the script to the head node (```JobNode01```).

3. **Expected output**
   ```shell
   -- Connecting to existing Ray cluster at address: xxx.xxx.xx.xx:6379...
   -- Connected to Ray cluster.
   INFO:     Started server process [1339357]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://xxx.xxx.xx.xx:8343 (Press CTRL+C to quit)
   ```
## Step 4: Use sllm-cli to manage models
1. **You can do this step on login node, and set the ```LLM_SERVER_URL``` environment variable:**
   ```shell
   $ conda activate sllm
   (sllm)$ export LLM_SERVER_URL=http://<HEAD_NODE_IP>:8343/
   ```
   - Replace ```<HEAD_NODE_IP>``` with the actual IP address of the head node.
   - Replace ```8343``` with the actual port number (`<avail_port>` in Step1) if you have changed it.
2. **Deploy a Model Using ```sllm-cli```**
   ```shell
   (sllm)$ sllm-cli deploy --model facebook/opt-1.3b
   ```
## Step 5: Query the Model Using OpenAI API Client
   **You can use the following command to query the model:**
   ```shell
   curl http://<HEAD_NODE_IP>:8343/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
         "model": "facebook/opt-1.3b",
         "messages": [
               {"role": "system", "content": "You are a helpful assistant."},
               {"role": "user", "content": "What is your name?"}
         ]
      }'
   ```
   - Replace ```<HEAD_NODE_IP>``` with the actual IP address of the head node.
   - Replace ```8343``` with the actual port number (`<avail_port>` in Step 1) if you have changed it.
## Step 6: Stop Jobs
On the SLURM cluster, we usually use the ```scancel``` command to stop the job. Firstly, list all jobs you have submitted (replace ```your_username``` with your username):
```shell
$ squeue -u your_username
JOBID    PARTITION     NAME                USER       ST  TIME  NODES NODELIST(REASON)
  1234    compute   sllm-head         your_username  R   0:01      1    JobNode01
  1235    compute   sllm-worker-store your_username  R   0:01      1    JobNode02
  1236    compute   sllm-serve        your_username  R   0:01      1    JobNode01
```
Then, use ```scancel``` to stop the job (```1234```, ```1235``` and ```1236``` are JOBIDs):
```shell
$ scancel 1234 1235 1236
```
