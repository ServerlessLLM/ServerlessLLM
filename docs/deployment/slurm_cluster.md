---
sidebar_position: 3
---

# SLURM cluster

This guide will help you get started with running ServerlessLLM on SLURM cluster. It provides two deployment methods, based on `sbatch` and `srun`. If you are in development, we recommend using `srun`, as it is easier to debug than `sbatch`, and if you are in production mode, `sbatch` is recommended. Please make sure you have installed the ServerlessLLM following the [installation guide](./single_machine.md#installation) on all machines.

## Pre-requisites
Before you begin, make sure you have checked the following:
### Some Tips about Installation
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

### Command for Querying GPU Resource Information
Run the following commands in the cluster to check GPU resource information.
```shell
sinfo -O partition,nodelist,gres
```
**Expected Output**
```shell
PARTITION           NODELIST            GRES
Partition1          JobNode[01,03]      gpu:gtx_1060:8
Partition2          JobNode[04-17]      gpu:a6000:2,gpu:gtx_
```

### Identify an idle node
Use `sinfo -p <partition>` to identify some idle nodes

**Expected Output**
```shell
$ sinfo -p compute
PARTITION AVAIL  NODES  STATE  TIMELIMIT  NODELIST
compute    up       10  idle   infinite   JobNode[01-10]
compute    up        5  alloc  infinite   JobNode[11-15]
compute    up        2  down   infinite   JobNode[16-17]
```

### Job Nodes Setup
**`srun` Node Selection**

Only one JobNode is enough.

**`sbatch` Node Selection**
Let's start a Pylet head on the main job node (`JobNode01`) and add the Pylet worker on other job node (`JobNode02`). The head and the worker should be on different job nodes to avoid resource contention. The `sllm-store` should be started on the job node that runs Pylet worker (`JobNode02`), for passing the model weights, and the `sllm start` should be started on the main job node (`JobNode01`), finally you can use `sllm` to manage the models on the login node.


Note: `JobNode02` requires GPU, but `JobNode01` does not.
- **Pylet Head**: JobNode01
- **Pylet Worker**: JobNode02
- **sllm-store**: JobNode02
- **SLLM Head**: JobNode01
- **sllm**: Login Node

---
## SRUN
If you are in development, we recommend using `srun` to start ServerlessLLM, as it is easier to debug than `sbatch`
### Step 1: Use `srun` enter the JobNode
To start an interactive session on the specified compute node (JobNode), use:
```
srun --partition <your-partition> --nodelist <JobNode> --gres <DEVICE>:1 --pty bash
```
This command requests a session on the specified node and provides an interactive shell. `--gres <DEVICE>:1` specifies the GPU device you will use, for example: `--gres gpu:gtx_1060:1`

### Step 2: Install ServerlessLLM
Firstly, please make sure CUDA driver available on the node. Here are some commands to check it.
```shell
nvidia-smi

which nvcc
```
If `nvidia-smi` has listed GPU information, but `which nvcc` has no output. Then use the following commands to load `nvcc`. Here is an example that cuda is located at `/opt/cuda-12.2.0`
```shell
export PATH=/opt/cuda-12.2.0/bin:$PATH
export LD_LIBRARY_PATH=/opt/cuda-12.2.0/lib64:$LD_LIBRARY_PATH
```
Then, following the [installation guide](./single_machine.md#installation) to install ServerlessLLM.
### Step 3: Prepare multiple windows with `tmux`
Since srun provides a single interactive shell, you can use tmux to create multiple windows. Start a tmux session:
```shell
tmux
```
This creates a new tmux session

**Create multiple windows**
- Use `Ctrl+B` → `C` to start a new window
- Repeat the shortcut 4 more times to create a total of 5 windows.

**What if `Ctrl+B` does not work?**

If `Ctrl + B` is unresponsive, reset tmux key bindings:
```shell
tmux unbind C-b
tmux set-option -g prefix C-b
tmux bind C-b send-prefix
```

**Command to switch windows**

Once multiple windows are created, you can switch between them using:

`Ctrl + B` → `N` (Next window)
`Ctrl + B` → `P` (Previous window)
`Ctrl + B` → `W` (List all windows and select)
`Ctrl + B` → [Number] (Switch to a specific window, e.g., Ctrl + B → 1)

### Step 4: Run ServerlessLLM on the JobNode
First find ports that are already occupied. Then pick your favourite number from the remaining ports to replace the following placeholder `<PORT>`. For example: `8000`

It should also be said that certain slurm system is a bit slow, **so please be patient and wait for the system to output**.

In the first window, start the Pylet head:
```shell
source /opt/conda/bin/activate
conda activate sllm
pylet start --port=<PORT>
```
In the second window, start the Pylet worker:
```shell
source /opt/conda/bin/activate
conda activate sllm-worker
export CUDA_VISIBLE_DEVICES=0
pylet worker --head http://127.0.0.1:<PORT>
```
In the third window, start ServerlessLLM Store server:
```shell
source /opt/conda/bin/activate
conda activate sllm-worker
export CUDA_VISIBLE_DEVICES=0
sllm-store start
```
In the 4th window, start SLLM Head:
```shell
source /opt/conda/bin/activate
conda activate sllm
sllm start --pylet-endpoint http://127.0.0.1:<PORT>
```
Everything is set!


In the 5th window, let's deploy a model to the ServerlessLLM server. You can deploy a model by running the following command:
```shell
source /opt/conda/bin/activate
conda activate sllm
sllm deploy --model facebook/opt-1.3b --backend vllm
```
This will download the model checkpoint. After deploying, you can query the model by any OpenAI API client. For example, you can use the following Python code to query the model:
```shell
curl http://127.0.0.1:8343/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
        "model": "facebook/opt-1.3b",
        "backend": "vllm",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is your name?"}
        ]
    }'
```
Expected output:
```shell
{"id":"chatcmpl-9f812a40-6b96-4ef9-8584-0b8149892cb9","object":"chat.completion","created":1720021153,"model":"facebook/opt-1.3b","choices":[{"index":0,"message":{"role":"assistant","content":"system: You are a helpful assistant.\nuser: What is your name?\nsystem: I am a helpful assistant.\n"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":16,"completion_tokens":26,"total_tokens":42}}
```

### Step 5: Clean up
To delete a deployed model, use the following command:
```shell
sllm delete facebook/opt-1.3b --backend vllm
```
This will remove the specified model from the ServerlessLLM server.

In each window, use `Ctrl + c` to stop server and `exit` to exit current `tmux` session.

---
## SBATCH
### Step 1: Start the Pylet Head Node
Since the Pylet head node does not require a gpu, you can find a low-computing capacity node to deploy it.
1. **Activate the `sllm` environment and start the Pylet head node:**

    Here is the example script, named `start_pylet_head.sh`.
    ```shell
    #!/bin/bash
    #SBATCH --partition=your-partition    # Specify the partition
    #SBATCH --nodelist=JobNode01          # Specify an idle node
    #SBATCH --job-name=pylet-head
    #SBATCH --output=pylet_head.out
    #SBATCH --error=pylet_head.err
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=12
    #SBATCH --gpus-per-task=0

    cd /path/to/ServerlessLLM

    source /opt/conda/bin/activate # make sure conda will be loaded correctly
    conda activate sllm

    pylet start --port=8000
    ```
   - Replace `your-partition`, `JobNode01` and `/path/to/ServerlessLLM`

2. **Submit the script**

    Use ```sbatch start_pylet_head.sh``` to submit the script to certain idle node.

3. **Expected output**

    In `pylet_head.out`, you will see the following output:

    ```shell
    INFO:     Started server process [<PID>]
    INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
    ```
   **Remember the IP address of the node**, denoted ```<HEAD_NODE_IP>```, you will need it in following steps.

4. **Find an available port for serve**
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
   Remember this `<avail_port>`, you will use it in Step 4

### Step 2: Start the Pylet Worker & Store
We will start the Pylet worker and store in the same script. Because the server loads the model weights onto the GPU and uses shared GPU memory to pass the pointer to the client. If you submit another script with ```#SBATCH --gpres=gpu:1```, it will be possibly set to use a different GPU, as specified by different ```CUDA_VISIBLE_DEVICES``` settings. Thus, they cannot pass the model weights.
1. **Activate the ```sllm-worker``` environment and start the Pylet worker.**

   Here is the example script, named```start_pylet_worker.sh```.
   ```shell
   #!/bin/sh
   #SBATCH --partition=your_partition
   #SBATCH --nodelist=JobNode02
   #SBATCH --gres=gpu:a6000:1             # Specify device on JobNode02
   #SBATCH --job-name=pylet-worker-store
   #SBATCH --output=pylet_worker.out
   #SBATCH --error=pylet_worker.err
   #SBATCH --gres=gpu:1                   # Request 1 GPU
   #SBATCH --cpus-per-task=4              # Request 4 CPU cores
   #SBATCH --mem=16G                      # Request 16GB of RAM

   cd /path/to/ServerlessLLM

   conda activate sllm-worker

   HEAD_NODE_IP=<HEAD_NODE_IP>

   export CUDA_HOME=/opt/cuda-12.5.0 # replace with your CUDA path
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

   pylet worker --head http://$HEAD_NODE_IP:8000 &

   sllm-store start &

   wait
   ```
   - Read the HPC's documentation to find out which partition you can use. Replace ```your_partition``` in the script with that partition name.
   - Replace ```/path/to/ServerlessLLM``` with the path to the ServerlessLLM installation directory.
   - Replace ```<HEAD_NODE_IP>``` with the IP address of the Pylet head node.
   - Replace ```/opt/cuda-12.5.0``` with the path to your CUDA path.

2. **Find the CUDA path**
   - Some slurm-based HPCs have a module system, you can use ```module avail cuda``` to find the CUDA module.
   - If it does not work, read the HPC's documentation carefully to find the CUDA path. For example, my doc said CUDA is in ```\opt```. Then you can use ```srun``` command to start an interactive session on the node, such as ```srun --pty -t 00:30:00 -p your_partition --gres=gpu:1 /bin/bash```. A pseudo-terminal will be started for you to find the path.
   - Find it and replace ```/opt/cuda-12.5.0``` with the path to your CUDA path.
3. **Submit the script on the other node**

    Use ```sbatch start_pylet_worker.sh``` to submit the script to certain idle node (here we assume it is ```JobNode02```). In addition, We recommend that you place the Pylet head and worker on different nodes so that the SLLM Head can start smoothly later, rather than queuing up for resource allocation.
4. **Expected output**

   In ```pylet_worker.out```, you will see the following output:

   - The Pylet worker expected output:
      ```shell
      INFO:     Worker registered with head at http://<HEAD_NODE_IP>:8000
      INFO:     Worker started successfully
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
### Step 3: Start the SLLM Head on the Head Node
1. **Activate the ```sllm``` environment and start the SLLM head.**

   Here is the example script, named```start_sllm_head.sh```.
   ```shell
   #!/bin/sh
   #SBATCH --partition=your_partition
   #SBATCH --nodelist=JobNode01           # This node should be the same as Pylet head
   #SBATCH --output=sllm_head.log

   cd /path/to/ServerlessLLM

   conda activate sllm

   export PYLET_ENDPOINT=http://<HEAD_NODE_IP>:8000

   sllm start --host <HEAD_NODE_IP> --pylet-endpoint $PYLET_ENDPOINT
   # sllm start --host <HEAD_NODE_IP> --port <avail_port> --pylet-endpoint $PYLET_ENDPOINT # if you have changed the port
   ```
   - Replace `your_partition` in the script as before.
   - Replace `/path/to/ServerlessLLM` as before.
   - Replace `<HEAD_NODE_IP>` with the IP address of the Pylet head node.
   - Replace `<avail_port>` you have found in Step 1 (if port 8343 is not available).
2. **Submit the script on the head node**

    Use ```sbatch start_sllm_head.sh``` to submit the script to the head node (```JobNode01```).

3. **Expected output**
   ```shell
   INFO:     Started server process [1339357]
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://xxx.xxx.xx.xx:8343 (Press CTRL+C to quit)
   ```
### Step 4: Use sllm to manage models
1. **You can do this step on login node, and set the ```LLM_SERVER_URL``` and ```PYLET_ENDPOINT``` environment variables:**
   ```shell
   $ conda activate sllm
   (sllm)$ export LLM_SERVER_URL=http://<HEAD_NODE_IP>:8343
   (sllm)$ export PYLET_ENDPOINT=http://<HEAD_NODE_IP>:8000
   ```
   - Replace `<HEAD_NODE_IP>` with the actual IP address of the head node.
   - Replace ```8343``` with the actual port number (`<avail_port>` in Step1) if you have changed it.
2. **Deploy a Model Using ```sllm```**
   ```shell
   (sllm)$ sllm deploy --model facebook/opt-1.3b --backend vllm
   ```
### Step 5: Query the Model Using OpenAI API Client
   **You can use the following command to query the model:**
   ```shell
   curl $LLM_SERVER_URL/v1/chat/completions \
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
### Step 6: Stop Jobs
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
