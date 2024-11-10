# Install SLLM on Windows Machines

## 1 Windows Subsystem for Linux
To install the SLLM dependencies, Ubuntu 20.04 is required by the installation guidance states. Packages, like vllm, cannot be installed on Windows. Therefore, we need a sub-system to run Linux on Windows machines. Here, I recommend downloading the Ubuntu app from Microsoft store and the WSL connection will allow to you install everything needed for. If you have more familiar to start Linux environment, please feel free to go ahead and share your experience with us. Generally speaking, it is a developed and stable app except WSL connection doesn't support GUI, and sometimes it can be nasty to set variable path.

![alt text](Ubuntu-app.png)

Description:
Install a complete Ubuntu terminal environment in minutes with Windows Subsystem for Linux (WSL). Develop cross-platform applications, improve your data science or web development workflows and manage IT infrastructure without leaving Windows.

Key features:
  - Efficient command line utilities including bash, ssh, git, apt, npm, pip and many more
  - Manage Docker containers with improved performance and startup times
  - Leverage GPU acceleration for AI/ML workloads with NVIDIA CUDA
  - A consistent development to deployment workflow when using Ubuntu in the cloud
  - 5 years of security patching with Ubuntu Long Term Support (LTS) releases

Ubuntu currently provides the 24.04.1 LTS release. When new LTS versions are released, Ubuntu can be upgraded once the first point release is available. This can be done from the command line by using:

    sudo do-release-upgrade

Installation tips:
  - Search for "Turn Windows features on or off" in the Windows search bar and ensure that "Windows Subsystem for Linux" is turned on before restarting your machine.
  - To launch, use "ubuntu" on the command-line prompt or Windows Terminal, or click on the Ubuntu tile in the Start Menu.

For more information about Ubuntu WSL and how Canonical supports developers please visit:

https://ubuntu.com/wsl

A step-by-step guidance is available on YT https://www.youtube.com/watch?v=sUsTQTJFmjs given by ProgrammingKnowledge, it is really detailed (basically telling you which button to click).

If everything goes right, you should have this terminal. The command line starts with your pre-set username.

<img src="wsl-terminal.png" alt="Description" width="800"/>


## 2 Configure the WSL connection
Back to the installation of sllm, as the processing of deploying LLM requires sufficient GPU memory. We need to ensure the WSL connection has enough resources. Use `free -h` to check the memory status. It's strongly recommended to assgining 20GB+ Mem and 8GB+ Swap. If it didn't meet this requirement, we need to set it manually in a file `.wslconfig`, and it should locate in `C:\Users\your_username\`. If not, we need to create it ourselves. Within the file, a sample setting is given below

    [wsl2]               # this line is a must-have
    memory=28GB          # Limits VM memory
    swap=8GB             # Sets swap file size

Then, following the installation guidance using pip install, everything should be work. 

## 3 Potential issues to be encountered:

### 3.1 CUDA issues

If you see error message like 'error while loading shared libraries: libcudart.so.12: cannot open shared object file: No such file or directory', you need to double-check if CUDA v12 are correctly installed in the sllm and sllm-worker envs. Using 'nvcc --version' can check the installed version. The correct setting should be looking like this.

![alt text](cuda-version-check.png)

### 3.2 Folder Access

Sometime, WSL connection might forbid accessing created folders. For example, I met this error message `No such file or directory: './model/vllm'`. When I check my home folder, I realised the `model/` folder marked in red, which means the system has no access to it. So I just `rm -rf model/` and `mkdir model/`, then it all runs well.

### 3.3 GPU Number

As most machines have only one GPU, the num_gpu needs to be set as 1 while starting the ray server in sllm-worker. You can double-check the available number of GPUs by `nvidia-smi -L`

    conda activate sllm-worker
    ray start --address=0.0.0.0:6379 --num-cpus=4 --num-gpus=2 \
    --resources='{"worker_node": 1, "worker_id_0": 1}' --block


At this point, it should allow you to deploy models on the windows machine. Please don't hesitate to post any concerns encountered during your installation.