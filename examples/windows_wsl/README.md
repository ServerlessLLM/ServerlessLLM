# Running ServerlessLLM on Windows via WSL

ServerlessLLM supports running on Windows through the Windows Subsystem for Linux (WSL). This guide provides step-by-step instructions to set up the environment and run ServerlessLLM on Windows from scratch. If you already have WSL set up, skip to [Installing ServerlessLLM](#3-installing-serverlessllm).

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installing Windows Subsystem for Linux](#1-installing-windows-subsystem-for-linux)
- [Configuring WSL for ServerlessLLM](#2-configuring-wsl-for-serverlessllm)
   - [Adjusting Memory Allocation](#21-adjusting-memory-allocation)
   - [Installing CUDA for GPU Acceleration](#22-installing-cuda-for-gpu-acceleration)
   - [Installing Conda](#23-installing-conda)
- [Installing ServerlessLLM](#3-installing-serverlessllm)
- [Potential Issues and Troubleshooting](#4-potential-issues-and-troubleshooting)
   - [Folder Access Issues](#41-folder-access-issues)

## Prerequisites

- **Windows 10 or Windows 11** with WSL 2 installed.
- **Administrative privileges** for installing and configuring WSL.
- **NVIDIA GPU** with up-to-date drivers (for GPU acceleration).
- **Basic command-line knowledge**.

---

## 1. Installing Windows Subsystem for Linux

1. Open PowerShell as an administrator and run:

```bash
# Run this command in PowerShell as an administrator
wsl --install
```

This command installs WSL 2 and the latest version of Ubuntu. Restart your computer when prompted.

2. After restarting, open **Ubuntu** from the Start menu and follow the on-screen setup instructions.

3. If the above command doesn't work, follow the [official Microsoft tutorial](https://learn.microsoft.com/en-us/windows/wsl/install)
or this [video guide](https://www.youtube.com/watch?v=sUsTQTJFmjs) for installing WSL.


## 2. Configuring WSL for ServerlessLLM

### 2.1 Adjusting Memory Allocation

1. Open a WSL terminal and check the current memory allocation:

```bash
# Run this command in the WSL terminal
free -h
```

By default, WSL uses half of the system's available memory.

2. To adjust memory settings, create or edit the `.wslconfig` file located in `C:\Users\your_username\.wslconfig`.

Add the following:

```ini
[wsl2]             # this line is a must-have
memory=16GB         # Allocate 16 GB of memory
swap=8GB           # Set swap file size
```

3. Restart WSL to apply the changes:
```bash
# Run this command in PowerShell as an administrator
wsl --shutdown
wsl
```

### 2.2 Installing CUDA for GPU Acceleration

1. Verify CUDA Installation:

Run the following in the WSL terminal:
```bash
# Run this command in the WSL terminal
nvcc --version
```

If CUDA is not installed, proceed to step 2.

2. Install Build Tools:

Update package lists and install build-essential:
```bash
# Run this command in the WSL terminal
sudo apt-get update
sudo apt-get install build-essential
```

3. Install CUDA Toolkit:
Follow the [NVIDIA CUDA installation guide for WSL](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=runfile_local).

4. Set CUDA Environment Variables:
Add the following lines to your `.bashrc` file:
```bash
# Add these lines to your .bashrc file in the WSL environment
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
```

Apply the changes:
```bash
# Run this command in the WSL terminal
source ~/.bashrc
```

5. Verify CUDA installation again:
```bash
# Run this command in the WSL terminal
nvcc --version
```

### 2.3 Installing Conda

Install Conda to manage dependencies for ServerlessLLM:
1. Download and install Miniconda by following the [Miniconda installation guide](https://docs.anaconda.com/miniconda/install/).
2. Ensure you follow the **Linux instructions**, not the Windows ones.

## 3. Installing ServerlessLLM

Follow the [ServerlessLLM installation guide](https://serverlessllm.github.io/docs/stable/getting_started/installation/) to set up the software.

Once installed, refer to the [Quick Start Guide](https://serverlessllm.github.io/docs/stable/getting_started/quickstart) for usage instructions.

## 4. Potential Issues and Troubleshooting

### 4.1 Folder Access Issues

If you encounter errors such as:
```bash
No such file or directory: `./models/vllm`
```

This usually indicates a permission or accessibility issue with the `./models` folder. Resolve this by:

1. Checking the folder status:
```bash
# Run this command in the WSL terminal
ls -a
```
If the folder appears in red, it is inaccessible.

2. Deleting and recreating the folder:
```bash
# Run this command in the WSL terminal
sudo rm -rf ./models
mkdir ./models
```
