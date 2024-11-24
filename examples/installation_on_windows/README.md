# Install ServerlessLLM on Windows Machines

## 1 Windows Subsystem for Linux
To install the ServerlessLLM dependencies, Ubuntu 20.04 is required by the installation guidance states. Packages, like vllm, cannot be installed on Windows. Therefore, the Windows machines require a sub-system to run Linux. Here, it's recommended downloading the Ubuntu app from Microsoft store and the WSL connection will allow to you install everything needed for. If you have more familiar to start Linux environment, please feel free to go ahead and share your experience with us. Generally speaking, it is a developed and stable app except WSL connection doesn't support GUI, and sometimes it can be nasty to set variable path.

![alt text](Ubuntu-app.png)

Description:
Install a complete Ubuntu terminal environment in minutes with Windows Subsystem for Linux (WSL). Develop cross-platform applications, improve your data science or web development workflows and manage IT infrastructure without leaving Windows.

For more information about Ubuntu WSL and how Canonical supports developers please visit: https://ubuntu.com/wsl

A step-by-step guidance is available on YT https://www.youtube.com/watch?v=sUsTQTJFmjs given by ProgrammingKnowledge, it is really detailed (basically telling you which button to click).

If everything goes right, you should have this terminal. The command line starts with your pre-set username.

<img src="wsl-terminal.png" alt="Description" width="800"/>


## 2 Configure the WSL connection
Back to the installation of ServerlessLLM, use `free -h` to check the memory status. After testing, the minimum setting is 17GB Mem and 8GB Swap as shown below. If it didn't meet this requirement, you need to set manually in the file `.wslconfig` which should locate in `C:\Users\your_username\`. If the file does not exist, simply create it. Within the file, copy and paste the setting below. WSL2's VM allocates a certain amount of memory, and Swap space is managed within that VM's memory allocation. This means the Swap memory potentially consumes the RAM in WSL connection. Therefore, the memory needs to be much bigger than Swap to run the ServerlessLLM successfully. After testing on different WSL config settings, the minimum memory and swap are shown below. Please feel free to increase them when necessary.

    [wsl2]               # this line is a must-have
    memory=17GB          # Limits VM memory
    swap=8GB             # Sets swap file size

Next, restart the WSL connection to implement the changes by `wsl --shutdown` preceding `wsl`. Then, following the installation guidance using pip install, everything should work.

## Potential issues:

### 3.1 CUDA Issue

If it shows error message like 'error while loading shared libraries: libcudart.so.12: cannot open shared object file: No such file or directory', you need to double-check if CUDA v12 are correctly installed in the ServerlessLLM and sllm-worker envs. Using 'nvcc --version' can check the installed version. The correct setting should be looking like this.

![alt text](cuda-version-check.png)

### 3.2 Folder Access Issue

If an error occurred while saving the model: No such file or directory: `./models/vllm`, it is probably because the `./models` folder created on the WSL connection are not accessible.
Running the codes below does a diagnosis on this access issue.

    cd examples/installation_on_windows/
    python model_folder_checker.py