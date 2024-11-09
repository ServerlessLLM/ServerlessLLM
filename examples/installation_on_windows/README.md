To install the SLLM dependencies, Ubuntu 20.04 is required by the installation guidance states. Packages, like vllm, cannot be installed on windows. Therefore, we need a sub-system to run linux on windows machines. Here, I recommand downloading the Ubuntu app from microsoft store and the wsl connection will allow to you install everything needed for. Generally speaking, it is a developed and stable app except wsl connection doesn't support GUI and sometimes it can be nesty to set variable names.

![alt text](Ubtuntu-app.png)

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

A step by step guidance is available on YT https://www.youtube.com/watch?v=sUsTQTJFmjs given by ProgrammingKnowledge, it is really detailed (basically telling you which button to click).

If everything goes right, you should have this terminal. The command line starts with your pre-set username.
![alt text](wsl-terminal.png)

