#!/bin/bash

VLLM_PATH=$(python -c "
try:
    import vllm
    import os
    print(os.path.dirname(os.path.abspath(vllm.__file__)))
except ImportError:
    pass
")

if [ -n \"$VLLM_PATH\" ]; then
    pip uninstall -y vllm
    rm -rf \"$VLLM_PATH\"
    echo \"vllm uninstalled and directory removed.\"
else
    echo \"vllm is not installed.\"
fi
