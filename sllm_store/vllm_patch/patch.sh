#!/bin/bash
# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PATCH_FILE="$SCRIPT_DIR/sllm_load.patch"

# Check if the patch file exists
if [ ! -f "$PATCH_FILE" ]; then
    echo "File does not exist: $PATCH_FILE"
    exit 1
fi

# Get the vLLM installation path
VLLM_PATH=$(python -c "import vllm; import os; print(os.path.dirname(os.path.abspath(vllm.__file__)))")

# Attempt a dry run of the patch to check if it's already applied
if patch -p2 --dry-run -d "$VLLM_PATH" < "$PATCH_FILE" > /dev/null 2>&1; then
    echo "vLLM patch is not applied. Applying the patch now..."
    # Apply the patch
    patch -p2 -d "$VLLM_PATH" < "$PATCH_FILE"
    echo "Patch applied successfully."
else
    echo "vLLM patch has already been applied. Skipping..."
fi