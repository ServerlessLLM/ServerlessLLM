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
PATCH_FILE="$SCRIPT_DIR/sllmLoad.patch"
if [ ! -f "$PATCH_FILE" ]; then
    echo "File does not exist: $PATCH_FILE"
    exit 1
fi

SGLANG_PATH_OUTPUT=$(python -c "import sglang; import os; print(os.path.dirname(os.path.abspath(sglang.__file__)))" 2>/dev/null)
SGLANG_PATH=$(echo "$SGLANG_PATH_OUTPUT" | tail -n 1)

# Sanity check the path
echo "Detected SGLANG_PATH: '$SGLANG_PATH'"
if [ ! -d "$SGLANG_PATH" ]; then
    echo "Error: Detected SGLANG_PATH is not a valid directory: '$SGLANG_PATH'"
    echo "Full output from python command was:"
    echo "$SGLANG_PATH_OUTPUT"
    exit 1
fi

patch -p2 -d $SGLANG_PATH -R < $PATCH_FILE