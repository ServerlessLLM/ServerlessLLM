#!/usr/bin/env bash
# ---------------------------------------------------------------------------- #
#  apply_sllm_patch.sh                                                         #
#  Copyright (c) SGLang Team 2025                                               #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#      http://www.apache.org/licenses/LICENSE-2.0                              #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")"; pwd)
PATCH_FILE="$SCRIPT_DIR/sllmLoad.patch"
if [ ! -f "$PATCH_FILE" ]; then
    echo "File does not exist: $PATCH_FILE"
    exit 1
fi

SGLANG_PATH_OUTPUT=$(python3 - <<'EOF'
import os, sglang
print(os.path.dirname(os.path.abspath(sglang.__file__)))
EOF
)

SGLANG_PATH=$(echo "$SGLANG_PATH_OUTPUT" | tail -n1)
echo "Detected SGLANG_PATH: '$SGLANG_PATH'"

if [ ! -d "$SGLANG_PATH" ]; then
    echo "Error: Detected SGLANG_PATH is not a valid directory: '$SGLANG_PATH'"
    echo "Python output was:"
    echo "$SGLANG_PATH_OUTPUT"
    exit 1
fi

# Check if patch is already applied
if patch -p2 --dry-run -d "$SGLANG_PATH" < "$PATCH_FILE" >/dev/null 2>&1; then
    echo "→ SGLang has not been patched yet, preparing to apply..."
    patch -p2 -d "$SGLANG_PATH" < "$PATCH_FILE"
    echo "✔ Patch applied successfully."
else
    echo "→ Patch already exists, skipping application."
fi
