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
import os
import shutil
import unittest

import torch
from sllm_store.transformers import save_model
from transformers import AutoModelForCausalLM


class TestSaveModelIntegration(unittest.TestCase):
    def setUp(self):
        # Set up a temporary directory for the test
        self.model_name = "facebook/opt-1.3b"
        self.save_dir = "./test_models"
        self.model_path = os.path.join(self.save_dir, self.model_name)

        # Ensure the save directory is clean before the test
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

        os.environ["STORAGE_PATH"] = self.save_dir

    def tearDown(self):
        # Clean up by deleting the directory after the test
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def test_save_model(self):
        # Load the actual model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        )

        # Save the model
        save_model(model, self.model_path)

        # Check if the model directory was created
        self.assertTrue(os.path.exists(self.model_path))

        # Check if certain files exist to verify that the model was saved
        expected_files = [
            "config.json",
            "generation_config.json",
            "no_split_modules.json",
            "tensor_index.json",
            "tensor.data_0",
            "tied_no_split_modules.json",
        ]
        for filename in expected_files:
            self.assertTrue(
                os.path.isfile(os.path.join(self.model_path, filename))
            )

        unexpected_files = ["tensor.data_1", "*.bin", "*.safetensors"]
        for filename in unexpected_files:
            self.assertFalse(
                os.path.isfile(os.path.join(self.model_path, filename))
            )
