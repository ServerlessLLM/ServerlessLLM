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

from peft import LoraConfig, get_peft_model
from sllm_store.transformers import save_lora
from transformers import AutoModelForCausalLM


class TestSaveLoraIntegration(unittest.TestCase):
    def setUp(self):
        # Set up a temporary directory for the test
        self.model_name = "bigscience/bloomz-560m"
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

    def test_save_lora(self):
        # Fine-tuning the actual peft model
        base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["query_key_value"],
            lora_dropout=0.1,
            bias="lora_only",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(base_model, lora_config)
        # Save the model
        save_lora(peft_model, self.model_path)

        # Check if the model directory was created
        self.assertTrue(os.path.exists(self.model_path))

        # Check if certain files exist to verify that the model was saved
        expected_files = [
            "tensor_index.json",
            "tensor.data_0",
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
