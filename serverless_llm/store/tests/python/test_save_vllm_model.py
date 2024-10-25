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
from huggingface_hub import snapshot_download
from vllm import LLM


class TestSaveModelIntegration(unittest.TestCase):
    def setUp(self):
        # Set up a temporary directory for the test
        self.model_name = "facebook/opt-1.3b"
        self.torch_dtype = "float16"
        self.tensor_parallel_size = 1
        self.save_dir = "./test_models"
        self.model_path = os.path.join(self.save_dir, self.model_name)

        # Check if at least 2 GPUs are available
        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() < self.tensor_parallel_size
        ):
            raise unittest.SkipTest("Not enough GPUs available for this test")

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
        # with TemporaryDirectory() as cache_dir:
        # cache_dir = "./test_models"
        # download model from huggingface
        input_dir = snapshot_download(
            self.model_name,
            # cache_dir=cache_dir,
            allow_patterns=["*.safetensors", "*.bin", "*.json", "*.txt"],
        )
        # load models from the input directory
        model_executer = LLM(
            model=input_dir,
            download_dir=input_dir,
            dtype=self.torch_dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            num_gpu_blocks_override=1,
            enforce_eager=True,
            max_model_len=1,
        ).llm_engine.model_executor
        # save the models in the ServerlessLLM format
        model_executer.save_serverless_llm_state(path=self.model_path)

        # Check if the model directory was created
        self.assertTrue(os.path.exists(self.model_path))

        # Check if each partition directory was created
        for i in range(self.tensor_parallel_size):
            self.assertTrue(os.path.exists(f"{self.model_path}/rank_{i}"))
        self.assertFalse(
            os.path.exists(
                f"{self.model_path}/rank_{self.tensor_parallel_size}"
            )
        )

        # Check if certain files exist to verify that the model was saved
        expected_files = [
            "tensor_index.json",
            "tensor.data_0",
        ]
        unexpected_files = ["tensor.data_1", "*.bin", "*.safetensors"]
        for i in range(self.tensor_parallel_size):
            for filename in expected_files:
                self.assertTrue(
                    os.path.isfile(
                        os.path.join(self.model_path, f"rank_{i}", filename)
                    )
                )
            for filename in unexpected_files:
                self.assertFalse(
                    os.path.isfile(
                        os.path.join(self.model_path, f"rank_{i}", filename)
                    )
                )

        for filename in unexpected_files:
            self.assertFalse(
                os.path.isfile(
                    os.path.join(self.model_path, f"rank_{i}", filename)
                )
            )
