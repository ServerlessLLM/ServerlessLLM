import os
import shutil
import unittest
from pathlib import Path
from click.testing import CliRunner
from sllm_store.cli import cli
from huggingface_hub import snapshot_download

TEST_MODEL_VLLM = "facebook/opt-125m"
TEST_MODEL_TRANSFORMERS = "facebook/opt-125m"
TEST_ADAPTER_MODEL = (
    "jeanlucmarsh/opt-125m-pattern-based_finetuning_with_lora-mnli-mm-d3_fs2"
)
TEST_DIRECTORY_SAVE = "./test_save_models"
TEST_DIRECTORY_LOAD = "./models"


class TestCliCommands(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = CliRunner()

        result = cls.runner.invoke(
            cli,
            [
                "save",
                "--model",
                TEST_MODEL_VLLM,
                "--backend",
                "vllm",
                "--storage-path",
                TEST_DIRECTORY_LOAD + "/vllm",
            ],
        )
        assert result.exit_code == 0, f"Initial save failed: {result.output}"

        result = cls.runner.invoke(
            cli,
            [
                "save",
                "--model",
                TEST_MODEL_TRANSFORMERS,
                "--backend",
                "transformers",
                "--storage-path",
                TEST_DIRECTORY_LOAD + "/transformers",
            ],
        )
        assert result.exit_code == 0, f"Initial save failed: {result.output}"

        result = cls.runner.invoke(
            cli,
            [
                "save",
                "--model",
                TEST_MODEL_TRANSFORMERS,
                "--backend",
                "transformers",
                "--adapter-name",
                TEST_ADAPTER_MODEL,
                "--storage-path",
                TEST_DIRECTORY_LOAD,
            ],
        )
        assert result.exit_code == 0, f"Initial save failed: {result.output}"

    def tearDown(self):
        path = Path(TEST_DIRECTORY_SAVE)
        if path.exists() and path.is_dir():
            try:
                shutil.rmtree(path)
                print(f"Removed old test folder: {path}")
            except OSError as e:
                print(f"Error removing {path}: {e}")

    @classmethod
    def tearDownClass(cls):
        path = Path(TEST_DIRECTORY_LOAD)
        if path.exists() and path.is_dir():
            try:
                shutil.rmtree(path)
                print(f"[tearDownClass] Removed load folder: {path}")
            except OSError as e:
                print(f"[tearDownClass] Error removing {path}: {e}")

    # SAVE

    def test_save_vllm_model_default(self):
        result = self.runner.invoke(
            cli, ["save", "--model", TEST_MODEL_VLLM, "--backend", "vllm"]
        )
        self.assertEqual(
            result.exit_code,
            0,
            f"Command failed with output: {result.output}\n"
            f"Stderr: {result.stderr}",
        )
        expected_folder = Path("models") / TEST_MODEL_VLLM
        self.assertTrue(
            expected_folder.is_dir(),
            f"Folder does not exist: {expected_folder}",
        )

        # Check if each partition directory was created
        tensor_parallel_size = 1
        for i in range(tensor_parallel_size):
            self.assertTrue(os.path.exists(f"{expected_folder}/rank_{i}"))
        self.assertFalse(
            os.path.exists(f"{expected_folder}/rank_{tensor_parallel_size}")
        )

        # Check if certain files exist to verify that the model was saved
        expected_files = [
            "tensor_index.json",
            "tensor.data_0",
        ]
        unexpected_files = ["tensor.data_1", "*.bin", "*.safetensors"]
        for i in range(tensor_parallel_size):
            for filename in expected_files:
                self.assertTrue(
                    os.path.isfile(
                        os.path.join(expected_folder, f"rank_{i}", filename)
                    )
                )
            for filename in unexpected_files:
                self.assertFalse(
                    os.path.isfile(
                        os.path.join(expected_folder, f"rank_{i}", filename)
                    )
                )

        for filename in unexpected_files:
            self.assertFalse(
                os.path.isfile(
                    os.path.join(expected_folder, f"rank_{i}", filename)
                )
            )

    def test_save_vllm_model_params(self):
        tensor_parallel_size = (
            1  # Change to 2 if you have multiple GPUs and want to test it
        )
        result = self.runner.invoke(
            cli,
            [
                "save",
                "--model",
                TEST_MODEL_VLLM,
                "--backend",
                "vllm",
                "--tensor-parallel-size",
                str(tensor_parallel_size),
                "--storage-path",
                TEST_DIRECTORY_SAVE,
            ],
        )
        self.assertEqual(
            result.exit_code,
            0,
            f"Command failed with output: {result.output}\n"
            f"Stderr: {result.stderr}",
        )
        expected_folder = Path(TEST_DIRECTORY_SAVE) / TEST_MODEL_VLLM
        self.assertTrue(
            expected_folder.is_dir(),
            f"Folder does not exist: {expected_folder}",
        )

        # Check if each partition directory was created
        for i in range(tensor_parallel_size):
            self.assertTrue(os.path.exists(f"{expected_folder}/rank_{i}"))
        self.assertFalse(
            os.path.exists(f"{expected_folder}/rank_{tensor_parallel_size}")
        )

        # Check if certain files exist to verify that the model was saved
        expected_files = [
            "tensor_index.json",
            "tensor.data_0",
        ]
        unexpected_files = ["tensor.data_1", "*.bin", "*.safetensors"]
        for i in range(tensor_parallel_size):
            for filename in expected_files:
                self.assertTrue(
                    os.path.isfile(
                        os.path.join(expected_folder, f"rank_{i}", filename)
                    )
                )
            for filename in unexpected_files:
                self.assertFalse(
                    os.path.isfile(
                        os.path.join(expected_folder, f"rank_{i}", filename)
                    )
                )

        for filename in unexpected_files:
            self.assertFalse(
                os.path.isfile(
                    os.path.join(expected_folder, f"rank_{i}", filename)
                )
            )

    def test_save_vllm_model_local_path(self):
        local_dummy_model_path = Path(TEST_DIRECTORY_SAVE) / "dummy_local_model"
        snapshot_download(
            repo_id=TEST_MODEL_VLLM,
            local_dir=local_dummy_model_path,
            local_dir_use_symlinks=False,
        )

        result = self.runner.invoke(
            cli,
            [
                "save",
                "--model",
                TEST_MODEL_VLLM,
                "--backend",
                "vllm",
                "--local-model-path",
                str(local_dummy_model_path),
                "--storage-path",
                TEST_DIRECTORY_SAVE,
            ],
        )

        self.assertEqual(
            result.exit_code,
            0,
            f"Command failed with output: {result.output}\n"
            f"Stderr: {result.stderr}",
        )
        expected_folder = Path(TEST_DIRECTORY_SAVE) / TEST_MODEL_VLLM
        self.assertTrue(
            expected_folder.is_dir(),
            f"Folder does not exist: {expected_folder}",
        )

        # Check if each partition directory was created
        tensor_parallel_size = 1
        for i in range(tensor_parallel_size):
            self.assertTrue(os.path.exists(f"{expected_folder}/rank_{i}"))
        self.assertFalse(
            os.path.exists(f"{expected_folder}/rank_{tensor_parallel_size}")
        )

        # Check if certain files exist to verify that the model was saved
        expected_files = [
            "tensor_index.json",
            "tensor.data_0",
        ]
        unexpected_files = ["tensor.data_1", "*.bin", "*.safetensors"]
        for i in range(tensor_parallel_size):
            for filename in expected_files:
                self.assertTrue(
                    os.path.isfile(
                        os.path.join(expected_folder, f"rank_{i}", filename)
                    )
                )
            for filename in unexpected_files:
                self.assertFalse(
                    os.path.isfile(
                        os.path.join(expected_folder, f"rank_{i}", filename)
                    )
                )

        for filename in unexpected_files:
            self.assertFalse(
                os.path.isfile(
                    os.path.join(expected_folder, f"rank_{i}", filename)
                )
            )

    def test_save_transformers_model_default(self):
        result = self.runner.invoke(
            cli,
            [
                "save",
                "--model",
                TEST_MODEL_TRANSFORMERS,
                "--backend",
                "transformers",
            ],
        )
        self.assertEqual(
            result.exit_code,
            0,
            f"Command failed with output: {result.output}\n"
            f"Stderr: {result.stderr}",
        )
        expected_folder = Path("./models") / TEST_MODEL_TRANSFORMERS
        self.assertTrue(
            expected_folder.is_dir(),
            f"Folder does not exist: {expected_folder}",
        )
        self.assertTrue(
            any(expected_folder.iterdir()),
            f"Folder is empty: {expected_folder}",
        )

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
                os.path.isfile(os.path.join(expected_folder, filename))
            )

        unexpected_files = ["tensor.data_1", "*.bin", "*.safetensors"]
        for filename in unexpected_files:
            self.assertFalse(
                os.path.isfile(os.path.join(expected_folder, filename))
            )

    def test_save_transformers_adapter_valid(self):
        result = self.runner.invoke(
            cli,
            [
                "save",
                "--model",
                TEST_MODEL_TRANSFORMERS,
                "--backend",
                "transformers",
                "--storage-path",
                TEST_DIRECTORY_SAVE,
            ],
        )

        result = self.runner.invoke(
            cli,
            [
                "save",
                "--model",
                TEST_MODEL_TRANSFORMERS,
                "--backend",
                "transformers",
                "--adapter-name",
                TEST_ADAPTER_MODEL,
                "--storage-path",
                TEST_DIRECTORY_SAVE,
            ],
        )
        self.assertEqual(
            result.exit_code,
            0,
            f"Command failed with output: {result.output}\n"
            f"Stderr: {result.stderr}",
        )
        expected_folder = (
            Path(TEST_DIRECTORY_SAVE) / TEST_ADAPTER_MODEL
        )  # Adapters save under adapter_name
        self.assertTrue(
            expected_folder.is_dir(),
            f"Adapter folder does not exist: {expected_folder}",
        )
        self.assertTrue(
            any(expected_folder.iterdir()),
            f"Adapter folder is empty: {expected_folder}",
        )

        # Check if certain files exist to verify that the model was saved
        expected_files = [
            "tensor_index.json",
            "tensor.data_0",
        ]
        for filename in expected_files:
            self.assertTrue(
                os.path.isfile(os.path.join(expected_folder, filename))
            )

        unexpected_files = ["tensor.data_1", "*.bin", "*.safetensors"]
        for filename in unexpected_files:
            self.assertFalse(
                os.path.isfile(os.path.join(expected_folder, filename))
            )

    def test_save_invalid_backend(self):
        result = self.runner.invoke(
            cli,
            [
                "save",
                "--model",
                "facebook/opt-125m",
                "--backend",
                "unsupported_backend",
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertFalse(
            (Path(TEST_DIRECTORY_SAVE) / "facebook/opt-125m").exists()
        )

    def test_save_missing_model_name(self):
        result = self.runner.invoke(cli, ["save", "--backend", "vllm"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Missing option '--model'", result.stderr)

    def test_save_non_existent_hf_model(self):
        non_existent_model = "nonexistent/model-xyz-123"
        result = self.runner.invoke(
            cli,
            [
                "save",
                "--model",
                non_existent_model,
                "--backend",
                "transformers",
                "--storage-path",
                TEST_DIRECTORY_SAVE,
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertFalse(
            (Path(TEST_DIRECTORY_SAVE) / non_existent_model).exists()
        )

    # LOAD

    def test_load_vllm_model_default(self):
        result = self.runner.invoke(
            cli,
            [
                "load",
                "--model",
                TEST_MODEL_VLLM,
                "--backend",
                "vllm",
                "--storage-path",
                TEST_DIRECTORY_LOAD + "/vllm",
            ],
        )
        self.assertEqual(
            result.exit_code,
            0,
            f"Command failed with output: {result.output}\n"
            f"Stderr: {result.stderr}",
        )
        self.assertIn("Prompt:", result.output)
        self.assertIn("Generated text:", result.output)

    def test_load_transformers_model_default(self):
        result = self.runner.invoke(
            cli,
            [
                "load",
                "--model",
                "transformers/" + TEST_MODEL_TRANSFORMERS,
                "--backend",
                "transformers",
                "--storage-path",
                TEST_DIRECTORY_LOAD,
            ],
        )
        self.assertEqual(
            result.exit_code,
            0,
            f"Command failed with output: {result.output}\n"
            f"Stderr: {result.stderr}",
        )
        self.assertIn("Hello, my dog is cute", result.output)

    def test_load_transformers_precision_int8(self):
        result = self.runner.invoke(
            cli,
            [
                "load",
                "--model",
                "transformers/" + TEST_MODEL_TRANSFORMERS,
                "--backend",
                "transformers",
                "--precision",
                "int8",
                "--storage-path",
                TEST_DIRECTORY_LOAD,
            ],
        )
        self.assertEqual(
            result.exit_code,
            0,
            f"Command failed with output: {result.output}\n"
            f"Stderr: {result.stderr}",
        )
        self.assertIn("Hello, my dog is cute", result.output)

    def test_load_transformers_precision_fp4(self):
        result = self.runner.invoke(
            cli,
            [
                "load",
                "--model",
                "transformers/" + TEST_MODEL_TRANSFORMERS,
                "--backend",
                "transformers",
                "--precision",
                "fp4",
                "--storage-path",
                TEST_DIRECTORY_LOAD,
            ],
        )
        self.assertEqual(
            result.exit_code,
            0,
            f"Command failed with output: {result.output}\n"
            f"Stderr: {result.stderr}",
        )
        self.assertIn("Hello, my dog is cute", result.output)

    def test_load_transformers_precision_nf4(self):
        result = self.runner.invoke(
            cli,
            [
                "load",
                "--model",
                "transformers/" + TEST_MODEL_TRANSFORMERS,
                "--backend",
                "transformers",
                "--precision",
                "nf4",
                "--storage-path",
                TEST_DIRECTORY_LOAD,
            ],
        )
        self.assertEqual(
            result.exit_code,
            0,
            f"Command failed with output: {result.output}\n"
            f"Stderr: {result.stderr}",
        )
        self.assertIn("Hello, my dog is cute", result.output)

    def test_load_transformers_precision_invalid(self):
        result = self.runner.invoke(
            cli,
            [
                "load",
                "--model",
                TEST_MODEL_TRANSFORMERS,
                "--backend",
                "transformers",
                "--precision",
                "invalid_precision",
            ],
        )
        self.assertNotEqual(result.exit_code, 0)

    def test_load_transformers_adapter_valid(self):
        result = self.runner.invoke(
            cli,
            [
                "load",
                "--model",
                "transformers/" + TEST_MODEL_TRANSFORMERS,
                "--backend",
                "transformers",
                "--adapter-name",
                TEST_ADAPTER_MODEL,
                "--storage-path",
                TEST_DIRECTORY_LOAD,
            ],
        )
        self.assertEqual(
            result.exit_code,
            0,
            f"Command failed with output: {result.output}\n"
            f"Stderr: {result.stderr}",
        )
        self.assertIn("Hello, my dog is cute", result.output)

    def test_load_invalid_backend(self):
        result = self.runner.invoke(
            cli,
            [
                "load",
                "--model",
                TEST_MODEL_TRANSFORMERS,
                "--backend",
                "unsupported_backend",
            ],
        )
        self.assertNotEqual(result.exit_code, 0)

    def test_load_non_existent_model(self):
        non_existent_model = "nonexistent/model-xyz-123"
        result = self.runner.invoke(
            cli,
            [
                "load",
                "--model",
                "transformers/" + non_existent_model,
                "--backend",
                "transformers",
                "--storage-path",
                TEST_DIRECTORY_LOAD,
            ],
        )
        self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
