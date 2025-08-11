# tests/new_cli_test/test_deploy.py

import json
import tempfile
import unittest
from unittest import mock

from click.testing import CliRunner

from sllm.cli.clic import cli


class TestDeployCommand(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @mock.patch("sllm.cli._cli_utils.requests.post")
    @mock.patch("sllm.cli._cli_utils.read_config")
    @mock.patch("sllm.cli._cli_utils.os.path.exists", return_value=True)
    def test_deploy_success_with_config(
        self, mock_exists, mock_read, mock_post
    ):
        default_config = {
            "model": "",
            "backend": "vllm",
            "num_gpus": 1,
            "auto_scaling_config": {
                "target": 1,
                "min_instances": 1,
                "max_instances": 5,
            },
            "backend_config": {
                "pretrained_model_name_or_path": "",
                "device_map": "auto",
                "torch_dtype": "float16",
            },
        }
        user_config = {
            "backend": "transformers",
        }

        mock_read.side_effect = [default_config, user_config]
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = "OK"

        with tempfile.NamedTemporaryFile("w+", suffix=".json") as f:
            json.dump(user_config, f)
            f.flush()
            result = self.runner.invoke(
                cli,
                ["deploy", "--model", "facebook/opt-2.7b", "--config", f.name],
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "[✅ SUCCESS] Model 'facebook/opt-2.7b' deployed successfully.",
            result.output,
        )

        request_json = mock_post.call_args[1]["json"]
        self.assertEqual(request_json["model"], "facebook/opt-2.7b")
        self.assertEqual(request_json["backend"], "transformers")
        self.assertEqual(
            request_json["auto_scaling_config"]["min_instances"], 1
        )

    @mock.patch("sllm.cli._cli_utils.requests.post")
    @mock.patch("sllm.cli._cli_utils.read_config")
    @mock.patch("sllm.cli._cli_utils.os.path.exists", return_value=True)
    def test_deploy_failure_status(self, mock_exists, mock_read, mock_post):
        mock_read.return_value = {
            "model": "",
            "backend": "vllm",
        }
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal Error"

        result = self.runner.invoke(cli, ["deploy", "--model", "m"])
        self.assertEqual(result.exit_code, 1)  # Should exit with error code
        self.assertIn(
            "[❌ ERROR] Deploy failed with status 500: Internal Error",
            result.output,
        )

        self.assertEqual(mock_post.call_args[1]["json"]["model"], "m")

    @mock.patch("sllm.cli._cli_utils.os.path.exists", return_value=False)
    def test_deploy_missing_default_config(self, mock_exists):
        result = self.runner.invoke(cli, ["deploy", "--model", "x"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("[ERROR] Default config not found", result.output)

    @mock.patch("sllm.cli._cli_utils.requests.post")
    @mock.patch("sllm.cli._cli_utils.read_config")
    @mock.patch("sllm.cli._cli_utils.os.path.exists", return_value=True)
    def test_deploy_with_extra_flags(self, mock_exists, mock_read, mock_post):
        mock_read.return_value = {
            "model": "",
            "backend": "vllm",
            "auto_scaling_config": {},
            "backend_config": {},
        }
        mock_post.return_value.status_code = 200

        result = self.runner.invoke(
            cli,
            [
                "deploy",
                "--model",
                "facebook/opt-1.3b",
                "--backend",
                "transformers",
                "--num-gpus",
                "2",
                "--target",
                "3",
                "--min-instances",
                "1",
                "--max-instances",
                "5",
            ],
        )
        self.assertEqual(result.exit_code, 0)

        request_data = mock_post.call_args[1]["json"]
        self.assertEqual(request_data["model"], "facebook/opt-1.3b")
        self.assertEqual(request_data["backend"], "transformers")
        self.assertEqual(request_data["num_gpus"], 2)
        self.assertEqual(request_data["auto_scaling_config"]["target"], 3)
        self.assertEqual(
            request_data["auto_scaling_config"]["min_instances"], 1
        )
        self.assertEqual(
            request_data["auto_scaling_config"]["max_instances"], 5
        )

    # NEW TESTS TO CATCH REAL CLI BUGS:

    @mock.patch("sllm.cli.clic.deploy_model")
    def test_lora_adapters_parsing_space_separated(self, mock_deploy):
        """Test that LoRA adapter parsing in CLI works correctly with space separation."""
        result = self.runner.invoke(
            cli,
            [
                "deploy",
                "--model",
                "test-model",
                "--lora-adapters",
                "adapter1=/path1 adapter2=/path2",
            ],
        )

        # Verify the parsed adapters are passed correctly
        mock_deploy.assert_called_once()
        call_args = mock_deploy.call_args[1]
        expected_adapters = {"adapter1": "/path1", "adapter2": "/path2"}
        self.assertEqual(call_args["lora_adapters"], expected_adapters)

    @mock.patch("sllm.cli.clic.deploy_model")
    def test_lora_adapters_parsing_comma_separated(self, mock_deploy):
        """Test that LoRA adapter parsing in CLI works correctly with comma separation."""
        result = self.runner.invoke(
            cli,
            [
                "deploy",
                "--model",
                "test-model",
                "--lora-adapters",
                "adapter1=/path1,adapter2=/path2",
            ],
        )

        mock_deploy.assert_called_once()
        call_args = mock_deploy.call_args[1]
        expected_adapters = {"adapter1": "/path1", "adapter2": "/path2"}
        self.assertEqual(call_args["lora_adapters"], expected_adapters)

    def test_lora_adapters_invalid_format_error(self):
        """Test that invalid LoRA adapter format shows error message."""
        result = self.runner.invoke(
            cli,
            [
                "deploy",
                "--model",
                "test-model",
                "--lora-adapters",
                "invalid-format-no-equals",
            ],
        )

        # Should show error message for invalid format
        self.assertIn(
            "[ERROR] Invalid LoRA module format: invalid-format-no-equals",
            result.output,
        )

    @mock.patch("sllm.cli.clic.deploy_model")
    def test_boolean_flags_passed_correctly(self, mock_deploy):
        """Test that boolean flags are passed correctly to deploy_model."""
        result = self.runner.invoke(
            cli, ["deploy", "--model", "test-model", "--enable-lora"]
        )

        mock_deploy.assert_called_once()
        call_args = mock_deploy.call_args[1]
        self.assertTrue(call_args["enable_lora"])

    @mock.patch("sllm.cli.clic.deploy_model")
    def test_integer_type_conversion(self, mock_deploy):
        """Test that string arguments are converted to integers where expected."""
        result = self.runner.invoke(
            cli,
            [
                "deploy",
                "--model",
                "test-model",
                "--num-gpus",
                "4",
                "--target",
                "100",
                "--min-instances",
                "2",
                "--max-instances",
                "10",
            ],
        )

        mock_deploy.assert_called_once()
        call_args = mock_deploy.call_args[1]

        # These should be integers, not strings
        self.assertIsInstance(call_args["num_gpus"], int)
        self.assertIsInstance(call_args["target"], int)
        self.assertIsInstance(call_args["min_instances"], int)
        self.assertIsInstance(call_args["max_instances"], int)

        self.assertEqual(call_args["num_gpus"], 4)
        self.assertEqual(call_args["target"], 100)
        self.assertEqual(call_args["min_instances"], 2)
        self.assertEqual(call_args["max_instances"], 10)

    def test_invalid_integer_argument(self):
        """Test that invalid integer arguments are handled properly."""
        result = self.runner.invoke(
            cli,
            ["deploy", "--model", "test-model", "--num-gpus", "not-a-number"],
        )

        # Click should catch this and show an error
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid value", result.output)

    @mock.patch("sllm.cli.clic.deploy_model")
    def test_all_parameters_passed_through(self, mock_deploy):
        """Test that all CLI parameters are correctly passed to deploy_model function."""
        result = self.runner.invoke(
            cli,
            [
                "deploy",
                "--model",
                "test-model",
                "--config",
                "test.json",
                "--backend",
                "vllm",
                "--num-gpus",
                "2",
                "--target",
                "50",
                "--min-instances",
                "1",
                "--max-instances",
                "5",
                "--lora-adapters",
                "adapter1=/path1",
                "--enable-lora",
                "--precision",
                "fp16",
            ],
        )

        mock_deploy.assert_called_once()
        call_args = mock_deploy.call_args[1]

        # Verify all parameters are passed correctly
        self.assertEqual(call_args["model"], "test-model")
        self.assertEqual(call_args["config"], "test.json")
        self.assertEqual(call_args["backend"], "vllm")
        self.assertEqual(call_args["num_gpus"], 2)
        self.assertEqual(call_args["target"], 50)
        self.assertEqual(call_args["min_instances"], 1)
        self.assertEqual(call_args["max_instances"], 5)
        self.assertEqual(call_args["lora_adapters"], {"adapter1": "/path1"})
        self.assertTrue(call_args["enable_lora"])
        self.assertEqual(call_args["precision"], "fp16")


if __name__ == "__main__":
    unittest.main()
