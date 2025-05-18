# tests/new_cli_test/test_deploy.py

import unittest
import json
import tempfile
from unittest import mock
from click.testing import CliRunner

from sllm.clic import cli


class TestDeployCommand(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @mock.patch("sllm._cli_utils.requests.post")
    @mock.patch("sllm._cli_utils.read_config")
    @mock.patch("sllm._cli_utils.os.path.exists", return_value=True)
    def test_deploy_success_with_config(self, mock_exists, mock_read, mock_post):
     
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
                ["deploy", "--model", "facebook/opt-2.7b", "--config", f.name]
            )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Model 'facebook/opt-2.7b' deployed successfully.", result.output)

        
        request_json = mock_post.call_args[1]["json"]
        self.assertEqual(request_json["model"], "facebook/opt-2.7b")
        self.assertEqual(request_json["backend"], "transformers")
        self.assertEqual(request_json["auto_scaling_config"]["min_instances"], 1)

    @mock.patch("sllm._cli_utils.requests.post")
    @mock.patch("sllm._cli_utils.read_config")
    @mock.patch("sllm._cli_utils.os.path.exists", return_value=True)
    def test_deploy_failure_status(self, mock_exists, mock_read, mock_post):
        mock_read.return_value = {
            "model": "",
            "backend": "vllm",
        }
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal Error"

        result = self.runner.invoke(cli, ["deploy", "--model", "m"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Deploy failed with status 500", result.output)

        
        self.assertEqual(mock_post.call_args[1]["json"]["model"], "m")

    @mock.patch("sllm._cli_utils.os.path.exists", return_value=False)
    def test_deploy_missing_default_config(self, mock_exists):
        result = self.runner.invoke(cli, ["deploy", "--model", "x"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Default config not found", result.output)

    @mock.patch("sllm._cli_utils.requests.post")
    @mock.patch("sllm._cli_utils.read_config")
    @mock.patch("sllm._cli_utils.os.path.exists", return_value=True)
    def test_deploy_with_extra_flags(self, mock_exists, mock_read, mock_post):
        mock_read.return_value = {
            "model": "",
            "backend": "vllm",
            "auto_scaling_config": {},
            "backend_config": {}
        }
        mock_post.return_value.status_code = 200

        result = self.runner.invoke(cli, [
            "deploy", "--model", "facebook/opt-1.3b", "--backend", "transformers",
            "--num-gpus", "2", "--target", "3", "--min-instances", "1", "--max-instances", "5"
        ])
        self.assertEqual(result.exit_code, 0)

        
        request_data = mock_post.call_args[1]["json"]
        self.assertEqual(request_data["model"], "facebook/opt-1.3b")
        self.assertEqual(request_data["backend"], "transformers")
        self.assertEqual(request_data["num_gpus"], 2)
        self.assertEqual(request_data["auto_scaling_config"]["target"], 3)
        self.assertEqual(request_data["auto_scaling_config"]["min_instances"], 1)
        self.assertEqual(request_data["auto_scaling_config"]["max_instances"], 5)


if __name__ == "__main__":
    unittest.main()
