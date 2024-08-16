import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

from serverless_llm.cli.deploy import DeployCommand


class TestDeployCommand(unittest.TestCase):
    @patch("serverless_llm.cli.deploy.read_config")
    @patch("serverless_llm.cli.deploy.requests.post")
    def test_deploy_with_model_only(self, mock_post, mock_read_config):
        # Mock read_config to return a default configuration
        mock_read_config.return_value = {
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

        # Mock the response of the requests.post
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        args = Namespace(
            model="facebook/opt-1.3b",
            config=None,
            backend=None,
            num_gpus=None,
            target=None,
            min_instances=None,
            max_instances=None,
        )
        command = DeployCommand(args)
        command.run()

        mock_read_config.assert_called_once()
        mock_post.assert_called_once()
        self.assertEqual(
            mock_post.call_args[1]["json"]["model"], "facebook/opt-1.3b"
        )

    @patch("serverless_llm.cli.deploy.read_config")
    @patch("serverless_llm.cli.deploy.requests.post")
    def test_deploy_with_custom_config(self, mock_post, mock_read_config):
        # Mock the default config
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
        mock_read_config.side_effect = [
            default_config,
            {"model": "custom-model", "backend": "transformers"},
        ]

        # Mock the response of the requests.post
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        args = Namespace(
            model=None,
            config="path/to/config.json",
            backend=None,
            num_gpus=None,
            target=None,
            min_instances=None,
            max_instances=None,
        )
        command = DeployCommand(args)
        command.run()

        expected_config = default_config
        expected_config.update(
            {"model": "custom-model", "backend": "transformers"}
        )

        mock_read_config.assert_called()
        mock_post.assert_called_once()
        self.assertEqual(
            mock_post.call_args[1]["json"]["model"], "custom-model"
        )
        self.assertEqual(
            mock_post.call_args[1]["json"]["backend"], "transformers"
        )

    @patch("serverless_llm.cli.deploy.read_config")
    def test_validate_config(self, mock_read_config):
        mock_read_config.return_value = {
            "model": "facebook/opt-1.3b",
            "backend": "transformers",
            "num_gpus": 1,
            "auto_scaling_config": {
                "target": 1,
                "min_instances": 1,
                "max_instances": 5,
            },
            "backend_config": {
                "pretrained_model_name_or_path": "facebook/opt-1.3b",
                "device_map": "auto",
                "torch_dtype": "float16",
            },
        }

        args = Namespace(
            model="facebook/opt-1.3b",
            config=None,
            backend=None,
            num_gpus=None,
            target=None,
            min_instances=None,
            max_instances=None,
        )
        command = DeployCommand(args)

        valid_config = mock_read_config.return_value
        command.validate_config(valid_config)  # Should not raise any exception

        invalid_config = valid_config.copy()
        invalid_config["num_gpus"] = -1
        with self.assertRaises(ValueError):
            command.validate_config(invalid_config)

    @patch("serverless_llm.cli.deploy.read_config")
    def test_update_config(self, mock_read_config):
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
        provided_config = {
            "model": "facebook/opt-2.7b",
            "backend": "transformers",
        }

        args = Namespace(
            model=None,
            config="path/to/config.json",
            backend=None,
            num_gpus=None,
            target=None,
            min_instances=None,
            max_instances=None,
        )
        command = DeployCommand(args)
        updated_config = command.update_config(default_config, provided_config)

        self.assertEqual(updated_config["model"], "facebook/opt-2.7b")
        self.assertEqual(updated_config["backend"], "transformers")
        self.assertEqual(
            updated_config["auto_scaling_config"]["target"], 1
        )  # Should remain as default

    @patch("serverless_llm.cli.deploy.requests.post")
    def test_deploy_model_success(self, mock_post):
        # Mock the response of the requests.post
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        config_data = {
            "model": "facebook/opt-2.7b",
            "backend": "transformers",
            "num_gpus": 2,
            "auto_scaling_config": {
                "target": 2,
                "min_instances": 1,
                "max_instances": 4,
            },
            "backend_config": {
                "pretrained_model_name_or_path": "facebook/opt-2.7b",
                "device_map": "auto",
                "torch_dtype": "float16",
            },
        }

        args = Namespace(
            model="facebook/opt-2.7b",
            config=None,
            backend=None,
            num_gpus=None,
            target=None,
            min_instances=None,
            max_instances=None,
        )
        command = DeployCommand(args)
        command.deploy_model(config_data)

        mock_post.assert_called_once()
        self.assertEqual(
            mock_post.call_args[1]["json"]["model"], "facebook/opt-2.7b"
        )


if __name__ == "__main__":
    unittest.main()
