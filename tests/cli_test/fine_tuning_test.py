import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

from sllm.cli.fine_tuning import FineTuningCommand


class TestFineTuningCommand(unittest.TestCase):
    @patch("sllm.cli.fine_tuning.requests.post")
    @patch("sllm.cli.fine_tuning.read_config")
    def test_fine_tuning_with_model_only(self, mock_read_config, mock_post):
        # Mock read_config to return a sample fine-tuning configuration
        mock_read_config.return_value = {
            "model": "facebook/opt-125m",
            "ft_backend": "peft_lora",
            "backend_config": {
                "dataset_config": {
                    "dataset_source": "hf_hub",
                    "hf_dataset_name": "fka/awesome-chatgpt-prompts",
                    "tokenization_field": "prompt",
                    "split": "train[:10%]",
                    "data_files": "",
                    "extension_type": "",
                },
                "lora_config": {
                    "r": 4,
                    "lora_alpha": 1,
                    "target_modules": ["query_key_value"],
                    "lora_dropout": 0.05,
                    "bias": "lora_only",
                    "task_type": "CAUSAL_LM",
                },
                "training_config": {
                    "auto_find_batch_size": True,
                    "num_train_epochs": 2,
                    "learning_rate": 0.0001,
                    "use_cpu": False,
                },
            },
        }

        # Mock a successful POST response returning job ID
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"job_id": "job-abc123"}
        mock_post.return_value = mock_response

        args = Namespace(base_model="facebook/opt-125m", config=None)
        command = FineTuningCommand(args)
        result = command.run()

        mock_read_config.assert_called_once_with(command.config_path)
        mock_post.assert_called_once()
        self.assertEqual(
            mock_post.call_args[1]["json"]["model"], "facebook/opt-125m"
        )
        self.assertEqual(
            mock_response.json.return_value["job_id"],
            "job-abc123",
        )

    @patch("sllm.cli.fine_tuning.read_config")
    def test_validate_config_missing_key(self, mock_read_config):
        mock_read_config.return_value = {
            "ft_backend": "peft",
            "backend_config": {
                "dataset_config": {"dataset_source": "hf_hub"},
            },
        }
        args = Namespace(base_model="facebook/opt-125m", config=None)
        command = FineTuningCommand(args)

        with self.assertRaises(ValueError) as context:
            command.validate_config(mock_read_config.return_value)
        self.assertIn("Missing key", str(context.exception))

    @patch("sllm.cli.fine_tuning.requests.post")
    @patch("sllm.cli.fine_tuning.read_config")
    def test_fine_tuning_request_failure(self, mock_read_config, mock_post):
        mock_read_config.return_value = {
            "model": "facebook/opt-125m",
            "ft_backend": "peft",
            "backend_config": {
                "dataset_config": {
                    "dataset_source": "hf_hub",
                    "hf_dataset_name": "fka/awesome-chatgpt-prompts",
                    "tokenization_field": "prompt",
                    "split": "train[:10%]",
                },
                "lora_config": {"r": 4},
                "training_config": {"num_train_epochs": 2},
            },
        }

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        args = Namespace(base_model="facebook/opt-125m", config=None)
        command = FineTuningCommand(args)

        with self.assertLogs("sllm.cli.fine_tuning", level="ERROR") as log:
            result = command.fine_tuning(mock_read_config.return_value)
            self.assertIsNone(result)
            mock_post.assert_called_once()
            self.assertIn("Failed to do fine-tuning", log.output[0])


if __name__ == "__main__":
    unittest.main()
