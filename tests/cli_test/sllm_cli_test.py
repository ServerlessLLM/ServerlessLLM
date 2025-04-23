import sys
import unittest
from unittest.mock import patch

from sllm.cli.sllm_cli import main


class TestSllmCLI(unittest.TestCase):
    @patch("sllm.cli.deploy.DeployCommand")
    def test_deploy_command(self, mock_deploy_command):
        # Simulate command-line input
        test_args = ["sllm-cli", "deploy", "--model", "facebook/opt-1.3b"]
        with patch.object(sys, "argv", test_args):
            main()

        # Check that DeployCommand was called with the correct arguments
        mock_deploy_command.assert_called_once()
        self.assertEqual(
            mock_deploy_command.call_args[0][0].model, "facebook/opt-1.3b"
        )

    @patch("sllm.cli.generate.GenerateCommand")
    def test_generate_command(self, mock_generate_command):
        # Simulate command-line input
        test_args = ["sllm-cli", "generate", "input.json"]
        with patch.object(sys, "argv", test_args):
            main()

        # Check that GenerateCommand was called with the correct arguments
        mock_generate_command.assert_called_once()
        self.assertEqual(
            mock_generate_command.call_args[0][0].input_path, "input.json"
        )

    @patch("sllm.cli.fine_tuning.FineTuningCommand")
    def test_fine_tuning_command(self, mock_fine_tuning_command):
        # Simulate command-line input
        test_args = [
            "sllm-cli",
            "fine-tuning",
            "--base-model",
            "bigscience/bloomz-560m",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        # Check that GenerateCommand was called with the correct arguments
        mock_fine_tuning_command.assert_called_once()
        self.assertEqual(
            mock_fine_tuning_command.call_args[0][0].base_model,
            "bigscience/bloomz-560m",
        )

    @patch("sllm.cli.replay.ReplayCommand")
    def test_replay_command(self, mock_replay_command):
        # Simulate command-line input
        test_args = [
            "sllm-cli",
            "replay",
            "--workload",
            "workload.json",
            "--dataset",
            "dataset.json",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        # Check that ReplayCommand was called with the correct arguments
        mock_replay_command.assert_called_once()
        self.assertEqual(
            mock_replay_command.call_args[0][0].workload, "workload.json"
        )
        self.assertEqual(
            mock_replay_command.call_args[0][0].dataset, "dataset.json"
        )

    @patch("sllm.cli.delete.DeleteCommand")
    def test_delete_command(self, mock_delete_command):
        # Simulate command-line input
        test_args = [
            "sllm-cli",
            "delete",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        # Check that DeleteCommand was called with the correct arguments
        mock_delete_command.assert_called_once()
        self.assertEqual(
            mock_delete_command.call_args[0][0].models,
            ["facebook/opt-1.3b", "facebook/opt-2.7b"],
        )

    @patch("sllm.cli.update.UpdateCommand")
    def test_update_command(self, mock_update_command):
        # Simulate command-line input
        test_args = ["sllm-cli", "update", "--model", "facebook/opt-1.3b"]
        with patch.object(sys, "argv", test_args):
            main()

        # Check that UpdateCommand was called with the correct arguments
        mock_update_command.assert_called_once()
        self.assertEqual(
            mock_update_command.call_args[0][0].model, "facebook/opt-1.3b"
        )

    @patch("argparse.ArgumentParser.print_help")
    def test_no_command(self, mock_print_help):
        with patch("sys.argv", ["sllm-cli"]), self.assertRaises(SystemExit):
            main()

        # Check that the help message was printed
        mock_print_help.assert_called_once()


if __name__ == "__main__":
    unittest.main()
