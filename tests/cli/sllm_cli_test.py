import sys
import unittest
from unittest.mock import MagicMock, patch

from serverless_llm.cli.sllm_cli import main


class TestSllmCLI(unittest.TestCase):
    @patch("serverless_llm.cli.deploy.DeployCommand.run")
    def test_deploy_command(self, mock_deploy_run):
        # Simulate command-line input
        test_args = ["sllm-cli", "deploy", "--model", "facebook/opt-1.3b"]
        with patch.object(sys, "argv", test_args):
            main()

        # Check that DeployCommand.run was called
        mock_deploy_run.assert_called_once()

    @patch("serverless_llm.cli.generate.GenerateCommand.run")
    def test_generate_command(self, mock_generate_run):
        # Simulate command-line input
        test_args = ["sllm-cli", "generate", "input.json"]
        with patch.object(sys, "argv", test_args):
            main()

        # Check that GenerateCommand.run was called
        mock_generate_run.assert_called_once()

    @patch("serverless_llm.cli.replay.ReplayCommand.run")
    def test_replay_command(self, mock_replay_run):
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

        # Check that ReplayCommand.run was called
        mock_replay_run.assert_called_once()

    @patch("serverless_llm.cli.delete.DeleteCommand.run")
    def test_delete_command(self, mock_delete_run):
        # Simulate command-line input
        test_args = [
            "sllm-cli",
            "delete",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
        ]
        with patch.object(sys, "argv", test_args):
            main()

        # Check that DeleteCommand.run was called
        mock_delete_run.assert_called_once()

    @patch("serverless_llm.cli.update.UpdateCommand.run")
    def test_update_command(self, mock_update_run):
        # Simulate command-line input
        test_args = ["sllm-cli", "update", "--model", "facebook/opt-1.3b"]
        with patch.object(sys, "argv", test_args):
            main()

        # Check that UpdateCommand.run was called
        mock_update_run.assert_called_once()

    @patch("argparse.ArgumentParser.print_help")
    def test_no_command(self, mock_print_help):
        with patch("sys.argv", ["sllm-cli"]):
            with self.assertRaises(SystemExit):
                main()

        # Check that the help message was printed
        mock_print_help.assert_called_once()


if __name__ == "__main__":
    unittest.main()
