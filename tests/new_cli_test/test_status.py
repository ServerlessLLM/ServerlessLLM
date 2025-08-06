# tests/new_cli_test/test_status.py

import unittest
from unittest import mock

import requests
from click.testing import CliRunner

from sllm.cli.clic import cli


class TestStatusCommand(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @mock.patch("sllm.cli._cli_utils.requests.get")
    def test_status_with_models(self, mock_get):
        mock_resp = mock_get.return_value
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"id": "a1"}, {"id": "b2"}]}

        result = self.runner.invoke(cli, ["status"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Model status retrieved successfully:", result.output)
        self.assertIn("- a1", result.output)
        self.assertIn("- b2", result.output)

    @mock.patch("sllm.cli._cli_utils.requests.get")
    def test_status_no_models(self, mock_get):
        mock_resp = mock_get.return_value
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": []}

        result = self.runner.invoke(cli, ["status"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No models currently deployed.", result.output)

    @mock.patch("sllm.cli._cli_utils.requests.get")
    def test_status_http_error(self, mock_get):
        mock_resp = mock_get.return_value
        mock_resp.status_code = 500
        mock_resp.text = "Internal Error"

        result = self.runner.invoke(cli, ["status"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "[❌ ERROR] Failed with status 500: Internal Error", result.output
        )

    @mock.patch(
        "sllm.cli._cli_utils.requests.get",
        side_effect=requests.exceptions.RequestException("ConnErr"),
    )
    def test_status_exception(self, mock_get):
        result = self.runner.invoke(cli, ["status"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "[EXCEPTION] Failed to query status: ConnErr", result.output
        )

    # NEW TESTS TO CATCH REAL CLI BUGS:

    @mock.patch("sllm.cli.clic.show_status")
    def test_status_command_calls_show_status(self, mock_show_status):
        """Test that the status command actually calls show_status function."""
        result = self.runner.invoke(cli, ["status"])

        # Verify the function was called
        mock_show_status.assert_called_once()
        # Verify it was called with no arguments
        self.assertEqual(
            len(mock_show_status.call_args[0]), 0
        )  # no positional args
        self.assertEqual(
            len(mock_show_status.call_args[1]), 0
        )  # no keyword args

    def test_status_command_accepts_no_arguments(self):
        """Test that status command doesn't accept any arguments."""
        # This should work
        with mock.patch("sllm.cli._cli_utils.show_status"):
            result = self.runner.invoke(cli, ["status"])
            self.assertEqual(result.exit_code, 0)

        # This should fail - status doesn't take arguments
        result = self.runner.invoke(cli, ["status", "extra-arg"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Got unexpected extra argument", result.output)

    def test_status_command_exists_and_has_correct_name(self):
        """Test that the status command is properly registered."""
        # This would catch if the command decorator was broken
        result = self.runner.invoke(cli, ["--help"])
        self.assertIn("status", result.output)

        # Test that the command has the right help text
        result = self.runner.invoke(cli, ["status", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Show all deployed models", result.output)

    @mock.patch("sllm.cli._cli_utils.requests.get")
    def test_status_handles_different_model_formats(self, mock_get):
        """Test that status can handle different model response formats."""
        # Test with models as strings instead of dicts
        mock_resp = mock_get.return_value
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": ["model1", "model2"]}

        result = self.runner.invoke(cli, ["status"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Model status retrieved successfully:", result.output)
        self.assertIn("- model1", result.output)
        self.assertIn("- model2", result.output)

    @mock.patch("sllm.cli._cli_utils.requests.get")
    def test_status_handles_missing_models_key(self, mock_get):
        """Test that status handles response without 'models' key gracefully."""
        mock_resp = mock_get.return_value
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"other_key": "value"}

        result = self.runner.invoke(cli, ["status"])
        self.assertEqual(result.exit_code, 0)
        # Should show no models when models key is missing
        self.assertIn("No models currently deployed.", result.output)


if __name__ == "__main__":
    unittest.main()
