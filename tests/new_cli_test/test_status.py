# tests/new_cli_test/test_status.py

import unittest
from unittest import mock
from click.testing import CliRunner
import requests

from sllm.clic import cli


class TestStatusCommand(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @mock.patch("sllm._cli_utils.requests.get")
    def test_status_with_models(self, mock_get):
        mock_resp = mock_get.return_value
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"id": "a1"}, {"id": "b2"}]
        }

        result = self.runner.invoke(cli, ["status"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Model status retrieved", result.output)
        self.assertIn("- a1", result.output)
        self.assertIn("- b2", result.output)

    @mock.patch("sllm._cli_utils.requests.get")
    def test_status_no_models(self, mock_get):
        mock_resp = mock_get.return_value
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": []}

        result = self.runner.invoke(cli, ["status"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No models currently deployed.", result.output)

    @mock.patch("sllm._cli_utils.requests.get")
    def test_status_http_error(self, mock_get):
        mock_resp = mock_get.return_value
        mock_resp.status_code = 500
        mock_resp.text = "Internal Error"

        result = self.runner.invoke(cli, ["status"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Failed with status 500", result.output)

    @mock.patch(
        "sllm._cli_utils.requests.get",
        side_effect=requests.exceptions.RequestException("ConnErr")
    )
    def test_status_exception(self, mock_get):
  
        result = self.runner.invoke(cli, ["status"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("[EXCEPTION] Failed to query status: ConnErr", result.output)


if __name__ == "__main__":
    unittest.main()
