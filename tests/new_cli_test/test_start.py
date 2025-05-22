# tests/new_cli_test/test_start.py

import unittest
from unittest import mock

from click.testing import CliRunner

from sllm.clic import cli


class TestStartCommand(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @mock.patch("sllm._cli_utils.subprocess.run")
    @mock.patch("sllm._cli_utils.os.path.exists", return_value=True)
    def test_start_success(self, mock_exists, mock_run):
        result = self.runner.invoke(cli, ["start"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("SLLM server started successfully.", result.output)

    @mock.patch("sllm._cli_utils.os.path.exists", return_value=False)
    def test_start_missing_compose(self, mock_exists):
        result = self.runner.invoke(cli, ["start"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Cannot find docker-compose.yml", result.output)

    @mock.patch(
        "sllm._cli_utils.subprocess.run", side_effect=Exception("docker error")
    )
    @mock.patch("sllm._cli_utils.os.path.exists", return_value=True)
    def test_start_subprocess_exception(self, mock_exists, mock_run):
        result = self.runner.invoke(cli, ["start"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Failed to start services: docker error", result.output)


if __name__ == "__main__":
    unittest.main()
