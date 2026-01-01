# tests/new_cli_test/test_start.py

import unittest
from unittest import mock

from click.testing import CliRunner

from sllm.cli.clic import cli


class TestStartCommand(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @mock.patch("sllm.cli.clic.start_head")
    def test_start_success(
        self,
        mock_start_head,
    ):
        result = self.runner.invoke(cli, ["start"])
        self.assertEqual(result.exit_code, 0)
        mock_start_head.assert_called_once()
        call_kwargs = mock_start_head.call_args[1]
        self.assertEqual(call_kwargs["host"], "0.0.0.0")
        self.assertEqual(call_kwargs["port"], 8343)

    @mock.patch("sllm.cli.clic.start_head")
    def test_start_with_pylet_endpoint(self, mock_start_head):
        """Test start with --pylet-endpoint flag."""
        result = self.runner.invoke(
            cli,
            [
                "start",
                "--pylet-endpoint",
                "http://pylet:8000",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        mock_start_head.assert_called_once()
        call_kwargs = mock_start_head.call_args[1]
        self.assertEqual(call_kwargs["pylet_endpoint"], "http://pylet:8000")

    @mock.patch("sllm.cli.clic.start_head")
    def test_start_with_database_path(self, mock_start_head):
        """Test start with --database-path flag."""
        result = self.runner.invoke(
            cli,
            [
                "start",
                "--database-path",
                "/custom/path/state.db",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        mock_start_head.assert_called_once()
        call_kwargs = mock_start_head.call_args[1]
        self.assertEqual(call_kwargs["database_path"], "/custom/path/state.db")

    @mock.patch("sllm.cli.clic.start_head")
    def test_start_with_storage_path(self, mock_start_head):
        """Test start with --storage-path flag."""
        result = self.runner.invoke(
            cli,
            [
                "start",
                "--storage-path",
                "/mnt/models",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        mock_start_head.assert_called_once()
        call_kwargs = mock_start_head.call_args[1]
        self.assertEqual(call_kwargs["storage_path"], "/mnt/models")

    @mock.patch("sllm.cli.clic.start_head")
    def test_start_with_all_options(self, mock_start_head):
        """Test start with all options."""
        result = self.runner.invoke(
            cli,
            [
                "start",
                "--host",
                "localhost",
                "--port",
                "9000",
                "--pylet-endpoint",
                "http://pylet:8000",
                "--database-path",
                "/var/lib/sllm/state.db",
                "--storage-path",
                "/models",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        mock_start_head.assert_called_once()
        call_kwargs = mock_start_head.call_args[1]
        self.assertEqual(call_kwargs["host"], "localhost")
        self.assertEqual(call_kwargs["port"], 9000)
        self.assertEqual(call_kwargs["pylet_endpoint"], "http://pylet:8000")
        self.assertEqual(call_kwargs["database_path"], "/var/lib/sllm/state.db")
        self.assertEqual(call_kwargs["storage_path"], "/models")

    @mock.patch(
        "sllm.cli.clic.start_head",
        side_effect=Exception("Head start failed"),
    )
    def test_start_exception(self, mock_start_head):
        # Test that exceptions during head start are handled
        result = self.runner.invoke(cli, ["start"])
        # The CLI should handle exceptions gracefully
        mock_start_head.assert_called_once()

    @mock.patch("sllm.cli.clic.start_head")
    def test_start_with_custom_host_port(self, mock_start_head):
        result = self.runner.invoke(
            cli, ["start", "--host", "localhost", "--port", "9000"]
        )
        self.assertEqual(result.exit_code, 0)
        mock_start_head.assert_called_once()
        call_kwargs = mock_start_head.call_args[1]
        self.assertEqual(call_kwargs["host"], "localhost")
        self.assertEqual(call_kwargs["port"], 9000)

    def test_start_help_shows_options(self):
        """Test that start help shows all options."""
        result = self.runner.invoke(cli, ["start", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--pylet-endpoint", result.output)
        self.assertIn("--database-path", result.output)
        self.assertIn("--storage-path", result.output)
        self.assertIn("--host", result.output)
        self.assertIn("--port", result.output)


if __name__ == "__main__":
    unittest.main()
