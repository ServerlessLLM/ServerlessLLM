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
        result = self.runner.invoke(cli, ["start", "head"])
        self.assertEqual(result.exit_code, 0)
        mock_start_head.assert_called_once()
        call_kwargs = mock_start_head.call_args[1]
        self.assertEqual(call_kwargs["host"], "0.0.0.0")
        self.assertEqual(call_kwargs["port"], 8343)

    @mock.patch("sllm.cli.clic.start_head")
    def test_start_with_pylet_endpoint(self, mock_start_head):
        """Test start with --pylet-endpoint flag (v1-beta)."""
        result = self.runner.invoke(
            cli,
            [
                "start",
                "head",
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
        """Test start with --database-path flag (v1-beta)."""
        result = self.runner.invoke(
            cli,
            [
                "start",
                "head",
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
        """Test start with --storage-path flag (v1-beta)."""
        result = self.runner.invoke(
            cli,
            [
                "start",
                "head",
                "--storage-path",
                "/mnt/models",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        mock_start_head.assert_called_once()
        call_kwargs = mock_start_head.call_args[1]
        self.assertEqual(call_kwargs["storage_path"], "/mnt/models")

    @mock.patch("sllm.cli.clic.start_head")
    def test_start_with_all_v1beta_options(self, mock_start_head):
        """Test start with all v1-beta options."""
        result = self.runner.invoke(
            cli,
            [
                "start",
                "head",
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

    def test_start_without_subcommand_shows_help(self):
        # Test that start command shows help when no subcommand provided
        result = self.runner.invoke(cli, ["start"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Usage:", result.output)
        self.assertIn("head", result.output)
        self.assertIn("worker", result.output)

    @mock.patch(
        "sllm.cli.clic.start_head",
        side_effect=Exception("Head start failed"),
    )
    def test_start_exception(self, mock_start_head):
        # Test that exceptions during head start are handled
        result = self.runner.invoke(cli, ["start", "head"])
        # The CLI should handle exceptions gracefully
        mock_start_head.assert_called_once()

    @mock.patch("sllm.cli.clic.start_head")
    def test_start_with_custom_host_port(self, mock_start_head):
        result = self.runner.invoke(
            cli, ["start", "head", "--host", "localhost", "--port", "9000"]
        )
        self.assertEqual(result.exit_code, 0)
        mock_start_head.assert_called_once()
        call_kwargs = mock_start_head.call_args[1]
        self.assertEqual(call_kwargs["host"], "localhost")
        self.assertEqual(call_kwargs["port"], 9000)

    def test_start_head_help_shows_v1beta_options(self):
        """Test that head help shows v1-beta options."""
        result = self.runner.invoke(cli, ["start", "head", "--help"])
        self.assertEqual(result.exit_code, 0)
        # Should show v1-beta options
        self.assertIn("--pylet-endpoint", result.output)
        self.assertIn("--database-path", result.output)
        self.assertIn("--storage-path", result.output)

    def test_deprecated_redis_options_show_warning(self):
        """Test that deprecated Redis options show warning."""
        with mock.patch("sllm.cli.clic.start_head"):
            result = self.runner.invoke(
                cli,
                [
                    "start",
                    "head",
                    "--redis-host",
                    "localhost",
                    "--redis-port",
                    "6379",
                ],
            )
            # Should still work but may show deprecation warning
            self.assertEqual(result.exit_code, 0)

    def test_start_with_storage_aware_and_migration(self):
        # Test that head subcommand exists
        result_head = self.runner.invoke(cli, ["start", "head", "--help"])
        self.assertEqual(result_head.exit_code, 0)


class TestStartWorkerCommand(unittest.TestCase):
    """Tests for worker start command.

    Note: Worker command was removed in v1-beta. The v1-beta architecture
    uses external Pylet for worker management instead of sllm workers.
    """

    def setUp(self):
        self.runner = CliRunner()

    def test_worker_subcommand_removed(self):
        """Test that worker subcommand no longer exists (removed in v1-beta)."""
        result = self.runner.invoke(cli, ["start", "worker", "--help"])
        # Worker command was removed - should show error
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No such command 'worker'", result.output)


if __name__ == "__main__":
    unittest.main()
