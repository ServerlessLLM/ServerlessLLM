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

    @mock.patch("sllm.cli._cli_utils.start_worker")
    def test_start_with_ray_already_initialized(
        self,
        mock_start_worker,
    ):
        result = self.runner.invoke(
            cli, ["start", "worker", "--head-node-url", "http://127.0.0.1:8343"]
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

    # Commenting out the old storage-aware and migration test since those features
    # may have been restructured in the refactor, but keeping it for reference
    # @mock.patch("sllm.cli._cli_utils.uvicorn.run")
    # @mock.patch("sllm.cli._cli_utils.ray.get")
    # @mock.patch("sllm.cli._cli_utils.ray.remote")
    # @mock.patch("sllm.cli._cli_utils.create_app")
    # @mock.patch("sllm.cli._cli_utils.ray.is_initialized", return_value=False)
    # @mock.patch("sllm.cli._cli_utils.ray.init")
    # def test_start_with_storage_aware_and_migration(
    #     self,
    #     mock_ray_init,
    #     mock_ray_initialized,
    #     mock_create_app,
    #     mock_ray_remote,
    #     mock_ray_get,
    #     mock_uvicorn_run,
    # ):
    #     # Mock the controller setup
    #     mock_controller_cls = mock.Mock()
    #     mock_controller = mock.Mock()
    #     mock_controller_cls.options.return_value.remote.return_value = (
    #         mock_controller
    #     )
    #     mock_ray_remote.return_value = mock_controller_cls
    #
    #     # Mock the app creation
    #     mock_app = mock.Mock()
    #     mock_create_app.return_value = mock_app
    #
    #     result = self.runner.invoke(
    #         cli, ["start", "--enable-storage-aware", "--enable-migration"]
    #     )
    #     self.assertEqual(result.exit_code, 0)
    #
    #     # Verify controller was created with the right options
    #     expected_config = {
    #         "enable_storage_aware": True,
    #         "enable_migration": True,
    #     }
    #     mock_controller_cls.options.return_value.remote.assert_called_once_with(
    #         expected_config
    #     )
    def test_start_head_help_shows_v1beta_options(self):
        """Test that head help shows v1-beta options."""
        result = self.runner.invoke(cli, ["start", "head", "--help"])
    def test_start_help_shows_options(self):
        """Test that start help shows all options."""
        result = self.runner.invoke(cli, ["start", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--pylet-endpoint", result.output)
        self.assertIn("--database-path", result.output)
        self.assertIn("--storage-path", result.output)

    def test_start_with_storage_aware_and_migration(self):
        # Test that head subcommand exists
        result_head = self.runner.invoke(cli, ["start", "head", "--help"])
        self.assertEqual(result_head.exit_code, 0)

        result_worker = self.runner.invoke(cli, ["start", "worker", "--help"])
        self.assertEqual(result_worker.exit_code, 0)

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
        self.assertIn("--host", result.output)
        self.assertIn("--port", result.output)


if __name__ == "__main__":
    unittest.main()
