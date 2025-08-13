# tests/new_cli_test/test_start.py

import unittest
from unittest import mock

from click.testing import CliRunner

from sllm.cli.clic import cli


class TestStartCommand(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @mock.patch("sllm.cli._cli_utils.uvicorn.run")
    @mock.patch("sllm.cli._cli_utils.ray.get")
    @mock.patch("sllm.cli._cli_utils.ray.remote")
    @mock.patch("sllm.cli._cli_utils.create_app")
    @mock.patch("sllm.cli._cli_utils.ray.is_initialized", return_value=False)
    @mock.patch("sllm.cli._cli_utils.ray.init")
    def test_start_success(
        self,
        mock_ray_init,
        mock_ray_initialized,
        mock_create_app,
        mock_ray_remote,
        mock_ray_get,
        mock_uvicorn_run,
    ):
        # Mock the controller setup
        mock_controller_cls = mock.Mock()
        mock_controller = mock.Mock()
        mock_controller_cls.options.return_value.remote.return_value = (
            mock_controller
        )
        mock_ray_remote.return_value = mock_controller_cls

        # Mock the app creation
        mock_app = mock.Mock()
        mock_create_app.return_value = mock_app

        result = self.runner.invoke(cli, ["start"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("[ℹ] Initializing Ray...", result.output)
        self.assertIn("[ℹ] Creating FastAPI application...", result.output)
        self.assertIn("[ℹ] Starting SLLM controller...", result.output)
        self.assertIn(
            "[✅] SLLM controller started successfully", result.output
        )
        self.assertIn(
            "[🚀] Starting SLLM server on 0.0.0.0:8343...", result.output
        )

        # Verify Ray was initialized
        mock_ray_init.assert_called_once()

        # Verify app was created
        mock_create_app.assert_called_once()

        # Verify controller was started
        mock_ray_get.assert_called_once_with(mock_controller.start.remote())

        # Verify uvicorn was called
        mock_uvicorn_run.assert_called_once_with(
            mock_app, host="0.0.0.0", port=8343
        )

    @mock.patch("sllm.cli._cli_utils.uvicorn.run")
    @mock.patch("sllm.cli._cli_utils.ray.get")
    @mock.patch("sllm.cli._cli_utils.ray.remote")
    @mock.patch("sllm.cli._cli_utils.create_app")
    @mock.patch("sllm.cli._cli_utils.ray.is_initialized", return_value=True)
    def test_start_with_ray_already_initialized(
        self,
        mock_ray_initialized,
        mock_create_app,
        mock_ray_remote,
        mock_ray_get,
        mock_uvicorn_run,
    ):
        # Mock the controller setup
        mock_controller_cls = mock.Mock()
        mock_controller = mock.Mock()
        mock_controller_cls.options.return_value.remote.return_value = (
            mock_controller
        )
        mock_ray_remote.return_value = mock_controller_cls

        # Mock the app creation
        mock_app = mock.Mock()
        mock_create_app.return_value = mock_app

        result = self.runner.invoke(cli, ["start"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("[ℹ] Ray already initialized", result.output)
        self.assertIn("[ℹ] Creating FastAPI application...", result.output)
        self.assertIn("[ℹ] Starting SLLM controller...", result.output)
        self.assertIn(
            "[✅] SLLM controller started successfully", result.output
        )
        self.assertIn(
            "[🚀] Starting SLLM server on 0.0.0.0:8343...", result.output
        )

    @mock.patch(
        "sllm.cli._cli_utils.uvicorn.run", side_effect=KeyboardInterrupt()
    )
    @mock.patch("sllm.cli._cli_utils.ray.get")
    @mock.patch("sllm.cli._cli_utils.ray.remote")
    @mock.patch("sllm.cli._cli_utils.create_app")
    @mock.patch("sllm.cli._cli_utils.ray.is_initialized", return_value=False)
    @mock.patch("sllm.cli._cli_utils.ray.init")
    def test_start_keyboard_interrupt(
        self,
        mock_ray_init,
        mock_ray_initialized,
        mock_create_app,
        mock_ray_remote,
        mock_ray_get,
        mock_uvicorn_run,
    ):
        # Mock the controller setup
        mock_controller_cls = mock.Mock()
        mock_controller = mock.Mock()
        mock_controller_cls.options.return_value.remote.return_value = (
            mock_controller
        )
        mock_ray_remote.return_value = mock_controller_cls

        # Mock the app creation
        mock_app = mock.Mock()
        mock_create_app.return_value = mock_app

        result = self.runner.invoke(cli, ["start"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("[ℹ] Shutting down SLLM server...", result.output)
        self.assertIn("[✅] SLLM server shut down successfully", result.output)

        # Verify shutdown was called
        mock_ray_get.assert_called_with(mock_controller.shutdown.remote())

    @mock.patch(
        "sllm.cli._cli_utils.create_app",
        side_effect=Exception("App creation failed"),
    )
    @mock.patch("sllm.cli._cli_utils.ray.is_initialized", return_value=False)
    @mock.patch("sllm.cli._cli_utils.ray.init")
    def test_start_exception(
        self, mock_ray_init, mock_ray_initialized, mock_create_app
    ):
        result = self.runner.invoke(cli, ["start"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn(
            "[❌] Failed to start SLLM server: App creation failed",
            result.output,
        )

    @mock.patch("sllm.cli._cli_utils.uvicorn.run")
    @mock.patch("sllm.cli._cli_utils.ray.get")
    @mock.patch("sllm.cli._cli_utils.ray.remote")
    @mock.patch("sllm.cli._cli_utils.create_app")
    @mock.patch("sllm.cli._cli_utils.ray.is_initialized", return_value=False)
    @mock.patch("sllm.cli._cli_utils.ray.init")
    def test_start_with_custom_host_port(
        self,
        mock_ray_init,
        mock_ray_initialized,
        mock_create_app,
        mock_ray_remote,
        mock_ray_get,
        mock_uvicorn_run,
    ):
        # Mock the controller setup
        mock_controller_cls = mock.Mock()
        mock_controller = mock.Mock()
        mock_controller_cls.options.return_value.remote.return_value = (
            mock_controller
        )
        mock_ray_remote.return_value = mock_controller_cls

        # Mock the app creation
        mock_app = mock.Mock()
        mock_create_app.return_value = mock_app

        result = self.runner.invoke(
            cli, ["start", "--host", "localhost", "--port", "9000"]
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "[🚀] Starting SLLM server on localhost:9000...", result.output
        )

        # Verify uvicorn was called with custom host/port
        mock_uvicorn_run.assert_called_once_with(
            mock_app, host="localhost", port=9000
        )

    @mock.patch("sllm.cli._cli_utils.uvicorn.run")
    @mock.patch("sllm.cli._cli_utils.ray.get")
    @mock.patch("sllm.cli._cli_utils.ray.remote")
    @mock.patch("sllm.cli._cli_utils.create_app")
    @mock.patch("sllm.cli._cli_utils.ray.is_initialized", return_value=False)
    @mock.patch("sllm.cli._cli_utils.ray.init")
    def test_start_with_storage_aware_and_migration(
        self,
        mock_ray_init,
        mock_ray_initialized,
        mock_create_app,
        mock_ray_remote,
        mock_ray_get,
        mock_uvicorn_run,
    ):
        # Mock the controller setup
        mock_controller_cls = mock.Mock()
        mock_controller = mock.Mock()
        mock_controller_cls.options.return_value.remote.return_value = (
            mock_controller
        )
        mock_ray_remote.return_value = mock_controller_cls

        # Mock the app creation
        mock_app = mock.Mock()
        mock_create_app.return_value = mock_app

        result = self.runner.invoke(
            cli, ["start", "--enable-storage-aware", "--enable-migration"]
        )
        self.assertEqual(result.exit_code, 0)

        # Verify controller was created with the right options
        expected_config = {
            "enable_storage_aware": True,
            "enable_migration": True,
        }
        mock_controller_cls.options.return_value.remote.assert_called_once_with(
            expected_config
        )


if __name__ == "__main__":
    unittest.main()
