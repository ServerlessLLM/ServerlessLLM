# tests/new_cli_test/test_observability.py
"""
Tests for CLI observability improvements.
These tests define the expected behavior - write code to make them pass.
"""

import unittest
from unittest import mock

from click.testing import CliRunner

from sllm.cli.clic import cli

try:
    import pylet.client

    PYLET_AVAILABLE = True
except ImportError:
    PYLET_AVAILABLE = False


class TestDeployOutput(unittest.TestCase):
    """deploy should print the deployment ID."""

    def setUp(self):
        self.runner = CliRunner()

    @mock.patch("sllm.cli._cli_utils.requests.post")
    @mock.patch("sllm.cli._cli_utils.read_config")
    @mock.patch("sllm.cli._cli_utils.os.path.exists", return_value=True)
    def test_deploy_prints_deployment_id(
        self, mock_exists, mock_read, mock_post
    ):
        """Deploy should print the deployment_id from response."""
        mock_read.return_value = {"model": "", "backend": "vllm"}
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "deployment_id": "facebook/opt-1.3b:vllm"
        }

        result = self.runner.invoke(
            cli, ["deploy", "--model", "facebook/opt-1.3b"]
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("facebook/opt-1.3b:vllm", result.output)

    @mock.patch("sllm.cli._cli_utils.requests.post")
    @mock.patch("sllm.cli._cli_utils.read_config")
    @mock.patch("sllm.cli._cli_utils.os.path.exists", return_value=True)
    def test_deploy_fallback_when_no_deployment_id(
        self, mock_exists, mock_read, mock_post
    ):
        """Deploy should construct ID from model:backend if not in response."""
        mock_read.return_value = {"model": "", "backend": "vllm"}
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {}  # No deployment_id

        result = self.runner.invoke(
            cli,
            ["deploy", "--model", "facebook/opt-1.3b", "--backend", "sglang"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("facebook/opt-1.3b:sglang", result.output)


class TestStatusTable(unittest.TestCase):
    """status should show a table, not a bullet list."""

    def setUp(self):
        self.runner = CliRunner()

    @mock.patch("sllm.cli._cli_utils.requests.get")
    def test_status_shows_table_header(self, mock_get):
        """Status should show table columns."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "deployments": [
                {
                    "id": "meta-llama/Llama-3.1-8B:vllm",
                    "status": "active",
                    "desired_replicas": 2,
                    "ready_replicas": 1,
                }
            ],
            "nodes": [],
        }

        result = self.runner.invoke(cli, ["status"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("DEPLOYMENT", result.output)
        self.assertIn("STATUS", result.output)
        self.assertIn("REPLICAS", result.output)

    @mock.patch("sllm.cli._cli_utils.requests.get")
    def test_status_shows_deployment_row(self, mock_get):
        """Status should show deployment info in table row."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "deployments": [
                {
                    "id": "meta-llama/Llama-3.1-8B:vllm",
                    "status": "active",
                    "desired_replicas": 2,
                    "ready_replicas": 1,
                }
            ],
            "nodes": [],
        }

        result = self.runner.invoke(cli, ["status"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("meta-llama/Llama-3.1-8B:vllm", result.output)
        self.assertIn("active", result.output)
        self.assertIn("1/2", result.output)

    @mock.patch("sllm.cli._cli_utils.requests.get")
    def test_status_uses_status_endpoint(self, mock_get):
        """Status should call /status, not /v1/models."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "deployments": [],
            "nodes": [],
        }

        self.runner.invoke(cli, ["status"])

        call_url = mock_get.call_args[0][0]
        self.assertIn("/status", call_url)
        self.assertNotIn("/v1/models", call_url)

    @mock.patch("sllm.cli._cli_utils.requests.get")
    def test_status_empty_shows_message(self, mock_get):
        """Status with no deployments should say so."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "deployments": [],
            "nodes": [],
        }

        result = self.runner.invoke(cli, ["status"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("No deployments", result.output)


class TestStatusDetail(unittest.TestCase):
    """status <deployment_id> should show details."""

    def setUp(self):
        self.runner = CliRunner()

    @mock.patch("sllm.cli._cli_utils.requests.get")
    def test_status_with_id_shows_detail(self, mock_get):
        """status <id> should show detailed view."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "deployments": [
                {
                    "id": "meta-llama/Llama-3.1-8B:vllm",
                    "status": "active",
                    "desired_replicas": 2,
                    "ready_replicas": 1,
                    "instances": [
                        {
                            "id": "abc123",
                            "node": "worker-1",
                            "endpoint": "10.0.0.5:8000",
                            "status": "running",
                        }
                    ],
                }
            ],
            "nodes": [],
        }

        result = self.runner.invoke(
            cli, ["status", "meta-llama/Llama-3.1-8B:vllm"]
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("meta-llama/Llama-3.1-8B:vllm", result.output)
        self.assertIn("abc123", result.output)
        self.assertIn("worker-1", result.output)
        self.assertIn("10.0.0.5:8000", result.output)

    @mock.patch("sllm.cli._cli_utils.requests.get")
    def test_status_with_invalid_id_fails(self, mock_get):
        """status <id> should fail if deployment not found."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "deployments": [],
            "nodes": [],
        }

        result = self.runner.invoke(cli, ["status", "nonexistent:vllm"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("not found", result.output.lower())


class TestStatusNodes(unittest.TestCase):
    """status --nodes should show cluster nodes."""

    def setUp(self):
        self.runner = CliRunner()

    @mock.patch("sllm.cli._cli_utils.requests.get")
    def test_status_nodes_shows_table(self, mock_get):
        """--nodes should show node table."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "deployments": [],
            "nodes": [
                {
                    "name": "worker-1",
                    "status": "online",
                    "total_gpus": 8,
                    "available_gpus": 6,
                }
            ],
        }

        result = self.runner.invoke(cli, ["status", "--nodes"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("NODE", result.output)
        self.assertIn("STATUS", result.output)
        self.assertIn("GPUS", result.output)
        self.assertIn("worker-1", result.output)
        self.assertIn("online", result.output)
        self.assertIn("6/8", result.output)

    @mock.patch("sllm.cli._cli_utils.requests.get")
    def test_status_nodes_empty(self, mock_get):
        """--nodes with no nodes should say so."""
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "deployments": [],
            "nodes": [],
        }

        result = self.runner.invoke(cli, ["status", "--nodes"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("No nodes", result.output)


class TestLogsCommand(unittest.TestCase):
    """logs command uses pylet Python API."""

    def setUp(self):
        self.runner = CliRunner()

    def test_logs_command_exists(self):
        """logs command should be registered."""
        result = self.runner.invoke(cli, ["--help"])
        self.assertIn("logs", result.output)

    @unittest.skipIf(not PYLET_AVAILABLE, "pylet not installed")
    def test_logs_fetches_from_pylet(self):
        """logs should fetch logs via PyletClient."""
        mock_client = mock.AsyncMock()
        mock_client.get_logs.return_value = {"data": b"test log output"}
        mock_client.client.aclose = mock.AsyncMock()

        with mock.patch("pylet.client.PyletClient", return_value=mock_client):
            result = self.runner.invoke(cli, ["logs", "instance123"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("test log output", result.output)
        mock_client.get_logs.assert_called()

    def test_logs_requires_instance_id(self):
        """logs without instance_id should fail."""
        result = self.runner.invoke(cli, ["logs"])
        self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
