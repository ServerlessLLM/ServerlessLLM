import sys
import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

# import os
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from sllm.cli.status import StatusCommand


class TestStatusCommand(unittest.TestCase):
    @patch("sllm.cli.status.requests.get")
    def test_query_status_success(self, mock_get):
        # Mock a successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "mocked_model_status"}
        mock_get.return_value = mock_response

        args = Namespace()
        command = StatusCommand(args)
        # execute query_status method
        status = command.query_status()
        mock_get.assert_called_once_with(
            "http://127.0.0.1:8343/v1/models",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, {"data": "mocked_model_status"})

    @patch("sllm.cli.status.requests.get")
    def test_query_status_failure(self, mock_get):
        # Mock a failed response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response

        # Initialize StatusCommand with dummy args
        args = Namespace()
        command = StatusCommand(args)

        # Execute query_status
        status = command.query_status()

        # Assertions
        mock_get.assert_called_once_with(
            "http://127.0.0.1:8343/v1/models",
            headers={"Content-Type": "application/json"},
        )
        self.assertIsNone(status)

    @patch("sllm.cli.status.requests.get")
    def test_query_status_invalid_json(self, mock_get):
        """Test query_status when the API responds with non-JSON data."""
        # Mock invalid JSON response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text = "Not a valid JSON"
        mock_get.return_value = mock_response

        # Initialize StatusCommand with dummy args
        args = Namespace()
        command = StatusCommand(args)

        # Execute query_status
        status = command.query_status()

        # Assertions
        mock_get.assert_called_once_with(
            "http://127.0.0.1:8343/v1/models",
            headers={"Content-Type": "application/json"},
        )
        self.assertIsNone(status)

    @patch("sllm.cli.status.requests.get")
    def test_query_status_custom_url(self, mock_get):
        """Test query_status with a custom server URL."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "mocked_model_status"}
        mock_get.return_value = mock_response

        # Set custom environment variable for server URL
        custom_url = "http://custom-server-url:1234/"
        with patch.dict("os.environ", {"LLM_SERVER_URL": custom_url}):
            args = Namespace()
            command = StatusCommand(args)

            # Execute query_status
            status = command.query_status()

            # Assertions
            mock_get.assert_called_once_with(
                custom_url + "v1/models",
                headers={"Content-Type": "application/json"},
            )
            self.assertEqual(status, {"data": "mocked_model_status"})

    @patch("builtins.print")
    @patch.object(
        StatusCommand,
        "query_status",
        return_value={
            "object": "list",
            "data": [
                {
                    "id": "facebook/opt-1.3b",
                    "object": "model",
                    "created": 1738960470,
                    # etc. – the rest of the dict...
                }
            ],
        },
    )
    def test_run_success(self, mock_query_status, mock_print):
        """
        Test that StatusCommand.run() logs a success if query_status returns a non-None result.
        """
        args = Namespace()
        cmd = StatusCommand(args)
        cmd.run()
        # Check that we actually called query_status
        mock_query_status.assert_called_once()

        # Extract all 'print(...)' calls.
        printed_lines = [args[0] for args, kwargs in mock_print.call_args_list]

        # You probably have only one print call in this scenario, but let's be safe and check them all:
        # We expect something like: "Model status: {'object': 'list', 'data': [...]}"
        found_line = any("Model status: {" in line for line in printed_lines)
        self.assertTrue(
            found_line,
            f"Expected a printed line containing 'Model status: {{', got: {printed_lines}",
        )

    @patch("sllm.cli.status.logger.error")
    @patch.object(StatusCommand, "query_status", return_value=None)
    def test_run_failure(self, mock_query_status, mock_logger_error):
        """
        Test that StatusCommand.run() logs an error if query_status returns None.
        """
        args = Namespace()
        cmd = StatusCommand(args)
        cmd.run()

        mock_query_status.assert_called_once()
        mock_logger_error.assert_called_once_with(
            "Failed to fetch model status."
        )


if __name__ == "__main__":
    unittest.main()
