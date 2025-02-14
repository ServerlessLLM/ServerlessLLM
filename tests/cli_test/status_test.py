import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

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

    @patch("sllm.cli.status.StatusCommand.query_status")
    @patch("sllm.cli.status.logger.info")
    def test_run_success(self, mock_logger_info, mock_query_status):
        """Test the run method when query_status is successful."""
        # Mock query_status to return a valid response
        mock_query_status.return_value = {"data": "mocked_model_status"}

        # Initialize StatusCommand with dummy args
        args = Namespace()
        command = StatusCommand(args)

        # Execute run
        command.run()

        # Debug: Print what was actually logged
        print("Logger info calls:", mock_logger_info.call_args_list)

        # Assertions
        mock_query_status.assert_called_once()
        mock_logger_info.assert_called_once()

    @patch("sllm.cli.status.StatusCommand.query_status")
    @patch("sllm.cli.status.logger.error")
    def test_run_failure(self, mock_logger_error, mock_query_status):
        """Test the run method when query_status fails."""
        # Mock query_status to return None
        mock_query_status.return_value = None

        # Initialize StatusCommand with dummy args
        args = Namespace()
        command = StatusCommand(args)

        # Execute run
        command.run()

        # Assertions
        mock_query_status.assert_called_once()
        mock_logger_error.assert_called_with("Failed to fetch model status.")

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


if __name__ == "__main__":
    unittest.main()
