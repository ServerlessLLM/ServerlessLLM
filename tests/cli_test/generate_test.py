import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

from serverless_llm.cli.generate import GenerateCommand


class TestGenerateCommand(unittest.TestCase):
    @patch("serverless_llm.cli.generate.requests.post")
    @patch("serverless_llm.cli.generate.read_config")
    def test_generate_single_thread_success(self, mock_read_config, mock_post):
        # Mock the configuration reading and the POST request
        mock_read_config.return_value = {
            "model": "facebook/opt-1.3b",
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.3,
            "max_tokens": 50,
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Hi!"}}]
        }

        args = Namespace(input_path="/path/to/input.json", threads=1)
        command = GenerateCommand(args)
        command.run()
        self.assertEqual(mock_post.return_value.status_code, 200)

        mock_read_config.assert_called_once_with("/path/to/input.json")
        mock_post.assert_called_once_with(
            "http://0.0.0.0:8343/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "facebook/opt-1.3b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "temperature": 0.3,
                "max_tokens": 50,
            },
        )

    @patch("serverless_llm.cli.generate.requests.post")
    @patch("serverless_llm.cli.generate.read_config")
    def test_generate_single_thread_failure(self, mock_read_config, mock_post):
        # Mock the configuration reading and the POST request with failure
        mock_read_config.return_value = {
            "model": "facebook/opt-1.3b",
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.3,
            "max_tokens": 50,
        }
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal Server Error"

        args = Namespace(input_path="/path/to/input.json", threads=1)
        command = GenerateCommand(args)
        command.run()

        mock_read_config.assert_called_once_with("/path/to/input.json")
        mock_post.assert_called_once()
        self.assertEqual(mock_post.return_value.status_code, 500)

    @patch("serverless_llm.cli.generate.requests.post")
    @patch("serverless_llm.cli.generate.read_config")
    def test_generate_multi_thread_success(self, mock_read_config, mock_post):
        # Mock the configuration reading and the POST request
        mock_read_config.return_value = {
            "model": "facebook/opt-1.3b",
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.3,
            "max_tokens": 50,
        }
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Hi!"}}]
        }

        args = Namespace(input_path="/path/to/input.json", threads=2)
        command = GenerateCommand(args)
        command.run()

        mock_read_config.assert_called_once_with("/path/to/input.json")
        self.assertEqual(mock_post.call_count, 2)

    @patch("serverless_llm.cli.generate.requests.post")
    @patch("serverless_llm.cli.generate.read_config")
    def test_generate_multi_thread_failure(self, mock_read_config, mock_post):
        # Mock the configuration reading and the POST request with failure
        mock_read_config.return_value = {
            "model": "facebook/opt-1.3b",
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.3,
            "max_tokens": 50,
        }
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal Server Error"

        args = Namespace(input_path="/path/to/input.json", threads=2)
        command = GenerateCommand(args)
        command.run()

        mock_read_config.assert_called_once_with("/path/to/input.json")
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(mock_post.return_value.status_code, 500)


if __name__ == "__main__":
    unittest.main()
