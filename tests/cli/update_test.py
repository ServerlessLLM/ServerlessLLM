import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

from serverless_llm.cli.update import UpdateCommand


class TestUpdateCommand(unittest.TestCase):
    @patch("serverless_llm.cli.update.requests.post")
    @patch("serverless_llm.cli.update.read_config")
    @patch("serverless_llm.cli.update.validate_config")
    def test_update_with_config_file(
        self, mock_validate, mock_read_config, mock_post
    ):
        # Mock the configuration reading and validation
        mock_read_config.return_value = {
            "model": "facebook/opt-1.3b",
            "backend_config": {
                "pretrained_model_name_or_path": "facebook/opt-1.3b"
            },
        }
        mock_post.return_value.status_code = 200

        args = Namespace(model=None, config="/path/to/config.json")
        command = UpdateCommand(args)
        command.run()

        mock_read_config.assert_called_once_with("/path/to/config.json")
        mock_validate.assert_called_once()
        mock_post.assert_called_once_with(
            "http://localhost:8343/update",
            headers={"Content-Type": "application/json"},
            json={
                "model": "facebook/opt-1.3b",
                "backend_config": {
                    "pretrained_model_name_or_path": "facebook/opt-1.3b"
                },
            },
        )

    @patch("serverless_llm.cli.update.requests.post")
    @patch("serverless_llm.cli.update.read_config")
    def test_update_with_model_name(self, mock_read_config, mock_post):
        # Mock the default configuration reading
        mock_read_config.return_value = {
            "model": "",
            "backend_config": {"pretrained_model_name_or_path": ""},
        }
        mock_post.return_value.status_code = 200

        args = Namespace(model="facebook/opt-1.3b", config=None)
        command = UpdateCommand(args)
        command.run()

        mock_read_config.assert_called_once()
        mock_post.assert_called_once_with(
            "http://localhost:8343/update",
            headers={"Content-Type": "application/json"},
            json={
                "model": "facebook/opt-1.3b",
                "backend_config": {
                    "pretrained_model_name_or_path": "facebook/opt-1.3b"
                },
            },
        )

    @patch("serverless_llm.cli.update.requests.post")
    @patch("serverless_llm.cli.update.read_config")
    def test_update_model_failure(self, mock_read_config, mock_post):
        # Mock the default configuration reading
        mock_read_config.return_value = {
            "model": "",
            "backend_config": {"pretrained_model_name_or_path": ""},
        self.assertEqual(mock_post.return_value.status_code, 500)
        mock_post.return_value.status_code = 500

        args = Namespace(model="facebook/opt-1.3b", config=None)
        command = UpdateCommand(args)
        command.run()

        mock_read_config.assert_called_once()
        mock_post.assert_called_once()
        self.assertTrue(mock_post.return_value.status_code, 500)

    def test_update_missing_arguments(self):
        args = Namespace(model=None, config=None)
        command = UpdateCommand(args)

        with self.assertRaises(SystemExit):
            command.run()


if __name__ == "__main__":
    unittest.main()
