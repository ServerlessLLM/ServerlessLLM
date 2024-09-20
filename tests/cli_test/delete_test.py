import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

from serverless_llm.cli.delete import DeleteCommand


class TestDeleteCommand(unittest.TestCase):
    @patch("serverless_llm.cli.delete.requests.post")
    def test_delete_single_model_success(self, mock_post):
        # Mock the response of the requests.post
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        args = Namespace(models=["facebook/opt-1.3b"])
        command = DeleteCommand(args)
        command.run()

        mock_post.assert_called_once_with(
            "http://0.0.0.0:8343/delete/",
            headers={"Content-Type": "application/json"},
            json={"model": "facebook/opt-1.3b"},
        )
        self.assertEqual(mock_post.return_value.status_code, 200)

    @patch("serverless_llm.cli.delete.requests.post")
    def test_delete_multiple_models_success(self, mock_post):
        # Mock the response of the requests.post
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        args = Namespace(models=["facebook/opt-1.3b", "facebook/opt-2.7b"])
        command = DeleteCommand(args)
        command.run()

        self.assertEqual(mock_post.call_count, 2)

        expected_calls = [
            (
                "http://0.0.0.0:8343/delete/",
                {"Content-Type": "application/json"},
                {"model": "facebook/opt-1.3b"},
            ),
            (
                "http://0.0.0.0:8343/delete/",
                {"Content-Type": "application/json"},
                {"model": "facebook/opt-2.7b"},
            ),
        ]
        for call, expected in zip(mock_post.call_args_list, expected_calls):
            args, kwargs = call
            self.assertEqual(args[0], expected[0])
            self.assertEqual(kwargs["headers"], expected[1])
            self.assertEqual(kwargs["json"], expected[2])

    @patch("serverless_llm.cli.delete.requests.post")
    def test_delete_model_failure(self, mock_post):
        # Mock the response of the requests.post
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        args = Namespace(models=["facebook/opt-1.3b"])
        command = DeleteCommand(args)
        command.run()

        mock_post.assert_called_once_with(
            "http://0.0.0.0:8343/delete/",
            headers={"Content-Type": "application/json"},
            json={"model": "facebook/opt-1.3b"},
        )
        self.assertEqual(mock_post.return_value.status_code, 500)


if __name__ == "__main__":
    unittest.main()
