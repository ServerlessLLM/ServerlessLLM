# tests/new_cli_test/test_delete.py

import unittest
from unittest import mock
from click.testing import CliRunner

from sllm.clic import cli


class TestDeleteCommand(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @mock.patch("sllm._cli_utils.requests.post")
    def test_delete_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = "OK"

        result = self.runner.invoke(cli, ["delete", "foo", "bar"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Model 'foo' deleted successfully.", result.output)
        self.assertIn("Model 'bar' deleted successfully.", result.output)

        self.assertEqual(mock_post.call_count, 2)

        expected_calls = [
            mock.call(
                "http://127.0.0.1:8343/delete/",
                headers={"Content-Type": "application/json"},
                json={"model": "foo"},
            ),
            mock.call(
                "http://127.0.0.1:8343/delete/",
                headers={"Content-Type": "application/json"},
                json={"model": "bar"},
            ),
        ]
        mock_post.assert_has_calls(expected_calls, any_order=False)

    @mock.patch("sllm._cli_utils.requests.post")
    def test_delete_failure(self, mock_post):
        mock_post.return_value.status_code = 404
        mock_post.return_value.text = "Not Found"

        result = self.runner.invoke(cli, ["delete", "nonexistent"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Failed to delete model 'nonexistent'", result.output)

        mock_post.assert_called_once_with(
            "http://127.0.0.1:8343/delete/",
            headers={"Content-Type": "application/json"},
            json={"model": "nonexistent"},
        )

    def test_delete_no_models(self):
        result = self.runner.invoke(cli, ["delete"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No model names provided for deletion", result.output)


if __name__ == "__main__":
    unittest.main()
