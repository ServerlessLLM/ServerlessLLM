# tests/new_cli_test/test_delete.py

import unittest
from unittest import mock

from click.testing import CliRunner

from sllm.cli.clic import cli


class TestDeleteCommand(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()

    @mock.patch("sllm.cli._cli_utils.requests.post")
    def test_delete_success(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = "OK"

        result = self.runner.invoke(cli, ["delete", "foo", "bar"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "[✅ SUCCESS] Delete request for 'foo' sent successfully.",
            result.output,
        )
        self.assertIn(
            "[✅ SUCCESS] Delete request for 'bar' sent successfully.",
            result.output,
        )

        self.assertEqual(mock_post.call_count, 2)

        expected_calls = [
            mock.call(
                "http://127.0.0.1:8343/delete",
                headers={"Content-Type": "application/json"},
                json={"model": "foo"},
            ),
            mock.call(
                "http://127.0.0.1:8343/delete",
                headers={"Content-Type": "application/json"},
                json={"model": "bar"},
            ),
        ]
        mock_post.assert_has_calls(expected_calls, any_order=False)

    @mock.patch("sllm.cli._cli_utils.requests.post")
    def test_delete_failure(self, mock_post):
        mock_post.return_value.status_code = 404
        mock_post.return_value.text = "Not Found"

        result = self.runner.invoke(cli, ["delete", "nonexistent"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "[❌ ERROR] Failed to delete 'nonexistent'. Status: 404, Response: Not Found",
            result.output,
        )

        mock_post.assert_called_once_with(
            "http://127.0.0.1:8343/delete",
            headers={"Content-Type": "application/json"},
            json={"model": "nonexistent"},
        )

    def test_delete_no_models(self):
        result = self.runner.invoke(cli, ["delete"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            "[⚠️ WARNING] No model names provided for deletion.", result.output
        )

    # NEW TESTS TO CATCH REAL CLI BUGS:

    @mock.patch("sllm.cli._cli_utils.delete_model")
    def test_multiple_models_passed_correctly(self, mock_delete):
        """Test that multiple model arguments are passed correctly."""
        result = self.runner.invoke(
            cli, ["delete", "model1", "model2", "model3"]
        )

        mock_delete.assert_called_once()
        call_args = mock_delete.call_args[0]  # positional args
        call_kwargs = mock_delete.call_args[1]  # keyword args

        # First argument should be tuple of models
        self.assertEqual(call_args[0], ("model1", "model2", "model3"))
        # lora_adapters should be None when not provided
        self.assertIsNone(call_kwargs["lora_adapters"])

    @mock.patch("sllm.cli._cli_utils.delete_model")
    def test_lora_adapters_option_passed_correctly(self, mock_delete):
        """Test that --lora-adapters option is passed correctly."""
        result = self.runner.invoke(
            cli, ["delete", "model1", "--lora-adapters", "adapter1,adapter2"]
        )

        mock_delete.assert_called_once()
        call_args = mock_delete.call_args[0]
        call_kwargs = mock_delete.call_args[1]

        self.assertEqual(call_args[0], ("model1",))
        self.assertEqual(call_kwargs["lora_adapters"], "adapter1,adapter2")

    @mock.patch("sllm.cli._cli_utils.delete_model")
    def test_empty_lora_adapters_converted_to_none(self, mock_delete):
        """Test that empty lora-adapters string is converted to None."""
        result = self.runner.invoke(
            cli, ["delete", "model1", "--lora-adapters", ""]
        )

        mock_delete.assert_called_once()
        call_kwargs = mock_delete.call_args[1]

        # Empty string should be converted to None
        self.assertIsNone(call_kwargs["lora_adapters"])

    def test_variadic_arguments_work(self):
        """Test that the variadic models argument accepts any number of models."""
        # Test with no models
        result1 = self.runner.invoke(cli, ["delete"])
        self.assertEqual(result1.exit_code, 0)

        # Test with one model
        with mock.patch("sllm.cli._cli_utils.delete_model") as mock_delete:
            result2 = self.runner.invoke(cli, ["delete", "single-model"])
            self.assertEqual(result2.exit_code, 0)
            mock_delete.assert_called_once()
            self.assertEqual(mock_delete.call_args[0][0], ("single-model",))

        # Test with many models
        with mock.patch("sllm.cli._cli_utils.delete_model") as mock_delete:
            result3 = self.runner.invoke(
                cli, ["delete", "m1", "m2", "m3", "m4", "m5"]
            )
            self.assertEqual(result3.exit_code, 0)
            mock_delete.assert_called_once()
            self.assertEqual(
                mock_delete.call_args[0][0], ("m1", "m2", "m3", "m4", "m5")
            )

    def test_command_exists_and_has_correct_name(self):
        """Test that the delete command is properly registered."""
        # This would catch if the command decorator was broken
        result = self.runner.invoke(cli, ["--help"])
        self.assertIn("delete", result.output)

        # Test that the command has the right help text
        result = self.runner.invoke(cli, ["delete", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Delete deployed models", result.output)


if __name__ == "__main__":
    unittest.main()
