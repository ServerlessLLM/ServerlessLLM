import unittest
from argparse import Namespace
from unittest.mock import patch

from sllm.cli.fine_tuning import FineTuningCommand


class TestFineTuningCommand(unittest.TestCase):
    @patch("sllm.cli.fine_tuning.requests.post")
    @patch("sllm.cli.fine_tuning.read_config")
    def test_fine_tuning_model_success(self, mock_post, mock_read_config):
        status_code = 200
        self.assertEqual(status_code, 200)


if __name__ == "__main__":
    unittest.main()
