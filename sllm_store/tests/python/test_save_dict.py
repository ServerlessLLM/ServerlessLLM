# Copyright (c) ServerlessLLM Team 2026
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile
import unittest

import torch
from sllm_store.torch import save_dict


class TestSaveDict(unittest.TestCase):
    def test_unaligned_weights_and_shared_storage(self):
        for size in (1, 2047, 2048, 2049):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as path:
                tensor = torch.arange(size, dtype=torch.float16)
                save_dict({"weight": tensor, "alias": tensor}, path)
                with open(os.path.join(path, "tensor_index.json")) as f:
                    index = json.load(f)
                with open(os.path.join(path, "tensor.data_0"), "rb") as f:
                    data = f.read()
                expected = tensor.numpy().tobytes()
                offset, length = index["weight"][:2]
                self.assertEqual((offset, length), (0, len(expected)))
                self.assertEqual(index["weight"], index["alias"])
                self.assertEqual(len(data) % 4096, 0)
                self.assertEqual(data[:length], expected)
                self.assertEqual(data[length:], bytes(len(data) - length))

    def test_failed_save_does_not_publish_an_index(self):
        with tempfile.TemporaryDirectory() as path:
            os.mkdir(os.path.join(path, "tensor.data_0"))
            with self.assertRaises(RuntimeError):
                save_dict({"weight": torch.ones(1)}, path)
            self.assertFalse(
                os.path.exists(os.path.join(path, "tensor_index.json"))
            )
