import unittest
from unittest import mock

from sllm_store.device_map_utils import (
    _compute_device_placement_from_map_fast,
    _get_sequential_placement,
)


def _fake_memory(devices):
    memory = dict(devices)
    memory["cpu"] = 100
    return memory


class TestSequentialPlacement(unittest.TestCase):
    """Unit tests for _get_sequential_placement."""

    def test_oversized_module_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            _get_sequential_placement({"a": 100}, {0: 50})

    def test_second_module_overflow_raises_runtime_error(self):
        with self.assertRaises(RuntimeError):
            _get_sequential_placement({"a": 10, "b": 100}, {0: 50, 1: 50})

    def test_skips_device_too_small_for_module(self):
        self.assertEqual(
            _get_sequential_placement({"a": 60, "b": 10}, {0: 50, 1: 100}),
            {"a": 1, "b": 1},
        )

    def test_exact_fit(self):
        self.assertEqual(
            _get_sequential_placement({"a": 50}, {0: 50}),
            {"a": 0},
        )


class TestFastPlacementEdgeCases(unittest.TestCase):
    """Unit tests for _compute_device_placement_from_map_fast."""

    def _place(self, modules, devices, device_map, tied=None):
        with mock.patch(
            "sllm_store.device_map_utils.get_max_memory",
            return_value=_fake_memory(devices),
        ):
            return _compute_device_placement_from_map_fast(
                dict(modules), list(tied or []), device_map
            )

    def test_sequential_drops_leading_device(self):
        self.assertEqual(
            self._place({"m1": 20, "m2": 10}, {0: 16, 1: 50}, "sequential"),
            {"m1": 1, "m2": 1},
        )

    def test_sequential_drops_two_leading_devices(self):
        self.assertEqual(
            self._place(
                {"m1": 20, "m2": 10},
                {0: 10, 1: 16, 2: 50},
                "sequential",
            ),
            {"m1": 2, "m2": 2},
        )

    def test_balanced_drops_leading_device(self):
        self.assertEqual(
            self._place({"m1": 20, "m2": 10}, {0: 16, 1: 50}, "balanced"),
            {"m1": 1, "m2": 1},
        )

    def test_first_module_too_large_for_all_devices(self):
        with self.assertRaises(RuntimeError):
            self._place({"m1": 20}, {0: 10}, "sequential")

    def test_no_drop_unchanged(self):
        self.assertEqual(
            self._place({"m1": 10, "m2": 10}, {0: 50, 1: 50}, "sequential"),
            {"m1": 0, "m2": 0},
        )

    def test_balanced_impossible_placement_raises(self):
        with self.assertRaises(RuntimeError):
            self._place({"a": 10, "b": 60}, {0: 50, 1: 50}, "balanced")

    def test_tied_modules_share_device_after_drop(self):
        self.assertEqual(
            self._place(
                {"a": 20, "b": 10},
                {0: 16, 1: 50},
                "sequential",
                tied=[(["a", "b"], 5)],
            ),
            {"a": 1, "b": 1},
        )


if __name__ == "__main__":
    unittest.main()
