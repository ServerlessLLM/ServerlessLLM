import unittest

from sllm_store.utils import to_num_bytes


class TestToNumBytes(unittest.TestCase):
    """Unit tests for the to_num_bytes function."""

    # -------------------
    # Valid Input Tests
    # -------------------
    def test_valid_bytes(self):
        self.assertEqual(to_num_bytes("1B"), 1)
        self.assertEqual(to_num_bytes("0B"), 0)
        self.assertEqual(to_num_bytes("1024B"), 1024)

    def test_valid_kilobytes(self):
        self.assertEqual(to_num_bytes("1KB"), 1024)
        self.assertEqual(to_num_bytes("512KB"), 512 * 1024)

    def test_valid_megabytes(self):
        self.assertEqual(to_num_bytes("1MB"), 1024**2)
        self.assertEqual(to_num_bytes("256MB"), 256 * 1024**2)

    def test_valid_gigabytes(self):
        self.assertEqual(to_num_bytes("1GB"), 1024**3)
        self.assertEqual(to_num_bytes("2GB"), 2 * 1024**3)

    def test_valid_terabytes(self):
        self.assertEqual(to_num_bytes("1TB"), 1024**4)
        self.assertEqual(to_num_bytes("5TB"), 5 * 1024**4)

    def test_valid_petabytes(self):
        self.assertEqual(to_num_bytes("1PB"), 1024**5)
        self.assertEqual(to_num_bytes("3PB"), 3 * 1024**5)

    def test_valid_exabytes(self):
        self.assertEqual(to_num_bytes("1EB"), 1024**6)
        self.assertEqual(to_num_bytes("7EB"), 7 * 1024**6)

    def test_valid_zettabytes(self):
        self.assertEqual(to_num_bytes("1ZB"), 1024**7)
        self.assertEqual(to_num_bytes("9ZB"), 9 * 1024**7)

    def test_valid_yottabytes(self):
        self.assertEqual(to_num_bytes("1YB"), 1024**8)
        self.assertEqual(to_num_bytes("2YB"), 2 * 1024**8)

    # -------------------
    # Invalid Input Tests
    # -------------------
    def test_invalid_unit_lowercase(self):
        with self.assertRaises(ValueError) as context:
            to_num_bytes("1gb")
        self.assertIn("Invalid format", str(context.exception))

    def test_invalid_unit_mixed_case(self):
        with self.assertRaises(ValueError) as context:
            to_num_bytes("1Gb")
        self.assertIn("Invalid format", str(context.exception))

    def test_invalid_unit_unknown(self):
        with self.assertRaises(ValueError) as context:
            to_num_bytes("1ABC")
        self.assertIn("Invalid format", str(context.exception))

    def test_leading_space(self):
        with self.assertRaises(ValueError) as context:
            to_num_bytes(" 1GB")
        self.assertIn("Invalid format", str(context.exception))

    def test_trailing_space(self):
        with self.assertRaises(ValueError) as context:
            to_num_bytes("1GB ")
        self.assertIn("Invalid format", str(context.exception))

    def test_middle_space(self):
        with self.assertRaises(ValueError) as context:
            to_num_bytes("1 GB")
        self.assertIn("Invalid format", str(context.exception))

    def test_leading_zero(self):
        # Depending on requirements, leading zeros might be allowed.
        self.assertEqual(to_num_bytes("0001GB"), 1024**3)

    def test_non_numeric_value(self):
        with self.assertRaises(ValueError) as context:
            to_num_bytes("GB")
        self.assertIn("Invalid format", str(context.exception))

    def test_decimal_number(self):
        with self.assertRaises(ValueError) as context:
            to_num_bytes("1.5GB")
        self.assertIn("Invalid format", str(context.exception))

    def test_negative_number(self):
        with self.assertRaises(ValueError) as context:
            to_num_bytes("-1GB")
        self.assertIn("Invalid format", str(context.exception))

    def test_empty_string(self):
        with self.assertRaises(ValueError) as context:
            to_num_bytes("")
        self.assertIn("Invalid format", str(context.exception))

    def test_additional_characters(self):
        with self.assertRaises(ValueError) as context:
            to_num_bytes("1GBs")
        self.assertIn("Invalid format", str(context.exception))

    # -------------------
    # Boundary Tests
    # -------------------
    def test_large_number(self):
        self.assertEqual(to_num_bytes("999999999YB"), 999999999 * 1024**8)

    def test_minimal_input(self):
        self.assertEqual(to_num_bytes("0B"), 0)

    def test_maximum_unit(self):
        self.assertEqual(to_num_bytes("1YB"), 1024**8)


# -------------------
# Running the Tests
# -------------------

if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
