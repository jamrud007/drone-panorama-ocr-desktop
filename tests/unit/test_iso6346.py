import unittest
from src.utils.iso6346 import calculate_check_digit, validate_iso6346


class TestIso6346(unittest.TestCase):
    def test_valid_container(self):
        res = validate_iso6346("MSCU 123456-6")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.owner_code, "MSC")
        self.assertEqual(res.category, "U")
        self.assertEqual(res.serial_number, "123456")
        self.assertEqual(res.check_digit, 6)
        self.assertIsNone(res.error_message)

    def test_check_digit_zero_modulo(self):
        # CSQU3054383: check digit calculation edge cases
        res = validate_iso6346("CSQU3054383")
        self.assertTrue(res.is_valid)
        self.assertEqual(res.check_digit, 3)

    def test_invalid_check_digit(self):
        res = validate_iso6346("MSCU 123456-7")
        self.assertFalse(res.is_valid)
        self.assertEqual(res.check_digit, 7)
        self.assertEqual(res.expected_check_digit, 6)
        self.assertIn("Check digit mismatch", res.error_message)

    def test_invalid_category(self):
        res = validate_iso6346("MSCA 123456-6")
        self.assertFalse(res.is_valid)
        self.assertIn("Category identifier", res.error_message)

    def test_short_string(self):
        res = validate_iso6346("MSCU123")
        self.assertFalse(res.is_valid)
        self.assertIn("less than required 11 characters", res.error_message)


if __name__ == "__main__":
    unittest.main()
