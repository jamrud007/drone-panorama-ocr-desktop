import unittest
from src.ocr_client.client import HalotecOcrClient
from src.utils.security import is_jwt_format


class TestAuthValidation(unittest.TestCase):
    def test_jwt_format_check(self):
        self.assertFalse(is_jwt_format(""))
        self.assertFalse(is_jwt_format("asal_asalan_12345"))
        self.assertFalse(is_jwt_format("header.payload"))
        self.assertFalse(is_jwt_format("header.payload.signature.extra"))
        self.assertTrue(is_jwt_format("header.payload.signature"))
        self.assertTrue(is_jwt_format("Bearer header.payload.signature"))
        self.assertTrue(is_jwt_format("'header.payload.signature'"))

    def test_client_validate_bogus_token(self):
        client = HalotecOcrClient()
        is_valid, msg, validated_token = client.validate_token("token_asal_asalan")
        self.assertFalse(is_valid)
        self.assertIn("Format token tidak valid", msg)
        self.assertIsNone(validated_token)


if __name__ == "__main__":
    unittest.main()
