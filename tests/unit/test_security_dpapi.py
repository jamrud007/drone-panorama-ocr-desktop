import unittest
from pathlib import Path
from src.utils.security import dpapi_decrypt, dpapi_encrypt, retrieve_auth_token, store_auth_token


class TestSecurityDpapi(unittest.TestCase):
    def test_dpapi_roundtrip(self):
        original = "secret_jwt_bearer_token_12345"
        encrypted = dpapi_encrypt(original)
        self.assertNotEqual(original.encode("utf-8"), encrypted)
        decrypted = dpapi_decrypt(encrypted)
        self.assertEqual(original, decrypted)

    def test_store_and_retrieve_token(self):
        temp_token_file = Path("data/.test_token.dpapi")
        try:
            token = "jwt_token_sample_abcxyz"
            store_auth_token(token, storage_path=temp_token_file)
            self.assertTrue(temp_token_file.exists())

            retrieved = retrieve_auth_token(storage_path=temp_token_file)
            self.assertEqual(token, retrieved)
        finally:
            if temp_token_file.exists():
                temp_token_file.unlink()


if __name__ == "__main__":
    unittest.main()
