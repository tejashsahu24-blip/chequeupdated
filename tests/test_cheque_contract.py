import unittest

from app.routes.cheque import _build_cheque_result, _validate_fields
from app.services.parser import Parser


class ChequeContractTests(unittest.TestCase):
    def test_cheque_response_has_required_json_keys(self):
        fields = {
            "account_number": "123456789",
            "ifsc": "SBIN0001234",
            "customer_name": "TEST CUSTOMER",
            "cheque_number": "123456",
            "cts": "CTS-2010",
        }

        validations = _validate_fields(fields, True)
        result = _build_cheque_result(fields, validations, True)

        self.assertTrue({"cheque_no", "account_no", "ifsc", "cts", "customer_name", "signature", "valid"} <= set(result))
        self.assertTrue(result["valid"])

    def test_customer_name_uses_payee_not_bank_boilerplate(self):
        text = "Pay SHRI RAM FINANCE CORPORATION PVT LTD or Bearer Rupees"
        self.assertEqual(Parser.extract_customer_name(text), "SHRI RAM FINANCE CORPORATION PVT")

    def test_garbled_customer_name_is_rejected(self):
        self.assertEqual(Parser.extract_customer_name("EEE EE ELL CE PAW INE STE"), "")
