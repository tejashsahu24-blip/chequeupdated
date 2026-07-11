import unittest

from app.routes.cheque import _build_cheque_result, _get_cached_cheque_text, _get_page_text, _validate_fields
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

    def test_customer_name_prefers_printed_honorific_name_over_noisy_hint(self):
        text = """Savinas at Par A/c. No.: sono0007023058083 | PSBA
        iFSC-CRGB0008210 Se
        ae Mr .SANTOSH KUMAR YADU
        Please sign above
        Field Hints: Cheque No: 018137 CTS-2010 Account No: 00007023058083
        Customer Name: as i? fe
        Mr. SANTOSH KUMAR YADU
        Please sign above"""
        self.assertEqual(Parser.extract_customer_name(text), "SANTOSH KUMAR YADU")

    def test_cheque_number_prefers_spaced_micr_digits_over_noise(self):
        text = "Field Hints: MICR Cheque No: 035151 Cheque No: 46696 497521520"
        self.assertEqual(Parser.extract_cheque_number(text), "035151")

    def test_cheque_number_prefers_micr_tag_over_other_printed_number(self):
        text = "Cheque No: 46696 MICR Cheque No: 035151 Account No: 497521520"
        self.assertEqual(Parser.extract_cheque_number(text), "035151")

    def test_pdf_page_uses_embedded_text_and_rendered_page_ocr(self):
        class FakeOCR:
            def read_text(self, _image_path):
                return "ocr-result"

            def extract_text(self, result):
                return result, 1.0

        text = _get_page_text("page.png", ["pdf-result"], 1, FakeOCR())
        self.assertEqual(text, "pdf-result ocr-result")

    def test_cached_cheque_text_is_computed_once_per_page(self):
        class FakeOCR:
            def __init__(self):
                self.page_reads = 0
                self.hint_reads = 0

            def read_text(self, _image_path):
                self.page_reads += 1
                return "ocr-result"

            def extract_text(self, result):
                return result, 1.0

            def read_cheque_field_hints(self, _image_path):
                self.hint_reads += 1
                return "MICR Cheque No: 035151"

        fake = FakeOCR()
        cache = {}

        first = _get_cached_cheque_text("page.png", ["pdf-result"], 1, fake, cache)
        second = _get_cached_cheque_text("page.png", ["pdf-result"], 1, fake, cache)

        self.assertEqual(first, "MICR Cheque No: 035151 pdf-result ocr-result")
        self.assertEqual(second, first)
        self.assertEqual(fake.page_reads, 1)
        self.assertEqual(fake.hint_reads, 1)
