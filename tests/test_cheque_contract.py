import unittest

from app.routes.cheque import _build_cheque_result, _get_cached_cheque_text, _get_page_text, _validate_fields
from app.services.ocr import OCRService
from app.services.parser import Parser
from app.services.validator import Validator


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

    def test_cheque_number_is_the_first_six_digit_micr_block_only(self):
        text = "MICR Line: 018137 497521520 29 123456789 31"
        self.assertEqual(Parser.extract_cheque_number(text), "018137")

    def test_cheque_number_never_uses_account_routing_or_last_micr_block(self):
        text = "MICR Line: 018137 497521520 29 654321"
        self.assertEqual(Parser.extract_cheque_number(text), "018137")

    def test_cheque_number_does_not_skip_to_a_later_micr_block(self):
        text = "MICR Line: 497521520 29 654321"
        self.assertEqual(Parser.extract_cheque_number(text), "")

    def test_cheque_number_rejects_non_micr_numbers(self):
        text = "Cheque No: 018137 Account No: 497521520 Routing: 654321"
        self.assertEqual(Parser.extract_cheque_number(text), "")

    def test_micr_ocr_font_substitutions_preserve_the_first_six_digits(self):
        self.assertEqual(OCRService._first_micr_serial("OLBL37OLG2"), "016137")

    def test_micr_serial_consensus_prefers_the_most_repeated_six_digits(self):
        candidates = ["055515", "035151", "035151", "not-a-number"]
        self.assertEqual(OCRService._select_micr_serial(candidates), "035151")

    def test_micr_serial_consensus_rejects_a_single_unverified_read(self):
        self.assertEqual(OCRService._select_micr_serial(["035151"]), "")

    def test_cheque_number_validator_requires_exactly_six_digits(self):
        self.assertTrue(Validator.validate_cheque_number("035151"))
        self.assertFalse(Validator.validate_cheque_number("03515"))
        self.assertFalse(Validator.validate_cheque_number("0351512"))

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
