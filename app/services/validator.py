import re


class Validator:

    @staticmethod
    def validate_ifsc(ifsc):

        pattern = r"^[A-Z]{4}0[A-Z0-9]{6}$"

        if re.match(pattern, ifsc):
            return True

        return False


    @staticmethod
    def validate_account(account):

        pattern = r"^\d{9,18}$"

        if re.match(pattern, account):
            return True

        return False


    @staticmethod
    def validate_cheque_number(number):
        # This API's MICR contract is explicit: the cheque serial is the
        # first *six-digit* block.  Accepting arbitrary 5-8 digit OCR output
        # made malformed account/routing fragments appear valid.
        return bool(re.fullmatch(r"\d{6}", str(number or "")))


    @staticmethod
    def validate_cts(cts):
        value = str(cts or "").strip().upper()
        # Either a MICR/CTS code or the official CTS-2010 mark is valid.
        return bool(re.fullmatch(r"\d{9}", value) or re.fullmatch(r"CTS-20\d{2}", value))


    @staticmethod
    def validate_customer_name(name):
        value = str(name or "").strip()
        words = value.split()
        return 2 <= len(words) <= 10 and all(len(word) >= 2 for word in words)

    @staticmethod
    def validate_signature(signature_present):

        return bool(signature_present)
