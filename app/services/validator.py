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

        pattern = r"^\d{6}$"

        if re.match(pattern, number):
            return True

        return False


    @staticmethod
    def validate_cts(cts):

        pattern = r"^\d{9}$"

        if re.match(pattern, cts):
            return True

        return False


    @staticmethod
    def validate_customer_name(name):

        if len(name.strip()) == 0:
            return False

        return True

    @staticmethod
    def validate_signature(signature_present):

        return bool(signature_present)