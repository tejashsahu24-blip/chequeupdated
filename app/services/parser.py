import re


class Parser:

    @staticmethod
    def clean_text(text):

        if text is None:
            return ""

        text = text.strip()

        text = text.replace("\n", " ")

        text = text.replace("\t", " ")

        return text

    @staticmethod
    def clean_ifsc(ifsc):

        ifsc = Parser.clean_text(ifsc)

        ifsc = ifsc.upper()

        ifsc = ifsc.replace("O", "0")

        ifsc = ifsc.replace(" ", "")

        return ifsc

    @staticmethod
    def clean_account(account):

        account = Parser.clean_text(account)

        account = account.replace(" ", "")

        account = account.replace("-", "")

        return account

    @staticmethod
    def clean_cheque_number(number):

        number = Parser.clean_text(number)

        number = number.replace(" ", "")

        return number

    @staticmethod
    def clean_cts(cts):

        cts = Parser.clean_text(cts)

        cts = cts.replace(" ", "")

        return cts

    @staticmethod
    def clean_name(name):

        name = Parser.clean_text(name)

        name = re.sub(r"[^A-Za-z ]", "", name)

        return name.upper()

    # ------------------------------------------------------------------
    # Field extraction helpers
    #
    # The clean_* helpers above simply normalise whatever text is handed
    # to them. The extract_* helpers below actually pull the relevant
    # field value out of the full OCR text of a cheque so that each field
    # can be validated on its own (IFSC, account number, cheque number,
    # CTS number, customer name) instead of validating the raw OCR text.
    # ------------------------------------------------------------------

    _NAME_STOPWORDS = {
        "PAY", "RUPEES", "ONLY", "BANK", "LIMITED", "LTD", "IFSC",
        "ACCOUNT", "PAYEE", "BEARER", "NOT", "NEGOTIABLE", "CHEQUE",
        "BRANCH", "DATE", "SIGN", "SIGNATURE", "VALID", "MONTHS", "OR",
        "AND", "FOR", "NO", "CTS", "MICR", "A", "C", "PLEASE", "ABOVE",
        "CROSSING", "LINES", "WITHIN"
    }

    @staticmethod
    def extract_ifsc(text):
        """
        Find the IFSC code anywhere within the OCR text.

        Matching is done token-by-token (instead of against the whole
        blob of text with spaces stripped) so that unrelated words like
        "BANK OF INDIA" can't accidentally merge into something that
        looks like an IFSC code.
        """

        cleaned_text = Parser.clean_text(text).upper()

        for token in re.split(r"\s+", cleaned_text):
            candidate = re.sub(r"[^A-Z0-9]", "", token)

            if len(candidate) != 11 or candidate[4] not in "0O":
                continue

            normalized = Parser.clean_ifsc(candidate)

            if re.match(r"^[A-Z]{4}0[A-Z0-9]{6}$", normalized):
                return normalized

        return ""

    @staticmethod
    def extract_account_number(text):
        """Find the account number, allowing for spaces/hyphens between digits."""

        cleaned_text = Parser.clean_text(text)

        groups = re.findall(r"\d[\d\s-]{7,20}\d", cleaned_text)

        candidates = [
            Parser.clean_account(group)
            for group in groups
            if 9 <= len(Parser.clean_account(group)) <= 18
        ]

        if not candidates:
            return ""

        candidates.sort(key=len, reverse=True)

        return candidates[0]

    @staticmethod
    def extract_cheque_number(text):
        """Find a standalone 6 digit cheque number."""

        cleaned_text = Parser.clean_text(text)

        matches = re.findall(r"\b\d{6}\b", cleaned_text)

        return matches[0] if matches else ""

    @staticmethod
    def extract_cts(text):
        """Find a standalone 9 digit CTS/MICR number, distinct from the account number."""

        cleaned_text = Parser.clean_text(text)

        matches = re.findall(r"\b\d{9}\b", cleaned_text)

        account_number = Parser.extract_account_number(cleaned_text)

        for match in matches:
            if match != account_number:
                return match

        return matches[0] if matches else ""

    @staticmethod
    def extract_customer_name(text):
        """Find the longest run of non bank-related words to use as the customer name."""

        cleaned_text = Parser.clean_text(text)

        words = re.findall(r"[A-Za-z]+", cleaned_text)

        best_phrase = []
        current_phrase = []

        for word in words:
            if word.upper() in Parser._NAME_STOPWORDS or len(word) < 2:
                if len(current_phrase) >= 2 and len(current_phrase) > len(best_phrase):
                    best_phrase = current_phrase
                current_phrase = []
                continue

            current_phrase.append(word)

        if len(current_phrase) >= 2 and len(current_phrase) > len(best_phrase):
            best_phrase = current_phrase

        return " ".join(best_phrase).upper()

    @staticmethod
    def extract_fields(text):
        """Extract all cheque fields from the raw OCR text in one call."""

        cleaned_text = Parser.clean_text(text)

        return {
            "ifsc": Parser.extract_ifsc(cleaned_text),
            "account_number": Parser.extract_account_number(cleaned_text),
            "cheque_number": Parser.extract_cheque_number(cleaned_text),
            "cts": Parser.extract_cts(cleaned_text),
            "customer_name": Parser.extract_customer_name(cleaned_text)
        }