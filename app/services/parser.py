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
        return Parser._clean_numeric(number)

    @staticmethod
    def clean_cts(cts):
        return Parser._clean_numeric(cts)

    @staticmethod
    def _clean_numeric(value):
        """Normalise common OCR substitutions in printed cheque numbers."""
        value = Parser.clean_text(value).upper()
        value = value.translate(str.maketrans({
            "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
            "Z": "2", "S": "5", "B": "8", "G": "6",
        }))
        return re.sub(r"[^0-9]", "", value)

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
        "CROSSING", "LINES", "WITHIN", "PAYABLE", "AT", "PAR", "ANY",
        "OF", "THE", "BRANCHES", "DELIGHT"
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

        # OCR often inserts spaces or punctuation inside an IFSC code, e.g.
        # ``SBIN 0 012345``.  Search a compact view as well as individual
        # tokens, but require an IFSC/account context so ordinary words cannot
        # be joined accidentally.
        for match in re.finditer(
            r"IFSC\s*(?:CODE)?\s*[:#-]?\s*([A-Z]{4}\s*[0O](?:\s*[A-Z0-9]){6})",
            cleaned_text,
        ):
            candidate = Parser.clean_ifsc(match.group(1))
            if re.fullmatch(r"[A-Z]{4}0[A-Z0-9]{6}", candidate):
                return candidate

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

        keyword_match = re.search(
            r"(?:a/c|account|acc(?:ount)?)(?:\s*(?:no|number|#|\.))?\s*[:\-]?\s*(\d[\d\s-]{7,20}\d)",
            cleaned_text,
            re.IGNORECASE
        )

        if keyword_match:
            candidate = Parser.clean_account(keyword_match.group(1))
            if 9 <= len(candidate) <= 18:
                return candidate

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
        """Find a cheque number by keyword or standalone digits."""

        cleaned_text = Parser.clean_text(text)

        micr_match = re.search(
            r"\bMICR(?:\s*(?:CHEQUE|CHQ))?(?:\s*(?:NO|NUMBER|GROUP|LINE))?\s*[:#-]?\s*((?:\d[\s-]*){5,8})",
            cleaned_text,
            re.IGNORECASE,
        )
        if micr_match:
            candidate = Parser.clean_cheque_number(micr_match.group(1))
            if 5 <= len(candidate) <= 8:
                return candidate

        keyword_match = re.search(
            r"(?:cheque|check|chq)(?:\s*(?:no|number)?)[^0-9OQDILZSBG]*([0-9OQDILZSBG][0-9OQDILZSBG\s-]{3,8}[0-9OQDILZSBG])",
            cleaned_text,
            re.IGNORECASE
        )

        matches = re.findall(r"\b\d{6}\b", cleaned_text)
        if matches:
            return matches[0]

        spaced_matches = []
        for match in re.finditer(r"(?:\b\d\b[\s-]*){6}", cleaned_text):
            candidate = re.sub(r"[^0-9]", "", match.group(0))
            if len(candidate) == 6:
                spaced_matches.append(candidate)

        if spaced_matches:
            return spaced_matches[0]

        if keyword_match:
            candidate = Parser.clean_cheque_number(keyword_match.group(1))
            if 5 <= len(candidate) <= 8:
                return candidate

        matches = re.findall(r"\b\d{5,8}\b", cleaned_text)
        return matches[0] if matches else ""

    @staticmethod
    def extract_cts(text):
        """Find a standalone 9 digit CTS/MICR number, distinct from the account number."""

        cleaned_text = Parser.clean_text(text)

        # Some cheque leaves expose the official CTS compliance mark instead
        # of a separate MICR/CTS number.
        cts_mark = re.search(r"\bCTS\s*[- ]?\s*20\d{2}\b", cleaned_text, re.IGNORECASE)

        keyword_match = re.search(
            r"(?:CTS|MICR)(?:\s*(?:NO|NUMBER|CODE))?\s*[:#-]?\s*(\d[\d\s-]{7,12}\d)",
            cleaned_text,
            re.IGNORECASE,
        )
        if keyword_match:
            candidate = Parser.clean_cts(keyword_match.group(1))
            if len(candidate) == 9 and candidate.isdigit():
                return candidate

        if cts_mark:
            return cts_mark.group(0).upper().replace(" ", "-")

        matches = re.findall(r"\b\d{9}\b", cleaned_text)

        account_number = Parser.extract_account_number(cleaned_text)

        for match in matches:
            if match != account_number:
                return match

        return matches[0] if matches else ""

    @staticmethod
    def extract_customer_name(text):
        """Extract the payee/customer name without accepting bank boilerplate."""

        cleaned_text = Parser.clean_text(text)

        # Account-holder names on Indian cheques are often printed with an
        # honorific (for example ``Mr. SANTOSH KUMAR YADU``).  Prefer this
        # highly specific pattern before looking at OCR-generated field
        # hints: a hint such as ``Customer Name: as i? fe`` can otherwise
        # consume the following real name after whitespace is normalised.
        titled_name = re.search(
            r"\b(?:mr|mrs|ms)\s*\.?\s*([A-Za-z]{2,}(?:\s+[A-Za-z]{2,}){1,5}?)(?=\s+(?:PLEASE|SIGN|FIELD|HINTS|RUPEES|OR\s+BEARER|BEARER|A/C|ACCOUNT|CBS)\b|$)",
            cleaned_text,
            re.IGNORECASE,
        )
        if titled_name:
            candidate = Parser._valid_name_candidate(titled_name.group(1))
            if candidate:
                return candidate

        # Prefer an explicitly labelled account-holder/customer name.  It is
        # substantially more reliable than choosing the longest OCR phrase on
        # a cheque, which is frequently bank boilerplate.
        labelled_name = re.search(
            r"(?:customer|account\s*holder|account)\s*name\s*[:#-]?\s*([A-Za-z][A-Za-z .]{2,100}?)(?=\s+(?:RUPEES|OR\s+BEARER|BEARER|A/C|ACCOUNT|CBS)\b|$)",
            cleaned_text,
            re.IGNORECASE,
        )
        if labelled_name:
            candidate = Parser._valid_name_candidate(labelled_name.group(1))
            if candidate:
                return candidate

        payee_name = re.search(
            r"\bPAY\s*[:#-]?\s*([A-Za-z][A-Za-z .]{2,100}?)(?=\s+(?:RUPEES|OR\s+BEARER|BEARER)\b|$)",
            cleaned_text,
            re.IGNORECASE,
        )
        if payee_name:
            candidate = Parser._valid_name_candidate(payee_name.group(1))
            if candidate:
                return candidate

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

        if not best_phrase and current_phrase:
            best_phrase = current_phrase

        return Parser._valid_name_candidate(" ".join(best_phrase))

    @staticmethod
    def _valid_name_candidate(value):
        """Return a plausible name, otherwise an empty string.

        This prevents noisy OCR or printed bank copy from making a cheque
        appear valid merely because it contains alphabetic characters.
        """
        name = Parser.clean_name(value)
        # Discard isolated OCR specks (for example, a stray ``H`` next to a
        # handwritten payee) rather than throwing away the entire name.
        words = [
            word for word in name.split()
            if word not in Parser._NAME_STOPWORDS and len(word) >= 2
        ]
        if not 2 <= len(words) <= 10:
            return ""
        if sum(len(word) for word in words) < 5:
            return ""
        if sum(len(word) for word in words) / len(words) < 3:
            return ""
        return " ".join(words)

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
