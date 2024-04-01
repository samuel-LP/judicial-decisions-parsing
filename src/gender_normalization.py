import re


class GenderNormalization():
    def __init__(self, text):
        self.text = text

    def normalize_gender(self):
        if (re.search(r"male", self.text, re.IGNORECASE)) or \
                (re.search(r"homme", self.text, re.IGNORECASE)):
            return "homme"
        elif (re.search(r"female", self.text, re.IGNORECASE)) or \
                (re.search(r"femme", self.text, re.IGNORECASE)):
            return "femme"
        else:
            return "n.c."
