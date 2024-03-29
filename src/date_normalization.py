import re
from datetime import datetime


class DateNormalization():

    def __init__(self, text):
        self.text = text

    def extract_and_reformat_date(self):
        french_months = {
            'janvier': 1, 'février': 2, 'mars': 3,
            'avril': 4, 'mai': 5, 'juin': 6,
            'juillet': 7, 'août': 8, 'septembre': 9,
            'octobre': 10, 'novembre': 11, 'décembre': 12
        }

        if 'n.c.' in self.text:
            return "n.c."
        elif "n.a." in self.text:
            return 'n.a.'

        date_pattern_ymd = r"\d{4}-\d{2}-\d{2}"
        date_pattern_dmy = r"(\d{1,2}),?\s([a-zA-Zéû]+)\s(\d{4})"
        date_pattern_slash = r"(\d{1,2})/(\d{1,2})/(\d{4})"
        date_pattern_mdy = r"([a-zA-Z]+) (\d{1,2}), (\d{4})"

        match_ymd = re.search(date_pattern_ymd, self.text)
        match_dmy = re.search(date_pattern_dmy, self.text)
        match_slash = re.search(date_pattern_slash, self.text)
        match_mdy = re.search(date_pattern_mdy, self.text)

        if match_ymd:
            return match_ymd.group(0)

        elif match_dmy:
            day, month_name, year = match_dmy.groups()

            try:
                month_number = datetime.strptime(month_name, "%B").month
            except ValueError:
                month_number = french_months.get(month_name.lower())

                if month_number is None:
                    return "n.c."
            return f"{year}-{month_number:02d}-{day.zfill(2)}"

        elif match_slash:
            part1, part2, year = match_slash.groups()
            day, month = part1, part2
            return f"{year}-{int(month):02d}-{int(day):02d}"

        elif match_mdy:
            month_name, day, year = match_mdy.groups()

            try:
                month_number = datetime.strptime(month_name, "%B").month

            except ValueError:
                return "n.c."
            return f"{year}-{month_number:02d}-{day.zfill(2)}"
        else:
            return "n.c."
