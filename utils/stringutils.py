import re


def lower_no_space(s: str):
    ss = re.sub("[^a-zA-Z0-9_]", "", s.lower().replace(" ", "_"))
    return ss
