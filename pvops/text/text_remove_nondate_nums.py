import re
import nltk


def text_remove_nondate_nums(document, PRINT_INFO=False):
    """Conduct initial text processing steps to prepare the text for date extractions.
    Function mostly uses regex-based text substitution to remove numerical structures
    within the text, which may be mistaken as a date by the date extractor.

    Parameters

    ----------
    document : str
        String representation of a document
    PRINT_INFO : bool
        Flag indicating whether to print information about the preprocessing progress

    Returns

    -------
    string
        string of processed document
    """

    if PRINT_INFO:
        print()
        print()
        print("IN: ", document)

    # Remove URLs
    find_URL = r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))"""
    document = re.sub(find_URL, " ", document)

    regexs = [
        r"\d+(\%|\s\%|\bpercent\b|\s\bpercent\b)",  # Take out 'd%', 'd %'
        r"(#\s|#)\d+",  # '#d', '# d'
        # r'-?\d+(,\d+)+(].]d*)?', # take out lists of numbers with no space
        r"-?\d+(,\d+)+(\.\d*)?",  # take out lists of numbers with no space
        # r'-?\d+(,\s\d+)+(].]d*)?', # take out list of numbers with space
        r"\s\d{3}\s",  # numeric with 3 digits
        r"\b(0|00|1[3-9]|[2-9]\d)\b-\d{4}",  # [1-12]-4DIGIT  allowed only
        r"\s+\d+\.\d+\s+",  # e.g.: 10.1 and 10.2 with space before and after
        r"\d{9,}",  # Take out numbers longer than 8 digits (8 because datetimes 20190320 should stay)
        r"\d+[.]+\d+[.]\d+[.][\d?]",  # Take out IP numbers
        r"\d-\d-\d",  # Take out single digit-hyphen trios e.g. 3-1-4  but leave 10-20-18 (possible date)
        r"\d[.]\d[.]\d",  # Take out single digit-hyphen trios e.g. 3-1-4  but leave 10-20-18 (possible date)
        r"\d+(\.\d*)?\s*[kK]?[wW]\s",
        r"\b(?!([jJ]an(uary)?|[fF]eb(r)?(uary)?|[mM]ar(ch)?|[aA]pr(il)?|[mM]ay|[jJ]un(e)?|[jJ]ul(y)?|[aA]ug(ust)?|[sS]ep(t)?(ember)?|[oO]ct(ober)?|[nN]ov(ember)?|[dD]ec(ember)?\b))[a-zA-Z]+-\d+",
        # ^ take out e.g. webbox-10
        r"[\w\.-]+@[\w\.-]+\.\w+",  # take out email addresses
        r"(\s\d{3}[-\.\s]?\d{3}[-\.\s]?\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]?\d{4}|\s\d{3}[-\.\s]\d{4})",  # take out phone numbers
        r"\s\d+[.]\d+\D+[.]\d+\s",  # e.g. neff - cb 2.1b.16 - forced outage ; unknown. at 1645 26-jun cb 2.1b.16 offline.. 0000 - unknown
        r"\s\D[.]\d[,\s]",
    ]

    replacements = [
        "",
        "",
        "",
        " ",
        " ",
        " ",
        "",
        "",
        "",
        "",
        " kW",
        " ",
        " ",
        " ",
        " ",
        " ",
    ]
    document = document.center(len(document) + 2)  # add spaces on either side
    for regex, repl in zip(regexs, replacements):
        # print('\t',regex)
        document = re.sub(regex, repl, document)
        # print('\t',words)

    if PRINT_INFO:
        print("SUB1:", document)
    # Decision to change all hyphens (-) to 'to'
    # to get rid of invalid timezone extrapolations

    document = str(document).lower()
    # print('prechk:',words)
    document = nltk.word_tokenize(document)

    if PRINT_INFO:
        print("TOKENED: ", document)

    # Remove single-character tokens (mostly punctuation)
    document = [word for word in document if len(word) > 1]
    if PRINT_INFO:
        print("FLTRD: ", document)
    document = " ".join(document)
    if PRINT_INFO:
        print("JOINED: ", document)

    # print('chkpt: ',words)
    regexs = [
        r"\d+(\%|\s\%|\bpercent\b|\s\bpercent\b)",  # Take out 'd%', 'd %'
        r"(#\s|#)\d+",  # '#d', '# d'
        r"\d+(,\d+)+(].]d*)?",  # take out lists of numbers with no space
        r"\d+(,\s\d+)+(].]d*)?",  # take out list of numbers with space
        r"\s\d{3}\s",  # numeric with 3 digits
        r"\s\b(0|00|1[3-9]|[2-9]\d)\b[-/]\d{4}\s",  # [1-12]-4DIGIT  allowed only: 91-1010
        r"\s\d{4}[-/]\b(0|00|1[3-9]|[2-9]\d)\b\s",  # 4DIGIT-[1-12]  allowed only: 4301/43
        r"\s\d+\[.]\d+\s",  # e.g.: ' 10.1 ' and 10.2
        r"\d{9,}",  # Take out numbers longer than 8 digits (8 because datetimes 20190320 should stay)
        r"\s\D[.]\s",  # Take out " m. " for maybe, ' c. ' for cerca, etc.
        r"\s\d{3}\/\d{2}\s",  # Take out 123/29 because not a date format, usually indicating temperature/etc.
        r"\s[a-zA-Z]+-[a-zA-Z]+\d\s",  # this and next one: e-a4 they are e7-1
        r"\s[a-zA-Z]\d+-\d+\s",
        r"\s[a-zA-Z]\d+\s",  # take out examples like `j23`
    ]
    replacements = ["", "", "", "", " ", " ", " ", " ", "", " ", " ", " ", " ", " "]

    document = document.center(len(document) + 2)  # add spaces on either side
    for regex, repl in zip(regexs, replacements):
        document = re.sub(regex, repl, document)

    if PRINT_INFO:
        print("TO DFINDER: ", document)

    return document