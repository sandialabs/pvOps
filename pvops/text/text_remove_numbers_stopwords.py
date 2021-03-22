import re
import nltk


def text_remove_numbers_stopwords(document, lst_stopwords):
    """Conduct final processing steps after date extraction

    Parameters

    ----------
    document : str
        String representation of a document
    lst_stopwords : list
        List of stop words which will be filtered in final preprocessing step

    Returns

    -------
    string
        string of processed document
    """

    for char in "<>,.*?!/\\:\"'@#$%^&(){}[]|~`_-":
        document = document.replace(char, " ")

    # many documents use ; or - as sentence partitioners
    # for char in ';-':
    # document = document.replace(char,'')

    rem_num = re.sub("[0-9]+", "", document)

    # remove all spaces
    document_tok = nltk.word_tokenize(rem_num)
    document = [i for i in document_tok if i not in lst_stopwords]
    document = " ".join(document)

    return document
