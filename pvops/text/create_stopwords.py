import nltk


def create_stopwords(lst_langs=["english"], lst_add_words=[], lst_keep_words=[]):
    """Concatenate a list of stopwords using both words grabbed from nltk and user-specified words.

    Parameters

    ---------
    lst_langs : list
        List of strings designating the languages for a nltk.corpus.stopwords.words query. If empty list is passed, no stopwords will be queried from nltk.
    lst_add_words : list
        List of words (e.g., "road" or "street") to add to stopwords list. If these words are already included in the nltk query, a duplicate will not be added.
    lst_keep_words : list
        List of words (e.g., "before" or "until") to remove from stopwords list. This is usually used to modify default stop words that might be of interest to PV.

    Returns

    -------
    List
        List of alphabetized stopwords
    """
    lst_stopwords = set()
    for lang in lst_langs:
        try:
            stopwords = nltk.corpus.stopwords.words(lang)
        except LookupError:
            nltk.download("stopwords")
        lst_stopwords = lst_stopwords.union(stopwords)
    lst_stopwords = lst_stopwords.union(lst_add_words)
    lst_stopwords = list(set(lst_stopwords) - set(lst_keep_words))
    return sorted(list(set(lst_stopwords)))
