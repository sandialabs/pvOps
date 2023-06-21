
"""
Temporary placeholder to hold classify by search functions
"""

def get_keywords_of_interest(list_of_txt, reference_dict=None):
    """Find keywords of interest in list of strings from reference dict.

    If keywords of interest given in a reference dict are in the list of
    strings, return the keyword category, or categories. For example,
    if the string 'inverter' is in the list of text, return ['inverter'].

    Parameters
    ----------
    list_of_txt : list of str
        Tokenized text, functionally a list of string values.
    reference_dict : dict with {'keyword': [list of synonyms]} or None
        Reference dictionary to search for keywords of interest,
        in the expected format
        {'keyword_a':
            ['keyword_a', 'keyword_a_synonym_0', 'keyword_a_synonym_1', keyword_a_synonym_2', ...],
         'keyword_b':
            ['keyword_b', 'keyword_b_synonym_0', 'keyword_b_synonym_1', keyword_b_synonym_2', ...],
         ...}
        If None, use default reference dictionary.
        Note: This function can currently only handle single words, no n-gram functionality.

    Returns
    -------
    included_equipment: list of str
        List of keywords from reference_dict found in list_of_txt, can be more than one value.
    """
    text_to_search = set(list_of_txt)
    
    if reference_dict is None:
        # use default equipment lookup if no reference dict is given
        reference_dict = EQUIPMENT_DICT

    equipment_keywords = set(reference_dict.keys())
    included_equipment = list(text_to_search.intersection(equipment_keywords))

    return included_equipment

def add_keyword_labels(df, text_col, new_col, reference_dict=None):
    """Find keywords of interest in specified column of dataframe, return as new column value.

    If keywords of interest given in a reference dict are in the specified column of the dataframe,
    return the keyword category, or categories. For example, if the string 'inverter'
    is in the list of text, return ['inverter'].

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to search for keywords of interest, must include text_col.
    text_col : str
        Column of tokenized text values, to search for keywords.
    new_col :str
        Column name of new label column.
    reference_dict : dict with {'keyword': [list of synonyms]} or None
        Reference dictionary to search for keywords of interest,
        in the expected format
        {'keyword_a':
            ['keyword_a', 'keyword_a_synonym_0', 'keyword_a_synonym_1', keyword_a_synonym_2', ...],
         'keyword_b':
            ['keyword_b', 'keyword_b_synonym_0', 'keyword_b_synonym_1', keyword_b_synonym_2', ...],
         ...}
        If None, use default reference dictionary.
        Note: This function can currently only handle single words, no n-gram functionality.

    Returns
    -------
    df: pd.DataFrame
        Input df with new_col added, where each found keyword is its own row, may result in
        duplicate rows if more than one keywords of interest was found in text_col.
    """
    df[new_col] = df[text_col].apply(get_keywords_of_interest, reference_dict)

    # each multi-category now in its own row, some logs have multiple equipment issues
    df = df.explode(new_col)

    return df

