import os
import sys

pvops_path = os.path.join("pvops")
sys.path.append(pvops_path)

from text import visualize, preprocess, nlp_utils

import pandas as pd
import numpy as np
import datetime
import matplotlib
import nltk

def test_text_remove_nondate_nums():
    example = r"This is a test example https://www.google.com 10% #10 101 1-1-1 a-e4 13-1010 10.1 123456789 123/12 executed on 2/4/2020"
    answer = r" this is test example executed on 2/4/2020 "
    assert preprocess.text_remove_nondate_nums(example) == answer


def test_text_remove_numbers_stopwords():
    example = r"This is a test example 10% #10 101 1-1-1 13-1010 10.1 123456789 123/12 executed on 2/4/2020"
    answer = r"This test example executed"

    stopwords_answer = [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "ain",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "couldn",
        "couldn't",
        "d",
        "did",
        "didn",
        "didn't",
        "do",
        "does",
        "doesn",
        "doesn't",
        "doing",
        "don",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn",
        "hadn't",
        "has",
        "hasn",
        "hasn't",
        "have",
        "haven",
        "haven't",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "just",
        "ll",
        "m",
        "ma",
        "me",
        "mightn",
        "mightn't",
        "more",
        "most",
        "mustn",
        "mustn't",
        "my",
        "myself",
        "needn",
        "needn't",
        "no",
        "nor",
        "not",
        "now",
        "o",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "re",
        "s",
        "same",
        "shan",
        "shan't",
        "she",
        "she's",
        "should",
        "should've",
        "shouldn",
        "shouldn't",
        "so",
        "some",
        "such",
        "t",
        "than",
        "that",
        "that'll",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "ve",
        "very",
        "was",
        "wasn",
        "wasn't",
        "we",
        "were",
        "weren",
        "weren't",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
        "y",
        "you",
        "you'd",
        "you'll",
        "you're",
        "you've",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]

    stopwords = nlp_utils.create_stopwords()
    assert stopwords_answer == stopwords
    assert preprocess.text_remove_numbers_stopwords(example, stopwords) == answer


def test_get_dates():
    df = pd.DataFrame(
        [
            {
                "Date": "2020/01/23 12:34:56",
                "Document": "Find this date 2020/01/23 12:34:56",
            },
            {
                "Date": np.nan,
                "Document": "Find this date March 5 2021 and April 7 2022",
            },
        ]
    )

    answer = [datetime.datetime.strptime(
        "2020/01/23 12:34:56", "%Y/%m/%d %H:%M:%S")]
    assert answer == preprocess.get_dates(
        df["Document"].iloc[0], df, 0, {
            "data": "Document", "eventstart": "Date"}, False
    )

    answer = [
        datetime.datetime.strptime("2021/03/05 00:00:00", "%Y/%m/%d %H:%M:%S"),
        datetime.datetime.strptime("2022/04/07 00:00:00", "%Y/%m/%d %H:%M:%S"),
    ]
    assert answer == preprocess.get_dates(
        df["Document"].iloc[1], df, 1, {
            "data": "Document", "eventstart": "Date"}, False
    )


def test_visualize_attribute_timeseries():

    dates = pd.Series(
        [
            "2020/01/23 12:34:56",
            "2020/01/24 12:34:56",
            "2020/01/25 12:34:56",
        ]
    )

    dates = pd.to_datetime(dates).tolist()

    df = pd.DataFrame(
        {"labels": ["A word", "B word", "C word"], "date": dates})

    fig = visualize.visualize_attribute_timeseries(
        df, {"label": "labels", "date": "date"}, date_structure="%Y-%m-%d"
    )
    assert isinstance(fig, matplotlib.figure.Figure)


def xtest_visualize_word_frequency_plot():
    # Decommissioned because nltk's freqplot automatically shows
    # the rendered plot, meaning the test will get caught up
    documents = ["A word", "B word", "C word"]
    words = " ".join(documents)
    tokenized_words = nltk.word_tokenize(words)

    fig = visualize.visualize_word_frequency_plot(tokenized_words)

    assert isinstance(fig, nltk.FreqDist)


def test_visualize_attribute_connectivity():
    Attribute1 = ["A", "B", "C", "C"]
    Attribute2 = ["X", "X", "Y", "Z"]

    df = pd.DataFrame({"Attr1": Attribute1, "Attr2": Attribute2})

    om_col_dict = {"attribute1_col": "Attr1", "attribute2_col": "Attr2"}

    fig, edges = visualize.visualize_attribute_connectivity(
        df,
        om_col_dict,
        figsize=(10, 8),
        edge_width_scalar=2,
        graph_aargs={
            "with_labels": True,
            "font_weight": "bold",
            "node_size": 30,
            "font_size": 35,
        },
    )

    assert isinstance(fig, matplotlib.pyplot.Figure)
    assert list(edges) == [("A", "X"), ("X", "B"), ("C", "Y"), ("C", "Z")]

    matplotlib.pyplot.close()


def test_summarize_text_data():

    df = pd.DataFrame(
        [
            {
                "Date": "2020/01/23 12:34:56",
                "Document": "Find this date 2020/01/23 12:34:56",
            },
            {
                "Date": np.nan,
                "Document": "Find this date March 5 2021 and April 7 2022",
            },
        ]
    )

    answer = {
        "n_samples": 2,
        "n_nan_docs": 0,
        "n_words_doc_average": 7.50,
        "n_unique_words": 12,
        "n_total_words": 15.00,
    }

    info = nlp_utils.summarize_text_data(df, "Document")

    assert answer == info
