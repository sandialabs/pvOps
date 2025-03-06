import os
import sys

from pvops.text import visualize, preprocess, nlp_utils

import pandas as pd
import numpy as np
import datetime
import matplotlib

def test_text_remove_nondate_nums():
    example = r"This is a test example https://www.google.com 10% #10 101 1-1-1 a-e4 13-1010 10.1 123456789 123/12 executed on 2/4/2020"
    answer = r" this is test example executed on 2/4/2020 "
    assert preprocess.text_remove_nondate_nums(example) == answer


def test_text_remove_numbers_stopwords():
    example = r"This is a test example 10% #10 101 1-1-1 13-1010 10.1 123456789 123/12 executed on 2/4/2020"
    answer = r"This test example executed"

    stopwords = nlp_utils.create_stopwords()
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


def test_visualize_word_frequency_plot():
    documents = ["A word", "B word", "C word"]
    words = " ".join(documents)
    tokenized_words = preprocess.regex_tokenize(words)

    result = visualize.visualize_word_frequency_plot(tokenized_words)

    assert isinstance(result[0], matplotlib.pyplot.Figure)
    assert isinstance(result[1], dict)


def test_visualize_attribute_connectivity():
    Attribute1 = ["A", "B", "C", "C"]
    Attribute2 = ["X", "X", "Y", "Z"]

    df = pd.DataFrame({"Attr1": Attribute1, "Attr2": Attribute2})

    om_col_dict = {"attribute1_col": "Attr1", "attribute2_col": "Attr2"}

    fig, G = visualize.visualize_attribute_connectivity(
        df,
        om_col_dict,
        figsize=(10, 8),
        edge_width_scalar=2,
        graph_aargs={
            "with_labels": True,
            "font_weight": "bold",
        },
    )

    assert isinstance(fig, matplotlib.pyplot.Figure)
    assert list(G.edges()) == [("A", "X"), ("B", "X"), ("C", "Y"), ("C", "Z")]

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
