import matplotlib
import matplotlib.pyplot as plt
import nltk


def visualize_word_frequency_plot(
    tokenized_words, title="", font_size=16, graph_aargs={}
):
    """Visualize the frequency distribution of words within a set of documents

    Parameters

    ----------
    tokenized_words : list
        List of tokenized words
    title : str
        Optional, title of plot
    font_size : int
        Optional, font size
    **aargs :
        Optional, other parameters passed to nltk.FreqDist.plot()

    Returns

    -------
    Matplotlib figure instance
    """
    matplotlib.rcParams.update({"font.size": font_size})
    fd = nltk.FreqDist(tokenized_words)
    fig = plt.figure(figsize=(12, 6))
    fd.plot(30, cumulative=False, title=title, figure=fig, **graph_aargs)
    return fd