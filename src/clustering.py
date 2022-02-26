import time
import warnings
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import tensorflow_hub as hub
from numpy.linalg import norm
from collections import Counter
import gensim.downloader as api
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from typing import List, Optional, Union
from .utils import count_values, count_words


def compute_sif_weights(words_counter: dict, alpha: Optional[float] = 0.001) -> dict:
    """
    A function that computes smooth inverse frequency (SIF) weights based on word frequencies.
    (See "Arora, S., Liang, Y., & Ma, T. (2016). A simple but tough-to-beat baseline for sentence embeddings.")
    Args:
        words_counter: a dictionary {"word": frequency}
        alpha: regularization parameter
    Returns:
        A dictionary {"word": SIF weight}
    """

    sif_dict = {}

    for word, count in words_counter.items():
        sif_dict[word] = alpha / (alpha + count)

    return sif_dict


class USE:

    """
    A class to call the Universal Sentence Encoder model.
    For further details: https://tfhub.dev/google/universal-sentence-encoder/4
    Download link/path: https://tfhub.dev/google/universal-sentence-encoder/4?tf-hub-format=compressed
    """

    def __init__(self, path: str):
        self._embed = hub.load(path)

    def __call__(self, tokens: List[str]) -> np.ndarray:
        return self._embed([" ".join(tokens)]).numpy()[0]
