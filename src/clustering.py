import pickle
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

    :param words_counter: A dictionary {"word": frequency}
    :param alpha:         Regularization parameter
    :return:              A dictionary {"word": SIF weight}
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


class SIF_word2vec:
    """
    A class to call a trained word2vec model using gensim's library.
    For basic code snippets and additional details: https://radimrehurek.com/gensim/models/word2vec.html
    """

    def __init__(self, path: str, sentences: List[str], alpha: Optional[float] = 0.001, normalize: bool = True):
        self.w2v_model: Word2Vec = self._load_keyed_vectors(path)
        self._words_counter = count_words(sentences)
        self._sif_dict = compute_sif_weights(self._words_counter, alpha)
        self._normalize = normalize

    def _load_keyed_vectors(self, path):
        """
        A tool to load w2v model from disk.
        :param path:   Model path.
        :return:       None
        """
        return Word2Vec.load(path)

    def __call__(self, tokens: List[str]):
        res = np.mean([self._sif_dict[token] * self.w2v_model.wv[token] for token in tokens], axis=0)
        if self._normalize: res = res / norm(res)
        return res

    def most_similar(self, v):
        return self.w2v_model.wv.most_similar(positive=[v], topn=1)[0]


class SIF_keyed_vectors(SIF_word2vec):
    """
    A class to call a pre-trained embeddings model from gensim's library.
    The embeddings are weighted by the smoothed inverse frequency of each token.
    For further details, see: https://github.com/PrincetonML/SIF
    # The list of pre-trained embeddings may be browsed by typing:
        import gensim.downloader as api
        list(api.info()['models'].keys())
    """

    def _load_keyed_vectors(self, path):
        return api.load(path)


def get_vector(tokens: List[str], model: Union[USE, SIF_word2vec, SIF_keyed_vectors]):
    """
    A function that computes an embedding vector for a list of tokens.

    :param tokens: List of string tokens to embed
    :param model: Trained embedding model. It can be either:
         - Universal Sentence Encoders (USE)
         - A full gensim Word2Vec model (SIF_word2vec)
         - Gensim Keyed Vectors based on a pre-trained model (SIF_keyed_vectors)
    :return: A two-dimensional numpy array (1, dimension of the embedding space)
    """

    if isinstance(model, SIF_word2vec):
        try:                return np.array([model(tokens)]).astype('double').reshape(1, -1)
        except KeyError:    return np.array([0 for _ in range(0, model.w2v_model.vector_size)]).reshape(1, -1)

    else:   res = np.array([model(tokens)])
    return  res


def get_vectors(postproc_roles, model: Union[USE, SIF_word2vec, SIF_keyed_vectors], used_roles=List[str]):
    """
    A function to train a K-Means model on the corpus.
    :param postproc_roles: List of statements
    :param model: Trained embedding model. It can be either:
         - Universal Sentence Encoders (USE)
         - A full gensim Word2Vec model (SIF_word2vec)
         - Gensim Keyed Vectors based on a pre-trained model (SIF_keyed_vectors)
    :param used_roles: List of semantic roles to cluster together
    :return: A list of vectors
    """

    role_counts = count_values(postproc_roles, keys=used_roles)
    role_counts = [role.split() for role in list(role_counts)]

    vecs = []
    for role in role_counts:
        vec = get_vector(role, model)
        if vec.shape[1] == 900:
            vecs.append(vec)

    vecs = np.concatenate(vecs)

    return vecs


def train_cluster_model(vecs, n_clusters, random_state: Optional[int] = 0, verbose: Optional[int] = 0):
    """
    Train a kmeans model on the corpus.

    :param vecs: list of vectors
    :param n_clusters: Number of clusters
    :param random_state: seed for replication (default is 0)
    :param verbose: see Scikit-learn documentation for details
    :return: A Scikit-learn kmeans model
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, verbose=verbose).fit(vecs)

    return kmeans


def get_clusters(postproc_roles: List[dict], model: Union[USE, SIF_word2vec, SIF_keyed_vectors], kmeans,
                 used_roles=List[str], progress_bar: bool = False, suffix: str = "_lowdim") -> List[dict]:
    """
    Predict clusters based on a pre-trained kmeans model.

    :param postproc_roles: list of statements
    :param model: Trained embedding model (e.g. either Universal Sentence Encoders, a full gensim Word2Vec model or gensim Keyed Vectors)
    :param kmeans: A pre-trained sklearn kmeans model
    :param used_roles: list of semantic roles to consider
    :param progress_bar: print a progress bar (default is False)
    :param suffix: suffix for the new dimension-reduced role's name (e.g. 'ARGO_lowdim')
    :return: A list of dictionaries with the predicted cluster for each role
    """

    roles_copy = deepcopy(postproc_roles)

    if progress_bar:
        print("Assigning clusters to roles...")
        time.sleep(1)
        postproc_roles = tqdm(postproc_roles)

    for i, statement in enumerate(postproc_roles):
        for role, tokens in statement.items():
            if role in used_roles:
                vec = get_vector(tokens.split(), model)
                if vec.shape[1] == 900:
                    vec = [[float(number) for number in vec[0]]]
                    vec = np.array(vec, dtype=np.double)
                    clu = kmeans.predict(vec)[0]
                    roles_copy[i][role] = clu
                else:
                    roles_copy[i].pop(role, None)
            else:
                roles_copy[i].pop(role, None)

    roles_copy = [
        {str(k + suffix): v for k, v in statement.items()} for statement in roles_copy
    ]

    return roles_copy


def label_clusters_most_freq(clustering_res: List[dict], postproc_roles: List[dict]) -> dict:
    """
    A function which labels clusters by their most frequent term.

    :param clustering_res: list of dictionaries with the predicted cluster for each role
    :param postproc_roles: list of statements
    :return: A dictionary associating to each cluster number a label (e.g. the most frequent term in this cluster)
    """

    temp = {}
    labels = {}

    for i, statement in enumerate(clustering_res):
        for role, cluster in statement.items():
            tokens = postproc_roles[i][role]
            cluster_num = cluster
            if cluster_num not in temp: temp[cluster_num] = [tokens]
            else:                       temp[cluster_num].append(tokens)

    for cluster_num, tokens in temp.items():
        token_most_common = Counter(tokens).most_common(2)
        if len(token_most_common) > 1 and (token_most_common[0][1] == token_most_common[1][1]):
            warnings.warn(
                f"Multiple labels for cluster {cluster_num}- 2 shown: {token_most_common}. First one is picked.",
                RuntimeWarning,
            )
        labels[cluster_num] = token_most_common[0][0]

    return labels


def label_clusters_most_similar(kmeans, model) -> dict:
    """
    A function which labels clusters by the term closest to the centroid in the embedding
    (i.e. distance is cosine similarity)

    :param kmeans: the trained kmeans model
    :param model: trained embedding model. It can be either:
         - a full gensim Word2Vec model (SIF_word2vec)
         - gensim Keyed Vectors based on a pre-trained model (SIF_keyed_vectors)
    :return: A dictionary associating to each cluster number a label (e.g. the most similar term to cluster's centroid)
    """

    labels = {}

    for i, vec in enumerate(kmeans.cluster_centers_):
        most_similar_term = model.most_similar(vec)
        labels[i] = most_similar_term[0]

    return labels