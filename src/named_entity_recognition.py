
import time
from collections import Counter
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .utils import clean_text, is_subsequence
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification  # for tensorflow
from transformers import pipeline

model_name_or_path = "HooshvareLab/bert-fa-zwnj-base-ner"  # Roberta

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)  # Tensorflow

nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=-1)


def mine_entities(sentences: List[str], ent_labels: Optional[List[str]] = ["PER", "FAC", "ORG", "LOC", "EVE", "PRO"],
                  remove_punctuation: bool = True, remove_digits: bool = True, remove_chars: str = "", stop_words: Optional[List[str]] = None,
                  lowercase: bool = True, strip: bool = True, remove_whitespaces: bool = True, lemmatize: bool = False,
                  stem: bool = False, tags_to_keep: Optional[List[str]] = None, remove_n_letter_words: Optional[int] = None,
                  progress_bar: bool = False) -> Counter:
    """
    A function that goes through sentences and counts named entities found in the corpus.

    :param sentences: list of sentences
    :param ent_labels: list of entity labels to be considered (see SpaCy documentation)
    :param remove_punctuation: removing punctuations from the input sentences
    :param remove_digits: removing digits from the input sentences
    :param remove_chars: removing characters from the input sentences
    :param stop_words: removing stop-words from the input sentences
    :param lowercase: lowercase the input sentences
    :return: Counter with the named entity and its associated frequency on the corpus
    """

    entities_all = []

    if progress_bar:
        print("Mining named entities...")
        time.sleep(1)
        sentences = tqdm(sentences)

    for sentence in sentences:
        sentence = nlp(sentence)
        for ent in sentence:
            if ent['entity'][ent['entity'].find('-')+1:] in ent_labels:
                entities_all.append(ent['word'])

    entities_all = clean_text(entities_all, remove_punctuation, remove_digits, remove_chars, stop_words, lowercase,
                              strip, remove_whitespaces, lemmatize, stem, tags_to_keep, remove_n_letter_words)

    # forgetting to remove those will break the pipeline
    entities_all = [entity for entity in entities_all if entity != ""]

    entity_counts = Counter(entities_all)

    return entity_counts


def map_entities(statements: List[dict], entities: Counter, used_roles: List[str], top_n_entities: Optional[int] = None,
                 progress_bar: bool = False) -> Tuple[dict, List[dict]]:

    """
    A function that goes through statements and identifies pre-defined named entities within processed semantic roles.

    :param statements: list of dictionaries of processed semantic roles
    :param entities: user-defined list of named entities
    :param used_roles: list of semantic roles to be considered for named entity recognition
    :param top_n_entities: print a progress bar (default is False)
    :return:
        entity_index: dictionary containing statements indices with entities for each role
        roles_copy: new list of postprocessed semantic roles (without the named entities mined since they will not be embedded)
    """

    entities_keys = [el[0] for el in entities.most_common(top_n_entities)]

    entity_index = {
        role: {entity: np.asarray([], dtype=int) for entity in entities_keys}
        for role in used_roles
    }

    roles_copy = deepcopy(statements)

    if progress_bar:
        print("Mapping named entities...")
        time.sleep(1)
        statements = tqdm(statements)

    for i, statement in enumerate(statements):
        for role, role_content in roles_copy[i].items():
            if role in used_roles:
                for entity in entities_keys:
                    if is_subsequence(entity.split(), role_content.split()) is True:
                        entity_index[role][entity] = np.append(entity_index[role][entity], [i])
                        roles_copy[i][role] = ""

    return entity_index, roles_copy