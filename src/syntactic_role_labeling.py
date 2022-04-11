import os.path
import time
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union, Generator, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from crf_pos.pos_tagger import WapitiPosTagger

from src.utils import clean_text


class SyntacticDP():
    def __init__(self, max_batch_char_length: Optional[int] = None, batch_size: Optional[int] = None,
                 max_sentence_length: Optional[int] = None, max_number_words: Optional[int] = None):
        self.tagger = WapitiPosTagger()
        self._max_batch_char_length = max_batch_char_length
        self._batch_size = batch_size
        self._max_sentence_length = max_sentence_length
        self._max_number_words = max_number_words

    def __call__(self, sentences: List[str], progress_bar: bool = False) -> List[Dict[str, Union[str, List[str]]]]:
        if progress_bar:
            print("Running SDP...")
            time.sleep(1)
            sentences = tqdm(sentences)

        return list(map(lambda item: SyntacticDP.sdp2srl_mock(item)[0], list(self.batch_tagger(sentences))))

    def batch_tagger(self, sentences: Union[tqdm, List[Dict[str, str]]]) -> Generator[List[tuple], None, None]:
        for sentence in sentences:
            yield self.tagger[sentence]

    @staticmethod
    def extract_pos_words(pos_tagged: List[Tuple[str, str]], role: str) -> Generator[str, None, None]:
        for item in pos_tagged:
            if item[0] == role: yield item[1]

    @staticmethod
    def find_pos_index(pos_tagged: List[Tuple[str, str]], role: str) -> int:
        try:    return [item[1] for item in pos_tagged].index(role)
        except: return len(pos_tagged)

    @staticmethod
    def findall_pos_index(pos_tagged: List[Tuple[str, str]], role: str) -> Generator[int, None, None]:
        pos_tags = [item[1] for item in pos_tagged]
        for ind, tag in enumerate(pos_tags):
            if tag == role: yield ind

    @staticmethod
    def find_pos_word(pos_tagged: List[Tuple[str, str]], role: str, index: int) -> str:
        try:    return list(SyntacticDP.extract_pos_words(pos_tagged, role))[index]
        except: return ''

    @staticmethod
    def pos2srl(pos_tagged: List[Tuple[str, str]]) -> Generator[Tuple[str, str], None, None]:
        keys = {'PRO': 'ARG0', 'V': 'V', 'N': 'ARG1', 'ADJ': 'ARG2',
                'NUM': '', 'CON': '', 'ADV': 'ARG2'}
        verb_ind = SyntacticDP.find_pos_index(pos_tagged, 'V')
        clit_ind = SyntacticDP.find_pos_index(pos_tagged, 'CLITIC')
        for ind, item in enumerate(pos_tagged):
            if item[1] == 'N' and ind < min(verb_ind, 3):   yield item[0], 'ARG0'
            elif item[1] == 'N' and ind < verb_ind-3:       yield item[0], 'ARG0'
            elif item[1] == 'N' and ind == clit_ind-1:      yield item[0], 'ARG1'
            elif item[1] == 'N' and ind >= verb_ind+2:      yield item[0], 'ARG1'
            else:
                try:    yield item[0], keys[item[1]]
                except: yield item[0], ''

    @staticmethod
    def concat_srl(rel_tagged: Generator[Tuple[str, str], None, None]) -> Generator[Tuple[str, str], None, None]:
        rel_tagged = list(rel_tagged)
        ind = 0
        while ind < len(rel_tagged):
            out_str = rel_tagged[ind][0]
            j = ind + 1
            while j < len(rel_tagged):
                if rel_tagged[j][1] == rel_tagged[ind][1]:  out_str += f' {rel_tagged[j][0]}'
                else:                                       break
                j += 1
            yield rel_tagged[ind][1], out_str
            ind = j

    @staticmethod
    def mock_description(rel_tagged: List[Tuple[str, str]]) -> str:
        return ' '.join([f"[{': '.join(item[::-1])}]" if item[0] != '' else item[1] for item in rel_tagged])

    @staticmethod
    def mock_tags(rel_tagged: List[Tuple[str, str]]) -> Generator[str, None, None]:
        for item in rel_tagged:
            if item[0] == '':       yield 'O'
            else:
                yield f'B-{item[0]}'
                for _ in item[1].split()[1:]:
                    if item[0] == '':       yield 'O'
                    elif item[0] != 'V':    yield f'I-{item[0]}'
                    else:                   yield f'B-{item[0]}'

    @staticmethod
    def sdp2srl_mock(pos_tagged: List[Tuple[str, str]]) -> List[Dict[str, Union[str, List[str]]]]:
        verb_index = SyntacticDP.findall_pos_index(pos_tagged, 'V')
        concat_srl_var = list(SyntacticDP.concat_srl(SyntacticDP.pos2srl(pos_tagged)))
        return [{'verbs': [{'verb': ' '.join([pos_tagged[ind][0] for ind in verb_index]).replace('\u200c', ' '),
                            'description': SyntacticDP.mock_description(concat_srl_var),
                            'tags': list(SyntacticDP.mock_tags(concat_srl_var))}],
                 'words': [item[0] for item in pos_tagged]}]


def extract_roles(srl: List[Dict[str, Any]], used_roles: List[str],
                  progress_bar: bool = False) -> Tuple[List[Dict[str, Union[str, bool]]], List[int]]:

    """
    A function that extracts semantic roles from the SRL output.
    Args:
        srl: srl output
        used_roles: list of semantic roles to extract
        progress_bar: print a progress bar (default is False)
    Returns:
        List of statements and numpy array of sentence indices (to keep track of sentences)
    """

    statements_role_list: List[Dict[str, Union[str, bool]]] = []
    sentence_index: List[int] = []

    if progress_bar:
        print("Processing SRL...")
        time.sleep(1)
        srl = tqdm(srl)

    for i, sentence_dict in enumerate(srl):
        role_per_sentence = extract_role_per_sentence(sentence_dict, used_roles)
        sentence_index.extend([i] * len(role_per_sentence))
        statements_role_list.extend(role_per_sentence)

    return statements_role_list, np.asarray(sentence_index, dtype=np.uint32)


def extract_role_per_sentence(sentence_dict: dict, used_roles: List[str]) -> List[Dict[str, Union[str, bool]]]:

    """
    A function that extracts the semantic roles for a given sentence.
    Args:
        srl: srl output
        used_roles: list of semantic roles to extract
    Returns:
        List of statements with their associated roles for a given sentence
    """

    word_list = sentence_dict["words"]
    sentence_role_list = []

    for statement_dict in sentence_dict["verbs"]:
        tag_list = statement_dict["tags"]

        statement_role_dict: Dict[str, Union[str, bool]] = {}
        for role in ["ARG0", "ARG1", "ARG2", "B-V", "B-ARGM-MOD"]:
            if role in used_roles:
                indices_role = [i for i, tok in enumerate(tag_list) if role in tok]
                toks_role = [
                    tok for i, tok in enumerate(word_list) if i in indices_role
                ]
                statement_role_dict[role] = " ".join(toks_role)

        if "B-ARGM-NEG" in used_roles:
            role_negation_value = any("B-ARGM-NEG" in tag for tag in tag_list)
            statement_role_dict["B-ARGM-NEG"] = role_negation_value

        key_to_delete = []
        for key, value in statement_role_dict.items():
            if not value:
                key_to_delete.append(key)
        for key in key_to_delete:
            del statement_role_dict[key]
        sentence_role_list.append(statement_role_dict)

    if not sentence_role_list:
        sentence_role_list = [{}]

    return sentence_role_list


def process_roles(statements: List[Dict[str, List]], max_length: Optional[int] = None, remove_punctuation: bool = True,
                  remove_digits: bool = True, remove_chars: str = "", stop_words: Optional[List[str]] = None,
                  lowercase: bool = True, strip: bool = True, remove_whitespaces: bool = True, lemmatize: bool = False,
                  stem: bool = False, tags_to_keep: Optional[List[str]] = None, remove_n_letter_words: Optional[int] = None,
                  progress_bar: bool = False) -> List[Dict[str, List]]:

    """
    Takes a list of raw extracted semantic roles and cleans the text.
    Args:
        max_length = remove roles of more than n characters (NB: very long roles tend to be uninformative)
        progress_bar: print a progress bar (default is False)
        For other arguments see utils.clean_text.
    Returns:
        List of processed statements
    """

    roles_copy = deepcopy(statements)

    if progress_bar:
        print("Cleaning SRL...")
        time.sleep(1)
        statements = tqdm(statements)

    for i, statement in enumerate(statements):
        for role, role_content in roles_copy[i].items():
            if isinstance(role_content, str):
                res = clean_text(
                    [role_content],
                    remove_punctuation=remove_punctuation,
                    remove_digits=remove_digits,
                    remove_chars=remove_chars,
                    stop_words=stop_words,
                    lowercase=lowercase,
                    strip=strip,
                    remove_whitespaces=remove_whitespaces,
                    lemmatize=lemmatize,
                    stem=stem,
                    tags_to_keep=tags_to_keep,
                    remove_n_letter_words=remove_n_letter_words,
                )[0]
                if max_length is not None:
                    if len(res) <= max_length:
                        roles_copy[i][role] = res
                    else:
                        roles_copy[i][role] = ""
                else:
                    roles_copy[i][role] = res
            elif isinstance(role_content, bool):
                pass
            else:
                raise ValueError(f"{role_content}")

    return roles_copy


def rename_arguments(statements: List[dict], progress_bar: bool = False, suffix: str = "_highdim"):

    """
    Takes a list of dictionaries and renames the keys of the dictionary with an extra user-specified suffix.
    Args:
        statements: list of statements
        progress_bar: print a progress bar (default is False)
        suffix: extra suffix to add to the keys of the dictionaries
    Returns:
        List of dictionaries with renamed keys.
    """

    roles_copy = deepcopy(statements)

    if progress_bar:
        print("Processing raw arguments...")
        time.sleep(1)
        statements = tqdm(statements)

    for i, statement in enumerate(statements):
        for role, role_content in statement.items():
            name = role + suffix
            roles_copy[i][name] = roles_copy[i].pop(role)
            roles_copy[i][name] = role_content

    return roles_copy


if __name__ == '__main__':
    sdp = SyntacticDP()
    print(os.path.dirname(os.path.dirname(__file__)))
    with open(os.path.dirname(os.path.dirname(__file__))+'/persian-sentences.txt', 'r') as ps:
        file = ps.read().splitlines()

    while True:
        try: file.remove(' ')
        except: break

    while True:
        try: file.remove('')
        except: break

    df = pd.DataFrame(file)
    df['sdp'] = df.apply(lambda item: sdp([item][0]))
    df.to_csv(os.path.dirname(os.path.dirname(__file__)) + '/persian-sentences.csv')