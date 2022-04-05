import time
from typing import Dict, List, Optional, Tuple, Union, Generator

from tqdm import tqdm
from crf_pos.pos_tagger import WapitiPosTagger


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

        return list(map(lambda item: sdp2srl_mock(item)[0], list(self.batch_tagger(sentences))))

    def batch_tagger(self, sentences: Union[tqdm, List[Dict[str, str]]]) -> Generator[List[tuple], None, None]:
        for sentence in sentences:
            yield self.tagger[sentence]


def extract_pos_words(pos_tagged: List[Tuple[str, str]], role: str) -> Generator[str, None, None]:
    for item in pos_tagged:
        if item[0] == role: yield item[1]


def find_pos_index(pos_tagged: List[Tuple[str, str]], role: str) -> int:
    try:    return [item[1] for item in pos_tagged].index(role)
    except: return len(pos_tagged)


def find_pos_word(pos_tagged: List[Tuple[str, str]], role: str, index: int) -> str:
    try:    return list(extract_pos_words(pos_tagged, role))[index]
    except: return ''


def pos2srl(pos_tagged: List[Tuple[str, str]]) -> Generator[Tuple[str, str], None, None]:
    keys = {'PRO': 'ARG0', 'V': 'V', 'N': 'ARG1', 'ADJ': 'ARG2',
            'NUM': '', 'CON': '', 'ADV': ''}
    verb_ind = find_pos_index(pos_tagged, 'V')
    for ind, item in enumerate(pos_tagged):
        if item[1] == 'N' and ind < verb_ind-3:     yield item[0], 'ARG0'
        elif item[1] == 'V' and ind > verb_ind+1:   yield item[0], 'ARG1'
        else:
            try:    yield item[0], keys[item[1]]
            except: yield item[0], ''


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


def mock_description(rel_tagged: List[Tuple[str, str]]) -> str:
    return ' '.join([f"[{': '.join(item[::-1])}]" if item[0] != '' else item[1] for item in rel_tagged])


def mock_tags(rel_tagged: List[Tuple[str, str]]) -> Generator[str, None, None]:
    for item in rel_tagged:
        if item[0] == '':       yield 'O'
        else:
            yield f'B-{item[0]}'
            for _ in item[1].split()[1:]:
                if item[0] == '':       yield 'O'
                elif item[0] != 'V':    yield f'I-{item[0]}'
                else:                   yield f'B-{item[0]}'


def sdp2srl_mock(pos_tagged: List[Tuple[str, str]]) -> List[Dict[str, Union[str, List[str]]]]:
    concat_srl_var = list(concat_srl(pos2srl(pos_tagged)))
    return [{'verbs': [{'verb': find_pos_word(concat_srl_var, 'V', 0),
                        'description': mock_description(concat_srl_var),
                        'tags': list(mock_tags(concat_srl_var))}],
             'words': [item[0] for item in pos_tagged]}]


if __name__ == '__main__':
    sdp = SyntacticDP()
    tagger = WapitiPosTagger()
    print(tagger['به خدا قسم میخورم این فریمرک کار میکند'])
    pos = sdp(['شروع کار کلاب صیانت از ۴ میاشد'])
    print(pos)
    print(sdp(['من به مدرسه میروم', 'به گزارش افق پیشبینی حاکی از افت سهام نیویورک است']))
    pos = sdp(['به گزارش افق پیشبینی حاکی از افت سهام نیویورک است'])
    print(pos)

    real_world_test_case = "دانشجویان کمک دانشجویان باشید قبال روابط ایران چین حمایت دانشجویان خاهش می کند"
    print(tagger[real_world_test_case])
    pos = sdp([real_world_test_case])
    print(pos)