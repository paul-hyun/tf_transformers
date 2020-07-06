# -*- coding:utf-8 -*-

import MeCab

m = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ko-dic")


def encode_mecab(string):
    """
    string을 mecab을 이용해서 형태소 분석
    :param string: 형태소 분석을 할 문자열
    :return tokens: 형태소 단위의 tokens
    :return indexs: 띄어쓰기 위치 (원래 문장 복원 용)
    """
    string = string.strip()
    if len(string) == 0:
        return [], []
    words = string.split()
    node = m.parseToNode(" ".join(words))
    tokens = []
    while node:
        feature = node.feature.split(",")
        if feature[0] == "BOS/EOS":  # BOS/EOS 는 사용안함
            node = node.next
            continue
        try:
            surface = node.surface
            if surface == "" and feature[3] != "*":  # surface가 없는 경우는 feature갑 사용
                surface = feature[3]
        except:
            assert feature[3] != "*"  # error 가 발생한 경우는 feature갑 사용
            surface = feature[3]

        surface = surface.strip()
        if 0 < len(surface):
            for s in surface.split():  # mecab 출력 중 '영치기 영차' 처리
                tokens.append(s)
        node = node.next

    indexs = []
    index, start, end = -1, 0, 100000
    for i, value in enumerate(tokens):  # 분류가 잘 되었는지 검증
        if end < len(words[index]):
            start = end
            end += len(value)
        else:
            index += 1
            start = 0
            end = len(value)
            indexs.append(i)  # values 중 실제 시작 위치 기록
        assert words[index][start:end] == value, f"{words[index][start:end]} != {value}"

    return tokens, indexs


def decode_mecab(tokens, indexs):
    """
    형태소 분석된 문장을 원래 형태로 변경
    :param tokens: 형태소 단위 tokens
    :param indexs: 띄어쓰기 위치
    :return string: 복원된 원래 문장
    """
    values = []
    for i, token in enumerate(tokens):
        if i in indexs and 0 < i:
            values.append(" ")
        values.append(token)
    string = "".join(values)
    return string
