# -*- coding:utf-8 -*-


def encode_mecab(tagger, string):
    """
    string을 mecab을 이용해서 형태소 분석
    :param tagger: 형태소 분석기 객체
    :param string: input text
    :return tokens: 형태소 분석 결과
    :return indexs: 띄어쓰기 위치
    """
    string = string.strip()
    if len(string) == 0:
        return [], []
    words = string.split()
    nodes = tagger.pos(" ".join(words))

    tokens = []
    for node in nodes:
        surface = node[0].strip()
        if 0 < len(surface):
            for s in surface.split():  # mecab 출력 중 '영치기 영차' 처리
                tokens.append(s)

    indexs = []
    index, start, end = -1, 0, 100000
    for i, token in enumerate(tokens):  # 분류가 잘 되었는지 검증
        if end < len(words[index]):
            start = end
            end += len(token)
        else:
            index += 1
            start = 0
            end = len(token)
            indexs.append(i)  # values 중 실제 시작 위치 기록
        assert words[index][start:end] == token, f"{words[index][start:end]} != {token}"

    return tokens, indexs


def decode_mecab(tokens, indexs):
    """
    형태소 분석된 문장을 원래 형태로 변경
    :param tokens: 형태소 tokens
    :param indexs: 띄어쓰기 위치
    :return: 형태소 분석 이전 문장
    """
    values = []
    for i, token in enumerate(tokens):
        if i in indexs and 0 < i:
            values.append(" ")
        values.append(token)
    return "".join(values)
