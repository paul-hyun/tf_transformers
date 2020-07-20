# -*- coding:utf-8 -*-

import json
import os

import numpy as np
from tqdm import tqdm
import zipfile


def load_korean_english_news(vocab_ko, vocab_en, filename, n_seq, src="ko", dst="en", i_pad=0):
    """
    korean_english_news 데이터를 로드
    :param vocab_ko: korean vocab
    :param vocab_en: english vocab
    :param filename: 전처리된 json 파일
    :param n_seq: 시퀀스 길이 (number of sequence)
    :param src: source language
    :param dst: target language
    :param i_pad: pad index
    :return enc_tokens: encoder input tokens
    :return dec_tokens: decoder input tokens
    :return labels: labels
    """
    vocab_src = vocab_ko if src == "ko" else vocab_en
    vocab_dst = vocab_en if dst == "en" else vocab_ko

    # 압축파일 열기
    with zipfile.ZipFile(filename) as z:
        basename = os.path.basename(filename)
        jsonfile, ext = os.path.splitext(basename)

        # 전체 입력 개수 계산
        total = 0
        with z.open(jsonfile, "r") as f:
            for line in f:
                total += 1

        # 데이터 미리 생성
        enc_tokens = np.zeros((total, n_seq), np.int)
        dec_tokens = np.zeros((total, n_seq ), np.int)
        labels = np.zeros((total, n_seq), np.int)

        # 라인단위로 데이터 생성
        with z.open(jsonfile, "r") as f:
            for i, line in enumerate(tqdm(f, total=total, desc=f"{jsonfile}")):
                data = json.loads(line.decode("utf-8"))

                # qestion을 id 형태로 변경
                src_id = [vocab_src.piece_to_id(p) for p in data[src]]
                # answer를 id 형태로 변경
                dst_id = [vocab_dst.piece_to_id(p) for p in data[dst]]

                # enc_token: <src tokens>, [PAD] tokens
                enc_token = src_id[:n_seq] + [i_pad] * (n_seq - len(src_id))
                # dec_token: [BOS], <dst tokens>, [PAD] tokens
                dec_token = [vocab_dst.bos_id()] + dst_id[:n_seq - 1] + [0] * (n_seq - len(dst_id) - 1)
                # label: <dst tokens>, [EOS], [PAD] tokens
                label = dst_id[:n_seq - 1] + [vocab_dst.eos_id()] + [0] * (n_seq - len(dst_id) - 1)

                assert len(enc_token) == len(dec_token) == len(label) == n_seq

                enc_tokens[i] = enc_token
                dec_tokens[i] = dec_token
                labels[i] = label

    return (enc_tokens, dec_tokens), labels
