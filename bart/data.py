# -*- coding:utf-8 -*-

import json
import os
import zipfile

import numpy as np
from tqdm import tqdm


def load_kowiki_pretrain(vocab, filename, n_seq, noise_type, i_pad=0):
    """
    kowiki pretrain data load
    :param vocab: 을
    :param filename: 전처리된 json 파일
    :param n_seq: 시퀀스 길이 (number of sequence)
    :param noise_type: 노이즈 타입 [mask, delete, permute, rotate, infill]
    :param i_pad: pad index
    :return enc_tokens: encoder input tokens
    :return dec_tokens: decoder input tokens
    :return labels: labels
    """
    # for [BOS], [EOS]
    max_seq = n_seq - 2

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
        dec_tokens = np.zeros((total, n_seq), np.int)
        labels = np.zeros((total, n_seq), np.int)

        # 라인단위로 데이터 생성
        with z.open(jsonfile, "r") as f:
            for i, line in enumerate(tqdm(f, total=total, desc=f"{jsonfile}")):
                data = json.loads(line.decode("utf-8"))

                # token id 형태로 변경
                token_id = [vocab.piece_to_id(p) for p in data["tokens"]]
                # noise id 형태로 변경
                noise_id = [vocab.piece_to_id(p) for p in data[noise_type]]

                # enc_token: [BOS] <noise_id>, [EOS], [PAD]*
                enc_token = [vocab.bos_id()] + noise_id[:max_seq] + [vocab.eos_id()] + [i_pad] * (n_seq - min(max_seq, len(noise_id)) - 2)
                # dec_token: [BOS] <token_id>, [EOS], [PAD]*
                dec_token = [vocab.bos_id()] + token_id[:max_seq] + [vocab.eos_id()] + [i_pad] * (n_seq - min(max_seq, len(token_id)) - 2)
                # label: <token_id>, [EOS], [PAD]*
                label = token_id[:max_seq] + [vocab.eos_id()] + [i_pad] * (n_seq - min(max_seq, len(token_id)) - 1)

                assert len(enc_token) == len(dec_token) == len(label) == n_seq

                enc_tokens[i] = enc_token
                dec_tokens[i] = dec_token
                labels[i] = label

    return (enc_tokens, dec_tokens), labels


def load_nsmc(vocab, filename, n_seq, i_pad=0):
    """
    kowiki pretrain data load
    :param vocab: 을
    :param filename: 전처리된 json 파일
    :param n_seq: 시퀀스 길이 (number of sequence)
    :param i_pad: pad index
    :return enc_tokens: encoder input tokens
    :return dec_tokens: decoder input tokens
    :return labels: labels
    """
    # for [BOS], [EOS]
    max_seq = n_seq - 2

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
        dec_tokens = np.zeros((total, n_seq), np.int)
        labels = np.zeros((total,), np.int)

        # 라인단위로 데이터 생성
        with z.open(jsonfile, "r") as f:
            for i, line in enumerate(tqdm(f, total=total, desc=f"{jsonfile}")):
                data = json.loads(line.decode("utf-8"))

                # token id 형태로 변경
                token_id = [vocab.piece_to_id(p) for p in data["tokens"]]
                # label
                label = data["label"]

                # enc_token: [BOS] <token_id>, [EOS], [PAD]*
                enc_token = [vocab.bos_id()] + token_id[:max_seq] + [vocab.eos_id()] + [i_pad] * (n_seq - min(max_seq, len(token_id)) - 2)
                # dec_token: [BOS] <token_id>, [EOS], [PAD]*
                dec_token = [vocab.bos_id()] + token_id[:max_seq] + [vocab.eos_id()] + [i_pad] * (n_seq - min(max_seq, len(token_id)) - 2)

                assert len(enc_token) == len(dec_token) == n_seq

                enc_tokens[i] = enc_token
                dec_tokens[i] = dec_token
                labels[i] = label

    return (enc_tokens, dec_tokens), labels
