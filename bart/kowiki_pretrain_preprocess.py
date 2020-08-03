# -*- coding:utf-8 -*-

import argparse
import copy
import json
import os
import random
import zipfile

import numpy as np
import sentencepiece as spm
import yaml
from tqdm import tqdm

SPAN_LEN = 10
SPAN_VALUE = np.array([i for i in range(SPAN_LEN)])
SPAN_RATIO = np.random.poisson(lam=3, size=SPAN_LEN).astype(np.float)
SPAN_RATIO /= np.sum(SPAN_RATIO)


def create_pretrain_infill(tokens, mask_cnt):
    """
    Infilling 생성
    :param tokens: tokens
    :param mask_cnt: mask 개수 (전체 tokens의 15%)
    :return infilled: mask된 tokens
    """
    # 단어 단위로 mask 하기 위해서 index 분할
    cand_idx = {}
    index = 0
    for (i, token) in enumerate(tokens):
        if token == "[BOS]" or token == "[EOS]":
            continue
        if 0 < len(cand_idx) and not token.startswith(u"\u2581"):
            cand_idx[index].append(i)
        else:
            index += 1
            cand_idx[index] = [i]
    keys = list(cand_idx.keys())
    # random mask를 위해서 순서를 섞음
    random.shuffle(keys)

    mask_lms = []  # mask 된 값
    covered_idx = set()  # mask 및 주변 값
    mask_indexs = {}  # [MASK] token이 들어갈 위치
    for index in keys:
        if len(mask_lms) >= mask_cnt:  # 핸재 mask된 개수가 15%를 넘으면 중지
            break
        # span 길이 선택
        span_len = random.choices(SPAN_VALUE, SPAN_RATIO)[0]
        span_len = min(span_len, len(cand_idx) - index)
        # mask할 indexs
        index_set = []
        for i in range(span_len):
            index_set.extend(cand_idx[index + i])
        mask_index = cand_idx[index][0]
        if len(mask_lms) + len(index_set) > mask_cnt:  # 이번에 mask할 개수를 포함해 15%를 넘으면 skip
            continue
        # mask가 이미 되었는지 확인 (주변 boundary 포함)
        is_idx_covered = False
        for index in index_set:
            if index in covered_idx:
                is_idx_covered = True
                break
        # 이미 처리 된 경우는 skip
        if is_idx_covered:
            continue

        mask_indexs[mask_index] = span_len
        for index in index_set:
            covered_idx.add(index)
            mask_lms.append(index)
            tokens[index] = None

        # span boundary
        if index_set:
            covered_idx.add(index_set[0] - 1)
            covered_idx.add(index_set[-1] + 1)
        else:
            covered_idx.add(mask_index)
            covered_idx.add(mask_index + 1)
            covered_idx.add(mask_index - 1)

    infilled = []
    count = 0
    for token in tokens:
        if token is not None:
            infilled.append(token)
        count += 1
        if count in mask_indexs:
            infilled.append(f"[MASK]")

    return infilled


def create_pretrain_rotate(tokens):
    """
    마스크 생성
    :param tokens: tokens
    :return tokens: rotated된 tokens
    """
    # 단어 단위로 mask 하기 위해서 index 분할
    cand_idx = []  # word 단위의 index array
    for (i, token) in enumerate(tokens):
        if token == "[BOS]" or token == "[EOS]":
            continue
        if token.startswith(u"\u2581"):
            cand_idx.append(i)

    if 1 < len(cand_idx):
        # 처음 index를 제외한 나머지에서 랜덤 선택
        index = random.choice(cand_idx[1:])
        # 순서 변경
        rotated = tokens[index:] + tokens[:index]
        assert len(tokens) == len(rotated)
    else:  # 단어가 1개인 경우는 rotate 불가
        rotated = tokens

    return rotated


def create_pretrain_permute(tokens, line_index):
    """
    Permuted 생성
    :param tokens: tokens
    :param mask_cnt: mask 개수 (전체 tokens의 15%)
    :return tokens: mask된 tokens
    """
    if 1 < len(line_index):
        # line 단위로 token 분리
        lines = []
        for i in range(len(line_index)):
            lines.append(tokens[line_index[i - 1] if 0 < i else 0:line_index[i]])
        # 순서 shuffle
        random.shuffle(lines)
        # 라인으로 변환
        permuted = []
        for line in lines:
            permuted.extend(line)
        assert len(tokens) == len(permuted)
    else:  # 문장이 1개인 경우는 permte 불가
        permuted = tokens

    return permuted


def create_pretrain_delete(tokens, mask_cnt):
    """
    Deleted 생성
    :param tokens: tokens
    :param mask_cnt: mask 개수 (전체 tokens의 15%)
    :return deleted: deleted된 tokens
    """
    # 단어 단위로 mask 하기 위해서 index 분할
    cand_idx = []  # word 단위의 index array
    for (i, token) in enumerate(tokens):
        if token == "[BOS]" or token == "[EOS]":
            continue
        if 0 < len(cand_idx) and not token.startswith(u"\u2581"):
            cand_idx[-1].append(i)
        else:
            cand_idx.append([i])
    # random mask를 위해서 순서를 섞음
    random.shuffle(cand_idx)

    mask_lms = []  # mask 된 값
    for index_set in cand_idx:
        if len(mask_lms) >= mask_cnt:  # 핸재 mask된 개수가 15%를 넘으면 중지
            break
        if len(mask_lms) + len(index_set) > mask_cnt:  # 이번에 mask할 개수를 포함해 15%를 넘으면 skip
            continue
        for index in index_set:
            mask_lms.append({"index": index, "label": tokens[index]})
            tokens[index] = None

    deleted = [token for token in tokens if token is not None]
    return deleted


def create_pretrain_mask(tokens, mask_cnt, vocab_list):
    """
    Masked 생성
    :param tokens: tokens
    :param mask_cnt: mask 개수 (전체 tokens의 15%)
    :param vocab_list: vocab list (random token 용)
    :return tokens: mask된 tokens
    """
    # 단어 단위로 mask 하기 위해서 index 분할
    cand_idx = []  # word 단위의 index array
    for (i, token) in enumerate(tokens):
        if token == "[BOS]" or token == "[EOS]":
            continue
        if 0 < len(cand_idx) and not token.startswith(u"\u2581"):
            cand_idx[-1].append(i)
        else:
            cand_idx.append([i])
    # random mask를 위해서 순서를 섞음
    random.shuffle(cand_idx)

    mask_lms = []  # mask 된 값
    for index_set in cand_idx:
        if len(mask_lms) >= mask_cnt:  # 핸재 mask된 개수가 15%를 넘으면 중지
            break
        if len(mask_lms) + len(index_set) > mask_cnt:  # 이번에 mask할 개수를 포함해 15%를 넘으면 skip
            continue
        dice = random.random()  # 0..1 사이의 확률 값
        for index in index_set:
            masked_token = None
            if dice < 0.8:  # 80% replace with [MASK]
                masked_token = "[MASK]"
            elif dice < 0.9:  # 10% keep original
                masked_token = tokens[index]
            else:  # 10% random word
                masked_token = random.choice(vocab_list)
            mask_lms.append({"index": index, "label": tokens[index]})
            tokens[index] = masked_token

    return tokens


def create_pretrain_instances(doc, n_seq, mask_prob, vocab_list):
    """
    pre-train instance 생성
    :param doc: 문장
    :param n_seq: 최대 sequence
    :param mask_prob: mask probability
    :param vocab_list: vocab list for random
    :return:
    """
    # for [BOS], [EOS]
    max_seq = n_seq - 2

    instances = []
    current_chunk = []
    current_length = 0
    line_index = []
    for i in range(len(doc)):
        current_chunk.extend(doc[i])  # line
        current_length += len(doc[i])
        line_index.append(current_length)
        if i == len(doc) - 1 or current_length >= max_seq:
            tokens = current_chunk[:max_seq]
            masked = create_pretrain_mask(copy.deepcopy(tokens), int(len(tokens) * mask_prob), vocab_list)
            deleted = create_pretrain_delete(copy.deepcopy(tokens), int(len(tokens) * mask_prob))
            permuted = create_pretrain_permute(copy.deepcopy(tokens), line_index)
            rotated = create_pretrain_rotate(copy.deepcopy(tokens))
            infilled = create_pretrain_infill(copy.deepcopy(tokens), int(len(tokens) * mask_prob))
            instance = {
                "tokens": tokens,
                "mask": masked,
                "delete": deleted,
                "permute": permuted,
                "rotate": rotated,
                "infill": infilled
            }
            instances.append(instance)

            current_chunk = []
            current_length = 0
            line_index = []

    return instances


def dump_json(vocab, in_file, out_file, n_seq, mask_prov):
    """
    json 형태로 pretrain data 파일 저장
    :param vocab: vocab
    :param in_file: kowiki corpus filename
    :param out_file: output json filename
    :param n_seq: number of sequence
    :param mask_prov: mask probability
    """

    def save_pretrain_instances(out_f, doc):
        instances = create_pretrain_instances(doc, n_seq, mask_prov, vocab_list)
        for instance in instances:
            out_f.write(json.dumps(instance, ensure_ascii=False))
            out_f.write("\n")

    # 특수 token 7개를 제외한 나머지 tokens 들
    vocab_list = []
    for id in range(7, len(vocab)):
        if not vocab.is_unknown(id):
            vocab_list.append(vocab.id_to_piece(id))

    with zipfile.ZipFile(in_file) as z:
        in_basename, ext = os.path.splitext(os.path.basename(in_file))
        total = 0
        with z.open(in_basename) as in_f:
            for line in in_f:
                total += 1

        out_filename, ext = os.path.splitext(out_file)
        with open(out_filename, "w") as out_f:
            with z.open(in_basename) as in_f:
                doc = []
                for line in tqdm(in_f, total=total):
                    line = line.strip().decode("utf-8")
                    if line == "":
                        if doc:
                            save_pretrain_instances(out_f, doc)
                            doc = []
                    else:
                        doc.append(vocab.encode_as_pieces(line))
                # 마지막에 처리 되지 않은 경우
                if doc:
                    save_pretrain_instances(out_f, doc)
                    doc = []
        # zip
        if os.path.exists(out_file):
            os.remove(out_file)
        with zipfile.ZipFile(out_file, "w") as z:
            z.write(out_filename, os.path.basename(out_filename))
        # file 삭제
        os.remove(out_filename)


def main(args):
    """
    main function
    :param args: input arguments
    """
    # load config
    with open(args.config) as f:
        config = yaml.load(f)

    print(config)

    # vocab loading
    vocab = spm.SentencePieceProcessor()
    vocab.load(os.path.join(args.kowiki_dir, config["vocab"]))

    # json dump
    dump_json(vocab, os.path.join(args.kowiki_dir, config["data"]["corpus"]), os.path.join(args.kowiki_dir, config["data"]["pre_train"]), config["model"]["n_seq"], config["data"]["mask_prob"])


def parse_args():
    """
    build arguments
    :return args: input arguments
    """
    parser = argparse.ArgumentParser(description="pre-train pre-processing kowiki arguments.")
    parser.add_argument("--config", type=str, default="config/pretrain_kowiki_finetune_nsmc.yaml", required=False, help="configuration file")
    parser.add_argument("--kowiki_dir", type=str, default="../data/kowiki", required=False, help="kowiki data directory")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
