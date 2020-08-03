# -*- coding:utf-8 -*-

import argparse
import json
import os
import sys
import zipfile

import konlpy
import pandas as pd
import sentencepiece as spm
import yaml
from tqdm import tqdm

sys.path.append("..")

from common.mecab import encode_mecab
from common.noun_splitter import NounSplitter

mecab = konlpy.tag.Mecab()
noun = NounSplitter("../common/np2.crfsuite")


def mecab_tagger(string):
    """
    mecab tokenize
    :param string: string
    :return: mecab tokenized string
    """
    tokens, _ = encode_mecab(mecab, string)
    return " ".join(tokens)


def noun_tagger(string):
    """
    noun splitter tokenize
    :param string: string
    :return: noun tokenized string
    """
    return noun.do_split(string)


def dump_json(vocab, data, out_file, tagger=None):
    """
    json 형태로 pretrain data 파일 저장
    :param vocab: vocab
    :param data: input data frame
    :param out_file: output json filename
    :param tagger: tagger
    """
    basename, _ = os.path.splitext(out_file)

    with open(basename, "w") as f:
        for i, row in tqdm(data.iterrows(), total=len(data), desc=f"{os.path.basename(basename)}"):
            document = row["document"]
            if type(document) == str:  # document가 str인 경우만 처리
                if tagger:
                    document = tagger(document)
                f.write(json.dumps({"tokens": vocab.encode_as_pieces(document), "label": row["label"]}, ensure_ascii=False))  # peice 저장
                f.write("\n")

    with zipfile.ZipFile(out_file, "w") as z:
        z.write(basename, os.path.basename(basename))
    # file 삭제
    os.remove(basename)


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

    with zipfile.ZipFile(os.path.join(args.nsmc_dir, args.zip)) as z:
        with z.open("train.txt") as f:
            train = pd.read_csv(f, header=0, delimiter="\t")
        with z.open("test.txt") as f:
            test = pd.read_csv(f, header=0, delimiter="\t")

    if "mecab" in config["vocab"]:
        tagger = mecab_tagger
    elif "noun" in config["vocab"]:
        tagger = noun_tagger
    else:
        tagger = None

    # json dump
    dump_json(vocab, train, os.path.join(args.nsmc_dir, config["data"]["nsmc_train"]), tagger)
    # json dump
    dump_json(vocab, test, os.path.join(args.nsmc_dir, config["data"]["nsmc_test"]), tagger)


def parse_args():
    """
    build arguments
    :return args: input arguments
    """
    parser = argparse.ArgumentParser(description="pre-train pre-processing kowiki arguments.")
    parser.add_argument("--config", type=str, default="config/pretrain_kowiki_finetune_nsmc.yaml", required=False, help="configuration file")
    parser.add_argument("--kowiki_dir", type=str, default="../data/kowiki", required=False, help="kowiki data directory")
    parser.add_argument("--nsmc_dir", type=str, default="../data/nsmc", required=False, help="nsmc data directory")
    parser.add_argument("--zip", type=str, default="nsmc.zip", required=False, help="nsmc zip file")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    main(args)
