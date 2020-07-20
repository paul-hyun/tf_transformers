# -*- coding:utf-8 -*-

import argparse
import json
import os
import tarfile
import zipfile

import sentencepiece as spm
import yaml


def dump_json(vocab_ko, vocab_en, file_list, output):
    """
    json 형태로 ko, en 파일 저장
    :param vocab_ko: korean vocab
    :param vocab_en: english vocab
    :param file_list: korean, english file list
    :param output: output json filename
    """
    # korean, english line 목록 저장
    ko_lines, en_lines = [], []
    for filename in file_list:
        if filename.endswith(".ko"):
            with open(filename) as f:
                ko_lines.extend(f.readlines())
        if filename.endswith(".en"):
            with open(filename) as f:
                en_lines.extend(f.readlines())
    assert len(ko_lines) == len(en_lines)

    filename, ext = os.path.splitext(output)
    with open(f"{filename}", "w") as f:
        for ko, en in zip(ko_lines, en_lines):
            f.write(json.dumps({"ko": vocab_ko.encode_as_pieces(ko.lower()), "en": vocab_en.encode_as_pieces(en.lower())}, ensure_ascii=False))
            f.write("\n")

    # zip
    with zipfile.ZipFile(output, "w") as z:
        z.write(filename, os.path.basename(filename))

    # file 삭제
    os.remove(filename)


def extract_tar(data_dir, filename):
    """
    tar.gz 압축파일 풀기
    :param data_dir: download dir
    :param filename: download filename
    :return names: tar file 목록
    """
    with tarfile.open(filename) as t:
        t.extractall(data_dir)
        names = t.getnames()
        for i in range(len(names)):
            names[i] = os.path.join(data_dir, names[i])
    return names


def main(args):
    """
    main function
    :param args: input arguments
    """
    # load config
    with open(args.config) as f:
        config = yaml.load(f)

    # vocab loading
    vocab_ko = spm.SentencePieceProcessor()
    vocab_ko.load(os.path.join(args.data_dir, config["vocab"]["ko"]))
    vocab_en = spm.SentencePieceProcessor()
    vocab_en.load(os.path.join(args.data_dir, config["vocab"]["en"]))

    # extract tar.gz files
    train_list = extract_tar(args.data_dir, os.path.join(args.data_dir, "korean-english-park.train.tar.gz"))
    dev_list = extract_tar(args.data_dir, os.path.join(args.data_dir, "korean-english-park.dev.tar.gz"))
    test_list = extract_tar(args.data_dir, os.path.join(args.data_dir, "korean-english-park.test.tar.gz"))

    # json dump
    dump_json(vocab_ko, vocab_en, train_list, os.path.join(args.data_dir, config["data"]["train"]))
    dump_json(vocab_ko, vocab_en, dev_list, os.path.join(args.data_dir, config["data"]["dev"]))
    dump_json(vocab_ko, vocab_en, test_list, os.path.join(args.data_dir, config["data"]["test"]))

    # 파일 목록 합침
    file_list = []
    file_list.extend(train_list)
    file_list.extend(dev_list)
    file_list.extend(test_list)

    for filename in file_list:
        os.remove(filename)


def parse_args():
    """
    build arguments
    :return args: input arguments
    """
    parser = argparse.ArgumentParser(description="Sentencepeice vocab korean_english_news arguments.")
    parser.add_argument("--config", type=str, default="config/korean_english_news_ko_en_32000.yaml", required=False, help="configuration file")
    parser.add_argument("--data_dir", type=str, default="../data/korean_english_news", required=False, help="korean english news data directory")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
