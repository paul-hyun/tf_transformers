# -*- coding:utf-8 -*-

import argparse
import os
import tarfile
import zipfile

import wget


def dump_txt(data_dir, file_list):
    """
    text로 저장
    :param data_dir: download dir
    :param file_list: file 목록
    """
    # 한국어 corpus 생성
    ko_corpus = os.path.join(data_dir, "corpus.ko.txt")
    with open(ko_corpus, "w") as o_f:
        for filename in file_list:
            if filename.endswith(".ko"):
                with open(filename) as i_f:
                    for line in i_f:
                        o_f.write(line.lower())
    # 영어 corpus 생성
    en_corpus = os.path.join(data_dir, "corpus.en.txt")
    with open(en_corpus, "w") as o_f:
        for filename in file_list:
            if filename.endswith(".en"):
                with open(filename) as i_f:
                    for line in i_f:
                        o_f.write(line.lower())
    return ko_corpus, en_corpus


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


def download_korean_english_news(data_dir):
    """
    dowload korean-english-park train, dev, test
    :param data_dir: download dir
    :return train_file: download train filenames
    :return dev_file: download dev filenames
    :return test_file: download test filenames
    """
    # train 파일이 없을 경우만 다운로드
    train_file = os.path.join(data_dir, "korean-english-park.train.tar.gz")
    if not os.path.exists(train_file):
        print(f"download {train_file} ...")
        wget.download("https://github.com/jungyeul/korean-parallel-corpora/raw/master/korean-english-news-v1/korean-english-park.train.tar.gz", train_file)
        print()

    # dev 파일이 없을 경우만 다운로드
    dev_file = os.path.join(data_dir, "korean-english-park.dev.tar.gz")
    if not os.path.exists(dev_file):
        print(f"download {dev_file} ...")
        wget.download("https://github.com/jungyeul/korean-parallel-corpora/raw/master/korean-english-news-v1/korean-english-park.dev.tar.gz", dev_file)
        print()

    # test 파일이 없을 경우만 다운로드
    test_file = os.path.join(data_dir, "korean-english-park.test.tar.gz")
    if not os.path.exists(test_file):
        print(f"download {test_file} ...")
        wget.download("https://github.com/jungyeul/korean-parallel-corpora/raw/master/korean-english-news-v1/korean-english-park.test.tar.gz", test_file)
        print()

    return train_file, dev_file, test_file


def main(args):
    """
    main function
    :param args: input arguments
    """
    # make kowiki dir
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        print(f"make korean english news data dir: {args.data_dir}")

    # download latest kowiki dump
    train_file, dev_file, test_file = download_korean_english_news(args.data_dir)

    # extract tar.gz files
    train_list = extract_tar(args.data_dir, train_file)
    dev_list = extract_tar(args.data_dir, dev_file)
    test_list = extract_tar(args.data_dir, test_file)

    # 파일 목록 합침
    file_list = []
    file_list.extend(train_list)
    file_list.extend(dev_list)
    file_list.extend(test_list)

    # dump text
    ko_corpus, en_corpus = dump_txt(args.data_dir, file_list)

    # zip
    with zipfile.ZipFile(os.path.join(args.data_dir, "corpus.txt.zip"), "w") as z:
        z.write(ko_corpus, "corpus.ko.txt")
        z.write(en_corpus, "corpus.en.txt")

    # file 삭제
    os.remove(ko_corpus)
    os.remove(en_corpus)
    for filename in file_list:
        os.remove(filename)


def parse_args():
    """
    build arguments
    :return args: input arguments
    """
    parser = argparse.ArgumentParser(description="Prepare korean english news arguments.")
    parser.add_argument("--data_dir", type=str, default="korean_english_news", required=False, help="korean english news data directory")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
