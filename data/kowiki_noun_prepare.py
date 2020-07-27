# -*- coding:utf-8 -*-

import argparse
import os
import sys
import zipfile

import konlpy
from tqdm import tqdm

sys.path.append("..")

from common.noun_splitter import NounSplitter


def noun_corpus(args):
    """
    KB noun_spliter corpus 작성 (https://github.com/KB-Bank-AI/KB-ALBERT-KO/blob/master/src/preprocessing/noun_splitter.py)
    :param args: input arguments
    :return: 결과 파일
    """
    noun_splitter = NounSplitter("../common/np2.crfsuite")
    output = os.path.join(args.data_dir, f"kowiki_noun.txt")
    with zipfile.ZipFile(f"{args.data_dir}/{args.zip}") as z:
        total = 0
        with z.open(args.txt) as i_f:
            for _, _ in enumerate(i_f):
                total += 1
        with z.open(args.txt) as i_f:
            with open(output, "w") as o_f:
                for i, line in enumerate(tqdm(i_f, total=total, desc=f"noun")):
                    line = line.strip().decode("UTF-8", "ignore")
                    if line:
                        line = noun_splitter.do_split(line)
                        o_f.write(line)
                    o_f.write("\n")

    return output


def main(args):
    """
    main function
    :param args: input arguments
    """
    output = noun_corpus(args)
    basename = os.path.basename(output)

    # zip
    with zipfile.ZipFile(os.path.join(args.data_dir, f"{basename}.zip"), "w") as z:
        z.write(output, os.path.basename(output))

    os.remove(output)


def parse_args():
    """
    build arguments
    :return args: input arguments
    """
    parser = argparse.ArgumentParser(description="Make mecab corpus arguments.")
    parser.add_argument("--data_dir", type=str, default="kowiki", required=False, help="kowiki data directory")
    parser.add_argument("--zip", type=str, default="kowiki.txt.zip", required=False, help="kowiki source zip file")
    parser.add_argument("--txt", type=str, default="kowiki.txt", required=False, help="kowiki source txt file")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
