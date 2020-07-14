# -*- coding:utf-8 -*-

import argparse
import collections
import zipfile

import konlpy
import pandas as pd
from tqdm import tqdm


def word_count(args, name, tagger):
    """
    형태소 분석기 별 word count 작성
    :param args: input arguments
    :param name: 형태소 분석기 이름
    :param tagger: 형태소 분석기 객체
    :param tags: count 계산할 tags
    :return: 형태소 분석 pandas dataframe
    """
    # tag 별 counter
    counter = {}

    with zipfile.ZipFile(f"{args.data_dir}/{args.zip}") as z:
        total = 0
        with z.open(args.txt) as f:
            for _, _ in enumerate(f):
                total += 1
        with z.open(args.txt) as f:
            for i, line in enumerate(tqdm(f, total=total, desc=f"{name}")):
                line = line.strip().decode("UTF-8", "ignore")
                line = " ".join(line.split())
                if line:
                    try:
                        tokens = tagger.pos(line)
                        for token in tokens:
                            word, tag = token
                            if tag not in counter:
                                counter[tag] = collections.defaultdict(int)
                            counter[tag][word] += 1
                    except Exception as e:
                        print(line)
                        print(e)
                else:
                    pass
                    # break

    # 형태소 count를 csv로 저장
    word_list = []
    for tag, value in counter.items():
        for word, cnt in value.items():
            word_list.append({"tag": tag, "word": word, "cnt": cnt})

    df = pd.DataFrame(data=word_list)
    df.to_csv(f"{args.data_dir}/{name}_word_count.csv.zip", compression="zip")
    return df


def main(args):
    """
    main function
    :param args: input arguments
    """
    word_count(args, "mecab", konlpy.tag.Mecab())
    word_count(args, "komoran", konlpy.tag.Komoran())
    word_count(args, "okt", konlpy.tag.Okt())
    # 시간이 너무 오래 걸림 (제외)
    # word_count(args, "kkma", konlpy.tag.Kkma())


def parse_args():
    """
    build arguments
    :return args: input arguments
    """
    parser = argparse.ArgumentParser(description="Make word count arguments.")
    parser.add_argument("--data_dir", type=str, default="kowiki", required=False, help="kowiki data directory")
    parser.add_argument("--zip", type=str, default="kowiki.txt.zip", required=False, help="kowiki source zip file")
    parser.add_argument("--txt", type=str, default="kowiki.txt", required=False, help="kowiki source txt file")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
