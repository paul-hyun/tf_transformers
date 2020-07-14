# -*- coding:utf-8 -*-

import argparse
import datetime
import json
import os
import re
import sys
import zipfile
import shutil

import wget
from tqdm import tqdm

sys.path.append("..")


def dump_txt(data_list, output):
    """
    wiki를 text로 저장
    :param data_list: wiki json 목록
    :param output: 저장할 파일
    """
    with open(output, "w") as f:
        for data in tqdm(data_list, desc="dump text"):
            f.write(data["text"])
            f.write("\n\n\n\n")


def trim_wiki(wiki_list):
    """
    wiki list를 json으로 읽어 옴.
    new line이 2개 이상인 경우는 모두 new line 1개로 변경
    :param wiki_list: wiki 파일 목록
    :return data_list: wiki json 목록
    """
    data_list = []
    for wiki in tqdm(wiki_list, desc="trim wiki"):
        with open(wiki) as f:
            for line in f:
                data = json.loads(line)
                # new line이 2개 이상인 경우를 new line 1개로 변경
                data["text"] = re.sub("\n+", "\n", data["text"])
                data_list.append(data)
    return data_list


def exec_WikiExtractor(data_dir, filename):
    """
    WikiExtractor 실행
    :param data_dir: download dir
    :param filename: kowiki download filename
    :return wiki_list: wiki file 목록
    """
    # WikiExtractor.py가 없을 경우 다운로드
    if not os.path.exists("WikiExtractor.py"):
        print(f"download WikiExtractor.py ...")
        wget.download("https://raw.githubusercontent.com/attardi/wikiextractor/master/WikiExtractor.py", "WikiExtractor.py")

    output = os.path.join(data_dir, f"json-{datetime.date.today().isoformat()}")
    # 폴더가 있으면 생성 된 것으로 판단. 폴더가 없는 경우 WikiExtractor 실행
    if not os.path.exists(output):
        print(f"exec WikiExtractor input: {filename}, output: {output} ...")
        os.system(f"python WikiExtractor.py -o {output} --json {filename}")

    # output dir의 wiki 목록 조회
    wiki_list = []
    for x in os.walk(output):
        # x[0]: path, x[1]: dirs, x[2]: files
        path = x[0]
        for f in x[2]:
            find = re.findall(r"wiki_[0-9][0-9]", f)
            if find:
                wiki_list.append(os.path.join(path, f))
    return wiki_list, output


def download_kowiki(data_dir):
    """
    dowload latest kowiki dump file
    :param data_dir: download dir
    :return filename: download filename
    """
    filename = os.path.join(data_dir, f"kowiki-pages-meta-current-{datetime.date.today().isoformat()}.xml.bz2")
    # 파일이 없을 경우만 다운로드
    if not os.path.exists(filename):
        print(f"download {filename} ...")
        wget.download("https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-meta-current.xml.bz2", filename)
    return filename


def main(args):
    """
    main function
    :param args: input arguments
    """
    # make kowiki dir
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        print(f"make kowiki data dir: {args.data_dir}")

    # download latest kowiki dump
    filename = download_kowiki(args.data_dir)

    # WikiExtractor 실행
    wiki_list, wiki_out = exec_WikiExtractor(args.data_dir, filename)

    # witk multi new line 제거
    data_list = trim_wiki(wiki_list)

    # wiki를 txt 파일로 저장
    wiki_txt = os.path.join(args.data_dir, "kowiki.txt")
    dump_txt(data_list, wiki_txt)

    # zip
    with zipfile.ZipFile(os.path.join(args.data_dir, "kowiki.txt.zip"), "w") as z:
        z.write(wiki_txt, "kowiki.txt")

    os.remove(filename)
    shutil.rmtree(wiki_out)
    os.remove(wiki_txt)


def parse_args():
    """
    build arguments
    :return args: input arguments
    """
    parser = argparse.ArgumentParser(description="Prepare kowiki arguments.")
    parser.add_argument("--data_dir", type=str, default="kowiki", required=False, help="kowiki data directory")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
