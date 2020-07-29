# -*- coding:utf-8 -*-

import argparse
import os
import zipfile

import wget


def download_msmc(data_dir):
    """
    dowload latest kowiki dump file
    :param data_dir: download dir
    :return train_file: download train filename
    :return train_file: download test filename
    """
    train_file = os.path.join(data_dir, f"ratings_train.txt")
    print(f"download {train_file} ...")
    wget.download("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", train_file)

    test_file = os.path.join(data_dir, f"ratings_test.txt")
    print(f"download {test_file} ...")
    wget.download("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", test_file)

    return train_file, test_file


def main(args):
    """
    main function
    :param args: input arguments
    """
    # make msmc dir
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
        print(f"make msmc data dir: {args.data_dir}")

    # download msmc
    train_file, test_file = download_msmc(args.data_dir)

    # train zip
    with zipfile.ZipFile(os.path.join(args.data_dir, f"nsmc.zip"), "w") as z:
        z.write(train_file, "train.txt")
        z.write(test_file, "test.txt")

    os.remove(train_file)
    os.remove(test_file)


def parse_args():
    """
    build arguments
    :return args: input arguments
    """
    parser = argparse.ArgumentParser(description="Prepare nsmc arguments.")
    parser.add_argument("--data_dir", type=str, default="nsmc", required=False, help="nsmc data directory")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
