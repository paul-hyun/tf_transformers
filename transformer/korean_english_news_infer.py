# -*- coding:utf-8 -*-

import argparse
import os
import sys

import numpy as np
import sentencepiece as spm
import yaml

from data import load_korean_english_news
from model import build_model

sys.path.append("..")

from common.losses import lm_loss
from common.metrics import lm_acc
from common.optimizers import get_schedule, get_optimizer

# 로그레벨 설정
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# GPU 목록 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"


def do_predict(model, vocab_ko, vocab_en, n_seq, string, src="ko", dst="en", i_pad=0):
    """
    입력을 번역하는 함수
    :param model: model object
    :param vocab_ko: korean vocab
    :param vocab_en: english vocab
    :param n_seq: sequence number
    :param string: input string
    :param src: source lang type
    :param dst: target lang type
    :param i_pad: pad index
    :return:
    """
    vocab_src = vocab_ko if src == "ko" else vocab_en
    vocab_dst = vocab_en if dst == "en" else vocab_ko

    # enc_token 생성: <string tokens>, [PAD] tokens
    enc_token = vocab_src.encode_as_ids(string)
    enc_token += [0] * (n_seq - len(enc_token))
    enc_token = enc_token[:n_seq]
    # dec_token 생성: [BOS], [PAD] tokens
    dec_token = [vocab_dst.bos_id()]
    dec_token += [0] * (n_seq - len(dec_token))
    dec_token = dec_token[:n_seq]

    response = []
    for i in range(n_seq - 1):
        # model 실행
        output = model.predict((np.array([enc_token]), np.array([dec_token])))
        # decoder의 마지막 위치의 token 예측 값
        word_id = int(np.argmax(output, axis=2)[0][i])
        # [EOS] 토큰이 생성되면 종료
        if word_id == vocab_dst.eos_id():
            break
        # 예측된 token을 응답에 저장
        response.append(word_id)
        # 예측된 token을 decoder의 다음 입력으로 저장
        dec_token[i + 1] = word_id

    # 생성된 token을 문자열로 변경
    return vocab_dst.decode_ids(response)


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

    config["model"]["n_vocab"] = len(vocab_ko)
    config["model"]["i_pad"] = vocab_ko.pad_id()

    test_inputs, test_labels = load_korean_english_news(vocab_ko, vocab_en, os.path.join(args.data_dir, config["data"]["test"]), config["model"]["n_seq"])

    model = build_model(config["model"])
    model.summary()

    # model compile
    learning_rate = get_schedule(config["schedule"])
    optimizer = get_optimizer(config["optimizer"], learning_rate)

    model.compile(loss=lm_loss, optimizer=optimizer, metrics=[lm_acc])

    # model init weight
    model.load_weights(os.path.join(args.out_dir, "transformer.hdf5"))

    # evaluate
    model.evaluate(test_inputs, test_labels, batch_size=args.batch_size)

    # test
    while True:
        print("input > ", end="")
        string = str(input())
        if len(string) == 0:
            break
        output = do_predict(model, vocab_ko, vocab_en, config["model"]["n_seq"], string)
        print(f"output > {output}")


def parse_args():
    """
    build arguments
    :return args: input arguments
    """
    parser = argparse.ArgumentParser(description="Train korean_english_news arguments.")
    parser.add_argument("--config", type=str, default="config/korean_english_news_ko_en_32000.yaml", required=False, help="configuration file")
    parser.add_argument("--out_dir", type=str, default=None, required=False, help="result save directory")
    parser.add_argument("--data_dir", type=str, default="../data/korean_english_news", required=False, help="korean english news data directory")
    parser.add_argument("--batch_size", type=int, default=128, required=False, help="train batch size")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # make out_dir by config name
    if args.out_dir is None:
        basename, ext = os.path.splitext(os.path.basename(args.config))
        args.out_dir = os.path.join("result", basename)
    print(f"result save dir: " + args.out_dir)

    # output directory create
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"create {args.out_dir}")

    # run main
    main(args)
