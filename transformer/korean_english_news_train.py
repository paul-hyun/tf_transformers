# -*- coding:utf-8 -*-

import argparse
import os
import sys

import matplotlib.pyplot as plt
import sentencepiece as spm
import tensorflow as tf
import yaml
import pandas as pd

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


def save_history(args, history):
    """
    save history (csv and graph)
    :param args: input args
    :param history: history data
    :return:
    """
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(args.out_dir, "history.csv"))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'b-', label='loss')
    plt.plot(history.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['lm_acc'], 'g-', label='acc')
    plt.plot(history.history['val_lm_acc'], 'k--', label='val_acc')
    plt.xlabel('Epoch')
    plt.legend()

    # plt.show()
    plt.savefig(os.path.join(args.out_dir, "history.png"))


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

    # load data
    train_inputs, train_labels = load_korean_english_news(vocab_ko, vocab_en, os.path.join(args.data_dir, config["data"]["train"]), config["model"]["n_seq"])
    dev_inputs, dev_labels = load_korean_english_news(vocab_ko, vocab_en, os.path.join(args.data_dir, config["data"]["dev"]), config["model"]["n_seq"])

    # build multi GPU model
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    with strategy.scope():
        model = build_model(config["model"])
    model.summary()

    # model compile
    learning_rate = get_schedule(config["schedule"])
    optimizer = get_optimizer(config["optimizer"], learning_rate)

    model.compile(loss=lm_loss, optimizer=optimizer, metrics=[lm_acc])

    # save weights callback
    save_weights = tf.keras.callbacks.ModelCheckpoint(os.path.join(args.out_dir, "transformer.hdf5"), monitor="lm_acc", verbose=1, save_best_only=True, mode="max", save_freq="epoch", save_weights_only=True)
    # train
    history = model.fit(train_inputs, train_labels, epochs=args.epochs, batch_size=args.batch_size, validation_data=(dev_inputs, dev_labels), callbacks=[save_weights])

    save_history(args, history)


def parse_args():
    """
    build arguments
    :return args: input arguments
    """
    parser = argparse.ArgumentParser(description="Train korean_english_news arguments.")
    parser.add_argument("--config", type=str, default="config/korean_english_news_ko_en_32000.yaml", required=False, help="configuration file")
    parser.add_argument("--out_dir", type=str, default=None, required=False, help="result save directory")
    parser.add_argument("--data_dir", type=str, default="../data/korean_english_news", required=False, help="korean english news data directory")
    parser.add_argument("--epochs", type=int, default=30, required=False, help="train epoch count")
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
