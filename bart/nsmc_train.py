# -*- coding:utf-8 -*-

import argparse
import os
import sys

import matplotlib.pyplot as plt
import sentencepiece as spm
import tensorflow as tf
import yaml
import pandas as pd
import pickle

from data import load_nsmc
from model import build_model_class

sys.path.append("..")

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
    df.to_csv(os.path.join(args.out_dir, "nsmc_history.csv"))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'b-', label='loss')
    plt.plot(history.history['val_loss'], 'r--', label='loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['acc'], 'g-', label='acc')
    plt.plot(history.history['val_acc'], 'k--', label='acc')
    plt.xlabel('Epoch')
    plt.legend()

    # plt.show()
    plt.savefig(os.path.join(args.out_dir, "nsmc_history.png"))


def main(args):
    """
    main function
    :param args: input arguments
    """
    # load config
    with open(args.config) as f:
        config = yaml.load(f)

    # vocab loading
    vocab = spm.SentencePieceProcessor()
    vocab.load(os.path.join(args.kowiki_dir, config["vocab"]))

    config["model"]["n_vocab"] = len(vocab)
    config["model"]["i_pad"] = vocab.pad_id()

    # load data (데어터 로딩 중복 수행 방지용 pickle 사용)
    data_pkl = os.path.join(args.out_dir, "nsmc.pkl")
    if os.path.exists(data_pkl):
        with open(data_pkl, "rb") as f:
            train_inputs, train_labels, test_inputs, test_labels = pickle.load(f)
    else:
        train_inputs, train_labels = load_nsmc(vocab, os.path.join(args.nsmc_dir, config["data"]["nsmc_train"]), 128)
        test_inputs, test_labels = load_nsmc(vocab, os.path.join(args.nsmc_dir, config["data"]["nsmc_test"]), 128)
        with open(data_pkl, "wb") as f:
            pickle.dump((train_inputs, train_labels, test_inputs, test_labels), f)

    # build multi GPU model
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    with strategy.scope():
        model = build_model_class(config["model"], 2)
    model.summary()
    model.load_weights(os.path.join(args.out_dir, "bart-pretrain.hdf5"), by_name=True)

    # model compile
    learning_rate = get_schedule(config["schedule"])
    optimizer = get_optimizer(config["optimizer"], learning_rate)

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy, optimizer=optimizer, metrics=["acc"])

    # save weights callback
    save_weights = tf.keras.callbacks.ModelCheckpoint(os.path.join(args.out_dir, "bart-nsmc.hdf5"), monitor="val_acc", verbose=1, save_best_only=True, mode="max", save_freq="epoch", save_weights_only=True)
    # train
    history = model.fit(train_inputs, train_labels, validation_data=(test_inputs, test_labels), epochs=args.epochs, batch_size=args.batch_size, callbacks=[save_weights])

    save_history(args, history)


def parse_args():
    """
    build arguments
    :return args: input arguments
    """
    parser = argparse.ArgumentParser(description="Train korean_english_news arguments.")
    parser.add_argument("--config", type=str, default="config/pretrain_kowiki_finetune_nsmc.yaml", required=False, help="configuration file")
    parser.add_argument("--out_dir", type=str, default=None, required=False, help="result save directory")
    parser.add_argument("--kowiki_dir", type=str, default="../data/kowiki", required=False, help="kowiki data directory")
    parser.add_argument("--nsmc_dir", type=str, default="../data/nsmc", required=False, help="kowiki data directory")
    parser.add_argument("--noise_type", type=str, required=True, help="noise type [mask, delete, permute, rotate, infill]")
    parser.add_argument("--epochs", type=int, default=10, required=False, help="train epoch count")
    parser.add_argument("--batch_size", type=int, default=512, required=False, help="train batch size")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # make out_dir by config name
    if args.out_dir is None:
        basename, ext = os.path.splitext(os.path.basename(args.config))
        args.out_dir = os.path.join("result", basename, args.noise_type)
    print(f"result save dir: " + args.out_dir)

    # output directory create
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
        print(f"create {args.out_dir}")

    # run main
    main(args)
