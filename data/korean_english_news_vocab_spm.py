# -*- coding:utf-8 -*-

import sentencepiece as spm
import argparse
import os
import zipfile


def train_sentencepiece(corpus, prefix, vocab_size):
    """
    sentencepiece를 이용해 vocab 학습
    :param corpus: 학습할 말뭉치
    :param prefix: 저장할 vocab 이름
    :param vocab_size: vocab 개수
    """
    spm.SentencePieceTrainer.train(
        f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 7}" +  # 7은 특수문자 개수
        " --model_type=bpe" +
        " --max_sentence_length=999999" +  # 문장 최대 길이
        " --pad_id=0 --pad_piece=[PAD]" +  # pad token 및 id 지정
        " --unk_id=1 --unk_piece=[UNK]" +  # unknown token 및 id 지정
        " --bos_id=2 --bos_piece=[BOS]" +  # begin of sequence token 및 id 지정
        " --eos_id=3 --eos_piece=[EOS]" +  # end of sequence token 및 id 지정
        " --user_defined_symbols=[SEP],[CLS],[MASK]")  # 기타 추가 토큰 SEP: 4, CLS: 5, MASK: 6


def main(args):
    """
    main function
    :param args: input arguments
    """
    # train corpus 앞축 풀기
    with zipfile.ZipFile(os.path.join(args.data_dir, args.zip)) as z:
        z.extract(args.txt_ko)
        z.extract(args.txt_en)

    # sentencepiece ko 학습
    train_sentencepiece(args.txt_ko, os.path.join(args.data_dir, f"korean_english_news_ko_{args.n_vocab}"), args.n_vocab)
    # train ko corpus 삭제
    os.remove(args.txt_ko)

    # sentencepiece en 학습
    train_sentencepiece(args.txt_en, os.path.join(args.data_dir, f"korean_english_news_en_{args.n_vocab}"), args.n_vocab)
    # train en corpus 삭제
    os.remove(args.txt_en)


def parse_args():
    """
    build arguments
    :return args: input arguments
    """
    parser = argparse.ArgumentParser(description="Sentencepeice vocab korean_english_news arguments.")
    parser.add_argument("--data_dir", type=str, default="korean_english_news", required=False, help="korean english news data directory")
    parser.add_argument("--zip", type=str, default="corpus.txt.zip", required=False, help="korean english news source zip file")
    parser.add_argument("--txt_ko", type=str, default="corpus.ko.txt", required=False, help="korean english news source ko txt file")
    parser.add_argument("--txt_en", type=str, default="corpus.en.txt", required=False, help="korean english news source en txt file")
    parser.add_argument("--n_vocab", type=int, default=32000, required=False, help="korean english news vocab count")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)