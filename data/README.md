# KoWiki python scripts

## kowiki_prepare.py
- 최신 kowiki 파일 다운로드 및 전처리
- 출력
  - kowiki.txt.zip: 토픽 단위로 정리된 text 파일

## kowiki_mecab_prepare.py
- kowiki.txt.zip 파일을 mecab 형태소 분석을 수행한 corpus 생성
- 출력
  - kowiki_mecab.txt.zip: 토픽 단위로 정리된 text 파일


## kowiki_noun_prepare.py
- kowiki.txt.zip 파일을 kb noun_splitter 분석을 수행한 corpus 생성 (https://github.com/KB-Bank-AI/KB-ALBERT-KO/blob/master/src/preprocessing/noun_splitter.py)
- 출력
  - kowiki_noun.txt.zip: 토픽 단위로 정리된 text 파일


## kowiki_word_count.py
- 형태소분석기 별로 형태소 및 단어 발생 빈도 측정
  - 향후 효과적인 vocab을 만들기 위함
- 출력
  - mecab_word_count.csv.zip: mecab을 이용한 단어 빈도수 csv 파일
  - komoran_word_count.csv.zip: komoran을 이용한 단어 빈도수 csv 파일
  - okt_word_count.csv.zip: okt를 이용한 단어 빈도수 csv 파일

## kowiki_vocab_spm.py
- sentencepiece vocab 생성
- 출력
  - kowiki_unigram_<n_vocab>.model: sentencepiece vocab model
  - kowiki_unigram_<n_vocab>.vocab: sentencepiece vocab txt
  - kowiki_bpe_<n_vocab>.model: sentencepiece vocab model
  - kowiki_bpe_<n_vocab>.vocab: sentencepiece vocab txt
  - kowiki_mecab_unigram_<n_vocab>.model: sentencepiece vocab model
  - kowiki_mecab_unigram_<n_vocab>.vocab: sentencepiece vocab txt
  - kowiki_mecab_bpe_<n_vocab>.model: sentencepiece vocab model
  - kowiki_mecab_bpe_<n_vocab>.vocab: sentencepiece vocab txt
  - kowiki_noun_unigram_<n_vocab>.model: sentencepiece vocab model
  - kowiki_noun_unigram_<n_vocab>.vocab: sentencepiece vocab txt
  - kowiki_noun_bpe_<n_vocab>.model: sentencepiece vocab model
  - kowiki_noun_bpe_<n_vocab>.vocab: sentencepiece vocab txt


# korean-english-news python scripts

## korean_english_news_prepare.py
- korean_english_news_prepare 다운로드 및 vocab용 corpus 생성
- 출력
  - korean-english-park.dev.tar.gz: 다운로드 한 dev 파일
  - korean-english-park.train.tar.gz: 다운로드 한 train 파일 
  - korean-english-park.test.tar.gz: 다운로드 한 test 파일
  - corpus.txt.zip: vocab용 corpus 파일


## korean_english_news_vocab_spm.py
- sentencepiece vocab 생성
- 출력
  - korean_english_news_ko_<n_vocab>.model: sentencepiece ko vocab model
  - korean_english_news_ko_<n_vocab>.vocab: sentencepiece en vocab txt
  - korean_english_news_en_<n_vocab>.model: sentencepiece en vocab model
  - korean_english_news_en_<n_vocab>.vocab: sentencepiece en vocab txt

