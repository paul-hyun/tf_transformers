# model.py
- transformer 모델

# data.py
- data loader

# korean_english_news_preprocess.py
- korean_english_news 데이터 전처리
- 출력
  - korean_english_news_train_32000.json.zip: train data
  - korean_english_news_dev_32000.json.zip: dev data
  - korean_english_news_test_32000.json.zip: test data


# korean_english_news_train.py
- korean_english_news 데이터 학습
- 출력
  - result/korean_english_news_ko_en_32000/transformer.hdf5: 학습 된 weights
  - result/korean_english_news_ko_en_32000/history.csv: train history csv
  - result/korean_english_news_ko_en_32000/history.png: train history graph


# korean_english_news_infer.py
- korean_english_news 학습결과 확인
