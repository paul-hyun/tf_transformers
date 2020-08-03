# model.py
- transformer 모델

# data.py
- data loader

# kowiki_pretrain_preprocess.py
- pre-train pre process script

# kowiki_pretrain.py
- pre-train script

# nsmc_preproces.py
- nsmc pre process script

# nsmc_train.py
- nsmc train script

## pretrain_kowiki_finetune_nsmc / mask
  - epoch: 10
  - loss: 0.2761
  - acc: 0.8845
  - val_loss: 0.3775
  - val_acc: 0.8464  

## pretrain_kowiki_finetune_nsmc / infill
  - epoch: 9
  - loss: 0.2891
  - acc: 0.8776
  - val_loss: 0.3366
  - val_acc: 0.8556

## pretrain_kowiki_mecab_finetune_nsmc / infill
  - eopch: 10
  - loss: 0.2604
  - acc: 0.8916
  - val_loss: 0.3387
  - val_acc: 0.8650
  
## pretrain_kowiki_noun_finetune_nsmc / infill
  - epoch: 8
  - loss: 0.3039
  - acc: 0.8703
  - val_loss: 0.3319
  - val_acc: 0.8570
