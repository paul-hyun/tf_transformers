# -*- coding:utf-8 -*-

import math

import tensorflow as tf


class InverseSquareRootSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    InverseSquareRootSchedule class
    """

    def __init__(self, d_model, warmup_steps=4000):
        """
        생성자
        :param d_model: 모델 hidden
        :param warmup_steps: warmup steps
        """
        super().__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)

    def __call__(self, step_num):
        """
        learning rate 계산
        :param step_num: 현재 step number
        :retrun: 계산된 learning rate
        """
        arg1 = tf.math.rsqrt(step_num)
        arg2 = step_num * (self.warmup_steps ** -1.5)
        arg = tf.math.minimum(arg1, arg2)
        lr = tf.math.rsqrt(self.d_model) * arg
        return lr


class CosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    CosineSchedule Class
    """

    def __init__(self, train_steps=4000, warmup_steps=2000, max_lr=2.5e-4):
        """
        생성자
        :param train_steps: 학습 step 총 합
        :param warmup_steps: warmup steps
        :param max_lr: 최대 learning rate
        """
        super().__init__()

        assert 0 < warmup_steps < train_steps
        self.warmup_steps = warmup_steps
        self.train_steps = train_steps
        self.max_lr = max_lr

    def __call__(self, step_num):
        """
        learning rate 계산
        :param step_num: 현재 step number
        :retrun: 계산된 learning rate
        """
        state = tf.cast(step_num <= self.warmup_steps, tf.float32)
        lr1 = tf.cast(step_num, tf.float32) / self.warmup_steps
        progress = tf.cast(step_num - self.warmup_steps, tf.float32) / max(1, self.train_steps - self.warmup_steps)
        lr2 = 0.5 * (1.0 + tf.math.cos(math.pi * progress))
        return (state * lr1 + (1 - state) * lr2) * self.max_lr


def get_schedule(config):
    if config["name"] == "InverseSquareRootSchedule":
        return InverseSquareRootSchedule(config["d_model"], config["warmup_steps"])
    elif config["name"] == "CosineSchedule":
        return CosineSchedule(config["train_steps"], config["warmup_steps"], config["max_lr"])


def get_optimizer(config, learning_rate=None):
    if config["name"] == "Adam":
        if learning_rate is None:
            learning_rate = config["learning_rate"]
        return tf.keras.optimizers.Adam(learning_rate, beta_1=config.get("beta_1", None), beta_2=config.get("beta_2", None), epsilon=config.get("epsilon", None))
