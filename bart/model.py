# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


@tf.function
def get_pad_mask(tokens, i_pad=0):
    """
    pad mask 계산하는 함수
    :param tokens: tokens (bs, n_seq)
    :param i_pad: id of pad
    :return mask: pad mask (pad: 1, other: 0)
    """
    mask = tf.cast(tf.math.equal(tokens, i_pad), tf.float32)
    mask = tf.expand_dims(mask, axis=1)
    return mask


@tf.function
def get_ahead_mask(tokens, i_pad=0):
    """
    ahead mask 계산하는 함수
    :param tokens: tokens (bs, n_seq)
    :param i_pad: id of pad
    :return mask: ahead and pad mask (ahead or pad: 1, other: 0)
    """
    n_seq = tf.shape(tokens)[1]
    ahead_mask = 1 - tf.linalg.band_part(tf.ones((n_seq, n_seq)), -1, 0)
    ahead_mask = tf.expand_dims(ahead_mask, axis=0)
    pad_mask = get_pad_mask(tokens, i_pad)
    mask = tf.maximum(ahead_mask, pad_mask)
    return mask


@tf.function(experimental_relax_shapes=True)
def gelu(x):
    """
    gelu activation 함수
    :param x: 입력 값
    :return: gelu activation result
    """
    return 0.5 * x * (1 + K.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))


def kernel_initializer(stddev=0.02):
    """
    parameter initializer 생성
    :param stddev: 생성할 랜덤 변수의 표준편차
    """
    return tf.keras.initializers.TruncatedNormal(stddev=stddev)


def bias_initializer():
    """
    bias initializer 생성
    """
    return tf.zeros_initializer


def new_embedding(input_dim, output_dim, **kwargs):
    """
    Embedding 객체 생성
    :param input_dim: 입력 dimension
    :param output_dim: 출력 dimension
    :param kwargs: 그외 arguments
    :return:
    """
    return tf.keras.layers.Embedding(input_dim, output_dim, embeddings_initializer=kernel_initializer(), **kwargs)


def new_dense(units, **kwargs):
    """
    Dense 객체 생성
    :param units: 출력 dimension
    :param kwargs: 그외 arguments
    :return:
    """
    return tf.keras.layers.Dense(units, kernel_initializer=kernel_initializer(), bias_initializer=bias_initializer(), **kwargs)


class SharedEmbedding(tf.keras.layers.Layer):
    """
    Weighed Shaed Embedding Class
    """

    def __init__(self, config, name="weight_shared_embedding"):
        """
        생성자
        :param config: Config 객체
        :param name: layer name
        """
        super().__init__(name=name)

        self.n_vocab = config["n_vocab"]
        self.d_model = config["d_model"]

    def build(self, input_shape):
        """
        shared weight 생성
        :param input_shape: Tensor Shape (not used)
        """
        with tf.name_scope("shared_embedding_weight"):
            self.shared_weights = self.add_weight(
                "weights",
                shape=[self.n_vocab, self.d_model],
                initializer=kernel_initializer()
            )

    def call(self, inputs, mode="embedding"):
        """
        layer 실행
        :param inputs: 입력
        :param mode: 실행 모드
        :return: embedding or linear 실행 결과
        """
        # mode가 embedding일 경우 embedding lookup 실행
        if mode == "embedding":
            return self._embedding(inputs)
        # mode가 linear일 경우 linear 실행
        elif mode == "linear":
            return self._linear(inputs)
        # mode가 기타일 경우 오류 발생
        else:
            raise ValueError(f"mode {mode} is not valid.")

    def _embedding(self, inputs):
        """
        embedding lookup
        :param inputs: 입력
        """
        embed = tf.gather(self.shared_weights, tf.cast(inputs, tf.int32))
        return embed

    def _linear(self, inputs):  # (bs, n_seq, d_model)
        """
        linear 실행
        :param inputs: 입력
        """
        n_batch = tf.shape(inputs)[0]
        n_seq = tf.shape(inputs)[1]
        inputs = tf.reshape(inputs, [-1, self.d_model])  # (bs * n_seq, d_model)
        outputs = tf.matmul(inputs, self.shared_weights, transpose_b=True)
        outputs = tf.reshape(outputs, [n_batch, n_seq, self.n_vocab])  # (bs, n_seq, n_vocab)
        return outputs


class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Positional Embedding Class
    """

    def __init__(self, config, name="position_embedding"):
        """
        생성자
        :param config: Config 객체
        :param name: layer name
        """
        super().__init__(name=name)

        pos_encoding = PositionalEmbedding.get_sinusoid_encoding(config["n_seq"], config["d_model"])
        self.embedding = new_embedding(config["n_seq"], config["d_model"], trainable=False, weights=[pos_encoding])

    def call(self, inputs):
        """
        layer 실행
        :param inputs: 입력
        :return embed: positional embedding lookup 결과
        """
        position = tf.cast(tf.math.cumsum(tf.ones_like(inputs), axis=1, exclusive=True), tf.int32)
        embed = self.embedding(position)
        return embed

    @staticmethod
    def get_sinusoid_encoding(n_seq, d_model):
        """
        sinusoid encoding 생성
        :param n_seq: sequence number
        :param n_seq: model hidden dimension
        :return: positional encoding table
        """
        angles = [np.power(10000, 2 * (i_ang // 2) / d_model) for i_ang in range(d_model)]
        pos_encoding = np.array([[pos / angle for angle in angles] for pos in range(n_seq)])
        pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
        return tf.cast(pos_encoding, tf.float32)


class ScaleDotProductAttention(tf.keras.layers.Layer):
    """
    Scale Dot Product Attention Class
    """

    def __init__(self, name="scale_dot_product_attention"):
        """
        생성자
        :param name: layer name
        """
        super().__init__(name=name)

    def call(self, Q, K, V, attn_mask):
        """
        layer 실행
        :param Q: Q value
        :param K: K value
        :param V: V value
        :param attn_mask: 실행 모드
        :return attn_out: attention 실행 결과
        """
        attn_score = tf.matmul(Q, K, transpose_b=True)
        scale = tf.math.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))
        attn_scale = tf.math.divide(attn_score, scale)
        attn_scale -= 1.e9 * attn_mask
        attn_prob = tf.nn.softmax(attn_scale, axis=-1)
        attn_out = tf.matmul(attn_prob, V)
        return attn_out


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi Head Attention Class
    """

    def __init__(self, config, name="multi_head_attention"):
        """
        생성자
        :param config: Config 객체
        :param name: layer name
        """
        super().__init__(name=name)

        self.d_model = config["d_model"]
        self.n_head = config["n_head"]
        self.d_head = config["d_head"]

        # Q, K, V input dense layer
        self.W_Q = new_dense(self.n_head * self.d_head)
        self.W_K = new_dense(self.n_head * self.d_head)
        self.W_V = new_dense(self.n_head * self.d_head)
        # Scale Dot Product Attention class
        self.attention = ScaleDotProductAttention(name="self_attention")
        # output dense layer
        self.W_O = new_dense(self.d_model)

    def call(self, Q, K, V, attn_mask):
        """
        layer 실행
        :param Q: Q value
        :param K: K value
        :param V: V value
        :param attn_mask: 실행 모드
        :return attn_out: attention 실행 결과
        """
        # reshape Q, K, V, attn_mask
        batch_size = tf.shape(Q)[0]
        Q_m = tf.transpose(tf.reshape(self.W_Q(Q), [batch_size, -1, self.n_head, self.d_head]), [0, 2, 1, 3])  # (bs, n_head, Q_len, d_head)
        K_m = tf.transpose(tf.reshape(self.W_K(K), [batch_size, -1, self.n_head, self.d_head]), [0, 2, 1, 3])  # (bs, n_head, K_len, d_head)
        V_m = tf.transpose(tf.reshape(self.W_V(V), [batch_size, -1, self.n_head, self.d_head]), [0, 2, 1, 3])  # (bs, n_head, K_len, d_head)
        attn_mask_m = tf.expand_dims(attn_mask, axis=1)
        # Scale Dot Product Attention with multi head Q, K, V, attn_mask
        attn_out = self.attention(Q_m, K_m, V_m, attn_mask_m)  # (bs, n_head, Q_len, d_head)
        # transpose and liner
        attn_out_m = tf.transpose(attn_out, perm=[0, 2, 1, 3])  # (bs, Q_len, n_head, d_head)
        attn_out = tf.reshape(attn_out_m, [batch_size, -1, self.n_head * self.d_head])  # (bs, Q_len, d_model)
        attn_out = self.W_O(attn_out)  # (bs, Q_len, d_model)

        return attn_out


class PositionWiseFeedForward(tf.keras.layers.Layer):
    """
    Position Wise Feed Forward Class
    """

    def __init__(self, config, name="feed_forward"):
        """
        생성자
        :param config: Config 객체
        :param name: layer name
        """
        super().__init__(name=name)

        self.W_1 = new_dense(config["d_ff"], activation=gelu)
        self.W_2 = new_dense(config["d_model"])

    def call(self, inputs):
        """
        layer 실행
        :param inputs: inputs
        :return ff_val: feed forward 실행 결과
        """
        ff_val = self.W_2(self.W_1(inputs))
        return ff_val


class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder Layer Class
    """

    def __init__(self, config, name="encoder_layer"):
        """
        생성자
        :param config: Config 객체
        :param name: layer name
        """
        super().__init__(name=name)

        self.self_attention = MultiHeadAttention(config)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=config["layernorm_epsilon"])

        self.ffn = PositionWiseFeedForward(config)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=config["layernorm_epsilon"])

        self.dropout = tf.keras.layers.Dropout(config["dropout"])

    def call(self, enc_embed, self_mask):
        """
        layer 실행
        :param enc_embed: enc_embed 또는 이전 EncoderLayer의 출력
        :param self_mask: enc_tokens의 pad mask
        :return enc_out: EncoderLayer 실행 결과
        """
        self_attn_val = self.self_attention(enc_embed, enc_embed, enc_embed, self_mask)
        norm1_val = self.norm1(enc_embed + self.dropout(self_attn_val))

        ffn_val = self.ffn(norm1_val)
        enc_out = self.norm2(norm1_val + self.dropout(ffn_val))

        return enc_out


class DecoderLayer(tf.keras.layers.Layer):
    """
    Decoder Layer Class
    """

    def __init__(self, config, name="decoder_layer"):
        """
        생성자
        :param config: Config 객체
        :param name: layer name
        """
        super().__init__(name=name)

        self.self_attention = MultiHeadAttention(config)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=config["layernorm_epsilon"])

        self.ende_attn = MultiHeadAttention(config)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=config["layernorm_epsilon"])

        self.ffn = PositionWiseFeedForward(config)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=config["layernorm_epsilon"])

        self.dropout = tf.keras.layers.Dropout(config["dropout"])

    def call(self, dec_embed, enc_out, self_mask, ende_mask):
        """
        layer 실행
        :param dec_embed: dec_embed 또는 이전 DecoderLayer의 출력
        :param enc_out: 마지막 EncoderLayer의 출력
        :param self_mask: dec_tokens의 ahead mask
        :param ende_mask: enc_tokens의 pad mask
        :return dec_out: DecoderLayer 실행 결과
        """
        self_attn_val = self.self_attention(dec_embed, dec_embed, dec_embed, self_mask)
        norm1_val = self.norm1(dec_embed + self.dropout(self_attn_val))

        ende_attn_val = self.ende_attn(norm1_val, enc_out, enc_out, ende_mask)
        norm2_val = self.norm2(norm1_val + self.dropout(ende_attn_val))

        ffn_val = self.ffn(norm2_val)
        dec_out = self.norm3(norm2_val + self.dropout(ffn_val))

        return dec_out


class BART(tf.keras.layers.Layer):
    """
    BART Class
    """

    def __init__(self, config, name="bart"):
        """
        생성자
        :param config: Config 객체
        :param name: layer name
        """
        super().__init__(name=name)

        self.i_pad = config["i_pad"]
        self.embedding = SharedEmbedding(config)
        self.position = PositionalEmbedding(config)

        self.enc_norm = tf.keras.layers.LayerNormalization(epsilon=config["layernorm_epsilon"])
        self.encoder_layers = [EncoderLayer(config, name=f"encoder_layer_{i}") for i in range(config["n_layer"])]
        self.dec_norm = tf.keras.layers.LayerNormalization(epsilon=config["layernorm_epsilon"])
        self.decoder_layers = [DecoderLayer(config, name=f"decoder_layer_{i}") for i in range(config["n_layer"])]

        self.dropout = tf.keras.layers.Dropout(config["dropout"])

    def call(self, inputs):
        """
        layer 실행
        :param inputs: (enc_tokens, dec_tokens) tuple
        :return logits: dec_tokens에 대한 다음 토큰 예측 결과 logits
        """
        enc_tokens, dec_tokens = inputs

        enc_self_mask = get_pad_mask(enc_tokens, self.i_pad)
        dec_self_mask = get_ahead_mask(dec_tokens, self.i_pad)
        ende_attn_mask = get_pad_mask(enc_tokens, self.i_pad)

        enc_embed = self.get_embedding(enc_tokens)
        dec_embed = self.get_embedding(dec_tokens)

        enc_embed = self.enc_norm(enc_embed)
        enc_out = self.dropout(enc_embed)
        for encoder_layer in self.encoder_layers:
            enc_out = encoder_layer(enc_out, enc_self_mask)

        dec_embed = self.dec_norm(dec_embed)
        dec_out = self.dropout(dec_embed)
        for decoder_layer in self.decoder_layers:
            dec_out = decoder_layer(dec_out, enc_out, dec_self_mask, ende_attn_mask)

        logits_cls = dec_out[:, -1]
        logits_lm = self.embedding(dec_out, mode="linear")
        return logits_cls, logits_lm

    def get_embedding(self, tokens):
        """
        token embedding, position embedding lookup
        :param tokens: 입력 tokens
        :return embed: embedding 결과
        """
        embed = self.embedding(tokens) + self.position(tokens)
        return embed


def build_model_pre_train(config):
    """
    model build for pre-train
    :param config: Configuration
    :return model
    """
    enc_tokens = tf.keras.layers.Input((None,), name="enc_tokens")
    dec_tokens = tf.keras.layers.Input((None,), name="dec_tokens")

    _, logits_lm = BART(config)((enc_tokens, dec_tokens))
    outputs = tf.keras.layers.Softmax(name="lm")(logits_lm)

    model = tf.keras.Model(inputs=(enc_tokens, dec_tokens), outputs=outputs)
    return model


def build_model_class(config, n_class):
    """
    model build for classifier
    :param config: Configuration
    :param n_class: output class number
    :return model
    """
    enc_tokens = tf.keras.layers.Input((None,), name="enc_tokens")
    dec_tokens = tf.keras.layers.Input((None,), name="dec_tokens")

    logits_cls, _ = BART(config)((enc_tokens, dec_tokens))
    outputs = tf.keras.layers.Dense(n_class, activation=tf.nn.softmax, name="cls")(logits_cls)

    model = tf.keras.Model(inputs=(enc_tokens, dec_tokens), outputs=outputs)
    return model
