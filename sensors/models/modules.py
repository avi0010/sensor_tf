import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DepthwiseSeparableConv(keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int = 7):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = layers.DepthwiseConv1D(kernel_size=kernel_size, padding="same")
        self.pointwise = layers.Conv1D(filters=filters, kernel_size=1)

    def call(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LearnablePositionalEncoding(keras.layers.Layer):
    def __init__(self, d_model: int, max_len: int):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_encoding = self.add_weight(
            name="pos_encoding",
            shape=(1, max_len, d_model),
            initializer="zeros",
            trainable=True,
        )
        self.max_len = max_len
        self.d_model = d_model

    def call(self, x):
        seq_len = tf.shape(x)[1]
        pos_enc = tf.slice(
            self.positional_encoding, begin=[0, 0, 0], size=[-1, seq_len, -1]
        )
        return x + pos_enc


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def call(self, q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # [..., seq_len_q, seq_len_k]
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_scores = matmul_qk / tf.math.sqrt(dk)

        attn_weights = tf.nn.softmax(scaled_scores, axis=-1)
        output = tf.matmul(attn_weights, v)  # [..., seq_len_q, depth_v]
        return output


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        self.attn = ScaledDotProductAttention()

    def split_heads(self, x, batch_size):
        # (batch_size, seq_len, d_model) → (batch_size, num_heads, seq_len, depth)
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scaled_attention = self.attn(q, k, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        return self.dense(concat)


class PositionwiseFFN(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super().__init__()
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

    def call(self, x):
        return self.ffn(x)


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFFN(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        # Multi-head attention with residual connection
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class TransformerEncoder(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, max_len, dropout_rate=0.1):
        super().__init__()
        self.pos_encoding = LearnablePositionalEncoding(max_len=max_len, d_model=d_model)
        self.enc_layers = [
            TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training, mask=mask)

        return x  # shape: (batch_size, input_seq_len, d_model)
