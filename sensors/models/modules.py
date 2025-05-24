import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DepthwiseSeparableConv(keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int = 5):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = layers.DepthwiseConv1D(kernel_size=kernel_size, padding="same")
        self.pointwise = layers.Conv1D(filters=filters, kernel_size=1)

    def call(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_encoding = self.add_weight(
            shape=(max_len, d_model),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
            name="learnable_positional_encoding"
        )

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:seq_len]


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


class SqueezeExcitation(tf.keras.layers.Layer):
    """
    Squeeze-and-Excitation block for channel attention
    """

    def __init__(self, channels, reduction_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels

        self.squeeze = tf.keras.layers.GlobalAveragePooling1D()
        self.excitation = tf.keras.Sequential([
            tf.keras.layers.Dense(max(channels // reduction_ratio, 4), activation='relu'),
            tf.keras.layers.Dense(channels, activation='sigmoid')
        ])

    def call(self, x, training=False):
        se = self.squeeze(x)  # [batch, channels]
        weights = self.excitation(se, training=training)  # [batch, channels]
        weights = tf.expand_dims(weights, 1)  # [batch, 1, channels]
        return x * weights


class ChannelAttention(tf.keras.layers.Layer):
    """Channel Attention Module of CBAM"""

    def __init__(self, channels, reduction_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.reduced_channels = max(1, channels // reduction_ratio)

        # Shared MLP layers
        self.mlp = tf.keras.Sequential([
            layers.Dense(self.reduced_channels, activation='relu'),
            layers.Dense(channels)
        ])

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=False):
        # x shape: (batch, time, channels) for 1D conv or (batch, height, width, channels) for 2D

        # Global Average Pool and Global Max Pool
        avg_pool = tf.reduce_mean(x, axis=1, keepdims=False)  # (batch, channels)
        max_pool = tf.reduce_max(x, axis=1, keepdims=False)  # (batch, channels)

        # Shared MLP
        avg_out = self.mlp(avg_pool, training=training)  # (batch, channels)
        max_out = self.mlp(max_pool, training=training)  # (batch, channels)

        # Element-wise sum and sigmoid activation
        channel_attention = tf.nn.sigmoid(avg_out + max_out)  # (batch, channels)

        # Reshape to match input dimensions for broadcasting
        channel_attention = tf.expand_dims(channel_attention, axis=1)

        return x * channel_attention


class SpatialAttention(tf.keras.layers.Layer):
    """Spatial Attention Module of CBAM"""

    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        # For 1D temporal data, use Conv1D
        self.conv = layers.Conv1D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            activation='sigmoid'
        )

        super().build(input_shape)

    def call(self, x, training=False):
        # Compute channel-wise statistics
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)  # Average across channels
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)  # Max across channels

        # Concatenate along channel dimension
        concat = tf.concat([avg_pool, max_pool], axis=-1)  # (..., 2)

        # Apply convolution and sigmoid
        spatial_attention = self.conv(concat, training=training)  # (..., 1)

        return x * spatial_attention


class CBAM(tf.keras.layers.Layer):
    """Convolutional Block Attention Module - Following ResNet CBAM Pattern"""

    def __init__(self, channels, reduction_ratio=4, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction_ratio=reduction_ratio
        )
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def call(self, x, training=False):
        # Store original input for final residual connection
        residual = x

        # Apply channel attention: multiply features with channel attention weights
        channel_weights = self.channel_attention(x, training=training)
        out = x * channel_weights

        # Apply spatial attention: multiply channel-attended features with spatial attention weights
        spatial_weights = self.spatial_attention(out, training=training)
        out = out * spatial_weights

        # Final residual connection
        out = out + residual

        return out
