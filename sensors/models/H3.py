from pickle import load

import tensorflow as tf
import tensorflow_models as tfm
from tensorflow.keras import layers

from sensors.models.modules import DepthwiseSeparableConv
from sensors.models.modules import LearnablePositionalEncoding, CBAM


@tf.keras.utils.register_keras_serializable()
class Conv_Attn_Conv_Scaled(tf.keras.Model):
    def __init__(
            self,
            n_heads: int,
            hidden: int,
            transformer_dim: int = 16,
            num_layers: int = 1,
            dropout_rate: float = 0.1,
            max_length: int = 101,  # Added max_length parameter
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.hidden = hidden
        self.transformer_dim = transformer_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.max_length = max_length

        # Temporal convolution
        self.temporal_conv = DepthwiseSeparableConv(filters=transformer_dim, kernel_size=7)

        self.pos_encoding = LearnablePositionalEncoding(max_len=max_length, d_model=transformer_dim)

        self.temporal_encoder = tfm.nlp.layers.TransformerEncoderBlock(
            num_attention_heads=n_heads,
            inner_dim=hidden,
            inner_activation='relu',
            output_dropout=dropout_rate,
            attention_dropout=dropout_rate,
            norm_first=True,
            norm_epsilon=1e-6,
            inner_dropout=dropout_rate
        )

        self.channel_attn = CBAM(channels=transformer_dim, kernel_size=3)

        self.cross_attention = tfm.nlp.layers.MultiHeadAttention(
            num_heads=n_heads,
            key_dim=transformer_dim // n_heads,
            dropout=dropout_rate
        )

        scaler = load(open("StandardScaler.pkl", "rb"))
        mean = scaler.mean_.tolist()
        std = scaler.scale_
        variance = (std ** 2).tolist()
        self.normalizer = tf.keras.layers.Normalization(
            mean=mean,
            variance=variance,
            axis=-1,
            trainable=False
        )
        self.normalizer.build([None, 101, 27])

        # Output MLP
        self.output_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation=tf.nn.relu),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1),
        ])

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]

        # Normalize input
        x = self.normalizer(x)

        # Temporal convolution
        temporal_embed = self.temporal_conv(x)

        # Add positional encoding
        temporal_features = self.pos_encoding(temporal_embed)

        temporal_features = self.temporal_encoder(temporal_features, training=training)

        channel_attention = self.channel_attn(temporal_features)

        channel_tokens = tf.reduce_mean(channel_attention, axis=1, keepdims=True)  # (B, 1, transformer_dim)

        # Cross attention
        cross_attn_out = self.cross_attention(
            query=channel_tokens,
            value=temporal_features,
            key=temporal_features,
            training=training
        )

        # Pool and generate output
        pooled = tf.reduce_mean(cross_attn_out, axis=1)
        return self.output_mlp(pooled, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_heads": self.n_heads,
            "hidden": self.hidden,
            "transformer_dim": self.transformer_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Create test data
    batch_size = 4
    seq_length = 101
    input_dim = 27

    model = Conv_Attn_Conv_Scaled(
        input_dim=27,
        n_heads=2,
        hidden=32,
        transformer_dim=16,
    )

    df = pd.read_excel("./data_generated/68B_6685_Q1_Ammonia_Liquid_1Rep_250102091107_data_03192025_102453.xlsx")

    LENGTH = 101
    for window in df.rolling(window=LENGTH):
        if len(window) < LENGTH:
            continue  # NOTE: Skip windows smaller than the required length.

        X, y = window.drop(columns=["Exposure"]), window.Exposure.values[-1]
        window_scaled = np.expand_dims(X, 0)
        out = model(window_scaled)
        break

    model.summary()
