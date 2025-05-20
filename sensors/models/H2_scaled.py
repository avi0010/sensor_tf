import tensorflow as tf
from keras.saving import register_keras_serializable
from tensorflow.keras import layers

from sensors.config import LENGTH
from sensors.models.modules import DepthwiseSeparableConv, LearnablePositionalEncoding, TransformerEncoderLayer

@register_keras_serializable()
class Conv_Attn_Conv_Scaled(tf.keras.Model):
    def __init__(
        self,
        input_dim,
        n_heads,
        hidden,
        transformer_dim=16,
        num_layers=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.hidden = hidden
        self.transformer_dim = transformer_dim

        self.embedding_temporal = DepthwiseSeparableConv(filters=transformer_dim)
        self.embedding_channel = DepthwiseSeparableConv(filters=transformer_dim)

        self.positional_encoding_temporal = LearnablePositionalEncoding(
            max_len=LENGTH, d_model=transformer_dim
        )
        self.positional_encoding_channel = LearnablePositionalEncoding(
            max_len=input_dim, d_model=transformer_dim
        )

        self.temporal_encoder = tf.keras.Sequential([
            TransformerEncoderLayer(transformer_dim, n_heads, hidden)
            for _ in range(num_layers)
        ])
        self.channel_encoder = tf.keras.Sequential([
            TransformerEncoderLayer(transformer_dim, n_heads, hidden)
            for _ in range(num_layers)
        ])

        self.gate_mlp = tf.keras.Sequential([
            layers.Dense(hidden, activation="relu"),
            layers.Dense(2, activation="softmax"),
        ])

        self.layer_norm_temporal = layers.LayerNormalization()
        self.layer_norm_channel = layers.LayerNormalization()
        self.output_mlp = tf.keras.Sequential([layers.Dense(1)])

    def call(self, x):
        batch_size = tf.shape(x)[0]

        # Temporal Transformer Block
        temporal_embed = self.embedding_temporal(x)
        temporal_embed = self.positional_encoding_temporal(temporal_embed)
        temporal_out = self.temporal_encoder(temporal_embed)
        temporal_flat = tf.reshape(temporal_out, [batch_size, -1])
        temporal_norm = self.layer_norm_temporal(temporal_flat)

        # Channel Transformer Block
        x_transposed = tf.transpose(x, [0, 2, 1])
        channel_embed = self.embedding_channel(x_transposed)
        channel_embed = self.positional_encoding_channel(channel_embed)
        channel_out = self.channel_encoder(channel_embed)
        channel_flat = tf.reshape(channel_out, [batch_size, -1])
        channel_norm = self.layer_norm_channel(channel_flat)

        # Gating Fusion
        gate = self.gate_mlp(tf.concat([temporal_norm, channel_norm], axis=-1))
        encoding = tf.concat(
            [temporal_norm * gate[:, 0:1], channel_norm * gate[:, 1:2]], axis=-1
        )

        return self.output_mlp(encoding)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "n_heads": self.n_heads,
            "hidden": self.hidden,
            "transformer_dim": self.transformer_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    # Create test data
    batch_size = 1
    seq_length = 101
    input_dim = 27

    # Test inputs
    inputs = tf.random.normal((batch_size, seq_length, input_dim))

    model = Conv_Attn_Conv_Scaled(
        input_dim=27,
        n_heads=2,
        hidden=32,
        transformer_dim=16,
    )

    out = model(inputs)
    print(out.shape)
    model.summary()
