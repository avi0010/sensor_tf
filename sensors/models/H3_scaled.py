import tensorflow as tf
from keras.saving import register_keras_serializable
from tensorflow.keras import layers

from sensors.config import LENGTH
from sensors.models.modules import (
    DepthwiseSeparableConv,
    LearnablePositionalEncoding,
    SqueezeExcitation,
    MultiHeadAttention,
    TransformerEncoder,
)


@register_keras_serializable()
class Conv_Attn_Conv_Scaled(tf.keras.Model):
    def __init__(
            self,
            n_heads: int,
            hidden: int,
            transformer_dim: int = 16,
            num_layers: int = 1,
            dropout_rate: float = 0.1,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_heads = n_heads
        self.hidden = hidden
        self.transformer_dim = transformer_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.temporal_conv = DepthwiseSeparableConv(filters=transformer_dim, kernel_size=7)
        self.channel_attention = SqueezeExcitation(channels=transformer_dim)  # Apply to conv output

        # Add positional encoding and optional transformer
        self.positional_encoding = LearnablePositionalEncoding(
            max_len=LENGTH, d_model=transformer_dim
        )
        self.temporal_encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=transformer_dim,
            num_heads=n_heads,
            dff=hidden,
            max_len=LENGTH,
            dropout_rate=dropout_rate
        )

        self.cross_attention = MultiHeadAttention(transformer_dim, num_heads=n_heads)

        # Efficient output processing
        self.output_mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(1)
        ])
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]

        temporal_embed = self.temporal_conv(x, training=training)
        temporal_embed_pos = self.positional_encoding(temporal_embed)
        temporal_features = self.temporal_encoder(temporal_embed_pos, training=training)
        temporal_pooled = tf.reduce_mean(temporal_features, axis=1, keepdims=True)

        channel_attended = self.channel_attention(temporal_features, training=training)

        cross_attn_out = self.cross_attention(
            q=temporal_pooled,  # Temporal features asking questions
            k=channel_attended,  # Channel features as keys
            v=channel_attended,  # Channel features as values
            training=training
        )

        cross_final = tf.squeeze(cross_attn_out, axis=1)  # [batch, dim]

        return self.output_mlp(cross_final, training=training)

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
    # Create test data
    batch_size = 4
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
