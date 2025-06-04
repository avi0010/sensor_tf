from pickle import load

import tensorflow as tf
from tensorflow.keras import layers

from sensors.models.modules import CBAM


@tf.keras.utils.register_keras_serializable()
class Exlite(tf.keras.Model):
    def __init__(
            self,
            hidden: int,
            transformer_dim: int = 27,
            dropout_rate: float = 0.1,
            max_length: int = 101,  # Added max_length parameter
            **kwargs
    ):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.transformer_dim = transformer_dim
        self.dropout_rate = dropout_rate
        self.hidden = hidden

        self.channel_attn = CBAM(channels=self.transformer_dim, kernel_size=3)

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
            tf.keras.layers.Dense(self.hidden, activation=tf.nn.relu),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(1),
        ])

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]

        # Normalize input
        x = self.normalizer(x)
        attn = self.channel_attn(x)

        channel_tokens = tf.reduce_mean(attn, axis=1)  # (B, 1, transformer_dim)
        return self.output_mlp(channel_tokens)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden": self.hidden,
            "transformer_dim": self.transformer_dim,
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

    model = Exlite(
        hidden=32,
        transformer_dim=27,
    )

    dummy_input = tf.random.uniform((batch_size, seq_length, input_dim))
    out = model(dummy_input)
    print(out.shape)
