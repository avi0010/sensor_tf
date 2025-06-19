from pickle import load

import tensorflow as tf
# import tensorflow_models as tfm
from tensorflow.keras import layers

from sensors.models.modules import DepthwiseSeparableConv
from sensors.models.modules import LearnablePositionalEncoding, CBAM
from sensors.models.transformer_encoder import TransformerEncoderBlock


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
            linformer_dim: int = 64,
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

        # self.pos_encoding = LearnablePositionalEncoding(max_len=max_length, d_model=transformer_dim)

        # self.pos_encoding = tf.keras.layers.Embedding(
        #     input_dim=max_length, 
        #     output_dim=transformer_dim, 
        #     trainable=True,
        #     embeddings_initializer='uniform',
        #     name='positional_encoding'
        # )

        self.temporal_encoder = TransformerEncoderBlock(
            num_attention_heads=n_heads,
            num_kv_heads=1,
            key_dim=transformer_dim,
            value_dim=None,
            inner_dim=hidden,
            inner_activation='relu',
            output_dropout=dropout_rate,
            attention_dropout=dropout_rate,
            inner_dropout=dropout_rate,
            linformer_dim=linformer_dim,
            norm_first=True,
            norm_epsilon=1e-6,
            use_rms_norm=True,
            use_query_residual=True,
        )

        self.channel_attn = CBAM(channels=transformer_dim, kernel_size=3)

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
        seq_len = tf.shape(x)[1]
        batch_size = tf.shape(x)[0]

        # Normalize input
        x = self.normalizer(x)

        # Temporal convolution
        temporal_embed = self.temporal_conv(x)

        # Channel Attention
        channel_attention = self.channel_attn(temporal_embed, training=training)

        # Add positional encoding
        # positions = tf.range(seq_len)  # Create position indices
        # pos_embeddings = self.pos_encoding(positions)  # Get positional embeddings
        # temporal_features = channel_attention + pos_embeddings

        temporal_features = self.temporal_encoder(channel_attention, training=training)

        # Pool and generate output
        pooled = tf.reduce_mean(temporal_features, axis=1)
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
        max_length=101,
    )

    dummpy_input = tf.random.uniform([batch_size, seq_length, input_dim])
    out = model(dummpy_input)
    print(out.shape)
    model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
          tf.lite.OpsSet.TFLITE_BUILTINS,
          tf.lite.OpsSet.SELECT_TF_OPS 
     ]
    converter.target_spec.supported_types = [tf.float32]
    converter._experimental_lower_tensor_list_ops = False
    converter.experimental_enable_resource_variables = False
    tflite_model = converter.convert()
    output_path = "Hello_3.tflite"
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Model saved as {output_path}")
