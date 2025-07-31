from pickle import load

import tensorflow as tf
from sensors.models.multihead_pooling import TFMCrossAttentionPooling
from sensors.models.transformer_encoder import TransformerEncoderBlock


@tf.keras.utils.register_keras_serializable()
class LinformerClassifier(tf.keras.Model):
    def __init__(
        self,
        num_classes: int,  # Number of classes for classification
        n_heads: int = 2,
        hidden: int = 64,
        transformer_dim: int = 16,
        num_layers: int = 1,
        dropout_rate: float = 0.1,
        max_length: int = 101,
        linformer_dim: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.n_heads = n_heads
        self.hidden = hidden
        self.transformer_dim = transformer_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.max_length = max_length

        # Enhanced input processing
        self.input_projection = tf.keras.layers.Dense(
            transformer_dim, activation="relu", name="input_projection"
        )

        # Enhanced transformer encoders with residual connections
        self.temporal_encoders = []
        for i in range(num_layers):
            self.temporal_encoders.append(
                TransformerEncoderBlock(
                    num_attention_heads=n_heads,
                    # num_kv_heads=n_heads,
                    inner_dim=hidden,
                    inner_activation="relu",
                    output_dropout=dropout_rate,
                    attention_dropout=dropout_rate,
                    inner_dropout=dropout_rate,
                    linformer_dim=linformer_dim,
                    norm_first=True,
                    norm_epsilon=1e-6,
                    use_rms_norm=True,
                    use_query_residual=True,
                    name=f"transformer_layer_{i}",
                )
            )

        # Enhanced cross-attention pooling
        self.pooling = TFMCrossAttentionPooling(
            num_heads=n_heads,
            key_dim=transformer_dim,
            num_query_tokens=num_classes,
            dropout=dropout_rate,
        )

        # Normalization (load from pickle file)
        scaler = load(open("StandardScaler.pkl", "rb"))
        mean = scaler.mean_.tolist()
        std = scaler.scale_
        variance = (std**2).tolist()
        self.normalizer = tf.keras.layers.Normalization(
            mean=mean, variance=variance, axis=-1, trainable=False
        )
        self.normalizer.build([None, 101, 27])

        # Enhanced classification head
        self.classification_head = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden * 2, activation="relu"),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(hidden, activation="relu"),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.Dropout(dropout_rate * 0.5),
                tf.keras.layers.Dense(hidden // 2, activation="relu"),
                tf.keras.layers.Dropout(dropout_rate * 0.5),
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, x, training=False, return_attention=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Normalize input
        temporal_features = self.normalizer(x)

        # Project to transformer dimension
        temporal_features = self.input_projection(temporal_features, training=training)

        # Apply transformer layers
        for encoder in self.temporal_encoders:
            temporal_features = encoder(temporal_features, training=training)

        # Cross-attention pooling
        pooled = self.pooling(temporal_features, training=training)

        # Main classification output
        logits = self.classification_head(pooled, training=training)
        logits = tf.squeeze(logits, axis=-1)

        return tf.nn.softmax(logits, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "n_heads": self.n_heads,
                "hidden": self.hidden,
                "transformer_dim": self.transformer_dim,
                "num_layers": self.num_layers,
                "dropout_rate": self.dropout_rate,
                "max_length": self.max_length,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    # Create test data
    batch_size = 1
    seq_length = 101
    input_dim = 27

    model = LinformerClassifier(
        num_classes=7,
        n_heads=2,
        hidden=64,
        transformer_dim=16,
        max_length=101,
        num_layers=1,
        linformer_dim=64,
    )

    dummpy_input = tf.random.uniform([batch_size, seq_length, input_dim])
    out = model(dummpy_input)
    print(out)
    model.summary()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    converter.target_spec.supported_types = [tf.float32]
    converter._experimental_lower_tensor_list_ops = False
    converter.experimental_enable_resource_variables = False
    tflite_model = converter.convert()
    output_path = "class.tflite"
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"Model saved as {output_path}")
