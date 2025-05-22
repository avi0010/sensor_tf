from pickle import load

import tensorflow as tf
from keras.saving import register_keras_serializable


def wrap_model_with_scaler_file(base_model: tf.keras.Model, scaler_path: str) -> tf.keras.Model:
    """
    Wraps a base model with input scaling (StandardScaler from scaler_path) and sigmoid output.

    Parameters:
        base_model (tf.keras.Model): The trained Keras model without scaling or sigmoid.
        scaler_path (str): Path to the joblib-serialized StandardScaler file (.gz).

    Returns:
        tf.keras.Model: A new model that applies input scaling and sigmoid activation.
    """
    # Define expected input feature order in the model
    actual_feature_order = [
        "Temperature Deriv.", "Humidity Deriv.",
        "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7",
        "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15",
        "S16", "S17", "S18", "S19", "S20", "S21", "S22", "S23", "BME"
    ]

    # Load the StandardScaler
    scaler = load(open("StandardScaler.pkl", "rb"))

    # Reorder mean and std to match actual feature order
    expected_feature_order = scaler.feature_names_in_
    index_mapping = [expected_feature_order.tolist().index(f) for f in actual_feature_order]

    mean_reordered = scaler.mean_[index_mapping]
    std_reordered = scaler.scale_[index_mapping]

    # Define the wrapper model
    @register_keras_serializable()
    class ModelWithScaler(tf.keras.Model):
        def __init__(self, base_model, **kwargs):
            super().__init__()
            self.base_model = base_model
            self.mean = tf.reshape(tf.constant(mean_reordered, dtype=tf.float32), [1, 1, -1])
            self.std = tf.reshape(tf.constant(std_reordered, dtype=tf.float32), [1, 1, -1])

        def call(self, inputs, training=False):
            x = (inputs - self.mean) / self.std
            x = self.base_model(x, training=training)
            return tf.nn.sigmoid(x)

        def get_config(self):
            config = super().get_config()
            config.update({
                'mean': self.mean.numpy().tolist(),
                'std': self.std.numpy().tolist(),
                # 'base_model' is not serializable by default, so it’s not included here.
            })

            return config

        @classmethod
        def from_config(cls, config):
            mean = tf.constant(config.pop('mean'), dtype=tf.float32)
            std = tf.constant(config.pop('std'), dtype=tf.float32)
            return cls(base_model=None, mean=mean, std=std, **config)

    return ModelWithScaler(base_model)

def create_standalone_model_with_embedded_scaling(base_model, scaler_path):
    """Create a standalone model with embedded scaling constants"""

    # Load scaler
    scaler = load(open(scaler_path, "rb"))
    actual_feature_order = [
        "Temperature Deriv.", "Humidity Deriv.",
        "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7",
        "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15",
        "S16", "S17", "S18", "S19", "S20", "S21", "S22", "S23", "BME"
    ]
    expected_feature_order = scaler.feature_names_in_
    index_mapping = [expected_feature_order.tolist().index(f) for f in actual_feature_order]
    mean_reordered = scaler.mean_[index_mapping]
    std_reordered = scaler.scale_[index_mapping]

    # Create input layer
    inputs = tf.keras.Input(shape=(101, 27), name='input')

    # Create scaling layer using Lambda
    def scaling_function(x):
        mean = tf.constant(mean_reordered.reshape(1, 1, -1), dtype=tf.float32)
        std = tf.constant(std_reordered.reshape(1, 1, -1), dtype=tf.float32)
        return (x - mean) / std

    scaled_inputs = tf.keras.layers.Lambda(scaling_function, name='scaling')(inputs)

    # Apply base model
    base_output = base_model(scaled_inputs)

    # Apply sigmoid
    outputs = tf.keras.layers.Activation('sigmoid', name='sigmoid')(base_output)

    # Create the complete model
    complete_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='complete_model')

    return complete_model
