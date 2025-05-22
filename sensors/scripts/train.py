import argparse
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf2onnx
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import trange

# from sensors.models.H2_scaled import Conv_Attn_Conv_Scaled
from sensors.models.H3_scaled import Conv_Attn_Conv_Scaled
from sensors.utils.dataset import create_sequence_dataset
from sensors.utils.onnx import create_standalone_model_with_embedded_scaling


def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model.")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--feature_length", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hidden_layers", type=int, default=32)
    parser.add_argument("--pos_weight", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--base_dir", type=Path, default="data_processed")
    parser.add_argument("--gamma", type=float, default=0.975)
    parser.add_argument("--save_dir", type=Path, default="training")
    return parser.parse_args()


def train_one_epoch(model, train_ds, optimizer):
    epoch_loss = []
    y_true, y_pred = [], []

    for x_batch, y_batch in train_ds:
        y_batch = tf.cast(y_batch, tf.float32)

        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            logits = tf.squeeze(logits, axis=-1)
            loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_batch, logits=logits, pos_weight=0.01)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        preds = tf.sigmoid(logits)
        preds = tf.cast(preds > 0.5, tf.float32)

        y_true.extend(y_batch.numpy().flatten())
        y_pred.extend(preds.numpy().flatten())
        epoch_loss.append(loss.numpy())

    metrics = compute_metrics(y_true, y_pred, epoch_loss)
    return metrics


def validate_one_epoch(model, val_ds):
    epoch_loss = []
    y_true, y_pred = [], []

    for x_batch, y_batch in val_ds:
        y_batch = tf.cast(y_batch, tf.float32)
        logits = model(x_batch, training=False)
        logits = tf.squeeze(logits, axis=-1)
        loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_batch, logits=logits, pos_weight=0.01)
        loss = tf.reduce_mean(loss)

        preds = tf.sigmoid(logits)
        preds = tf.cast(preds > 0.5, tf.float32)

        y_true.extend(y_batch.numpy().flatten())
        y_pred.extend(preds.numpy().flatten())
        epoch_loss.append(loss.numpy())

    metrics = compute_metrics(y_true, y_pred, epoch_loss)
    print(metrics)
    return metrics


def compute_metrics(y_true, y_pred, losses):
    return {
        "loss": np.mean(losses),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }


def train(model: tf.keras.Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, args):
    model_save_path = Path(args.save_dir) / str(uuid.uuid4())

    train_writer = tf.summary.create_file_writer(str(model_save_path / "results" / "train"))
    val_writer = tf.summary.create_file_writer(str(model_save_path / "results" / "val"))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=args.learning_rate,
        decay_steps=1,
        decay_rate=args.gamma,
        staircase=True,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    checkpoint_path = model_save_path / "best_model.keras"

    best_val_loss = float("inf")
    for epoch in trange(args.epochs):
        train_metrics = train_one_epoch(model, train_ds, optimizer)
        val_metrics = validate_one_epoch(model, val_ds)

        # Logging
        with train_writer.as_default():
            tf.summary.scalar("loss", train_metrics["loss"], step=epoch + 1)
            tf.summary.scalar("accuracy", train_metrics["accuracy"], step=epoch + 1)
            tf.summary.scalar("precision", train_metrics["precision"], step=epoch + 1)
            tf.summary.scalar("recall", train_metrics["recall"], step=epoch + 1)
            tf.summary.scalar("f1", train_metrics["f1"], step=epoch + 1)

        with val_writer.as_default():
            tf.summary.scalar("loss", val_metrics["loss"], step=epoch + 1)
            tf.summary.scalar("accuracy", val_metrics["accuracy"], step=epoch + 1)
            tf.summary.scalar("precision", val_metrics["precision"], step=epoch + 1)
            tf.summary.scalar("recall", val_metrics["recall"], step=epoch + 1)
            tf.summary.scalar("f1", val_metrics["f1"], step=epoch + 1)

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            model.save(checkpoint_path)

    best_model = tf.keras.models.load_model(checkpoint_path)
    # best_model = wrap_model_with_scaler_file(best_model, "standardscaler.pkl")
    best_model = create_standalone_model_with_embedded_scaling(best_model, "StandardScaler.pkl")
    dummy_input = tf.random.uniform([1, 101, 27], dtype=tf.float32)
    _ = best_model(dummy_input)

    # onnx save
    input_signature = [tf.TensorSpec([1, 101, 27], tf.float32, name='x')]

    @tf.function(input_signature=input_signature)
    def model_inference(x):
        return best_model(x)

    # onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature, opset=16)
    onnx_model, _ = tf2onnx.convert.from_function(model_inference, input_signature=input_signature, opset=16,
                                                  output_path=str(model_save_path / "best_model.onnx"))

    # onnx.save(onnx_model, model_save_path / "best_model.onnx")

    def representative_data_gen():

        temp_ds = create_sequence_dataset(
            args.base_dir / "train", batch_size=1)
        for x_batch, _ in temp_ds.take(10):
            yield [x_batch]

    # Convert to TFLite
    concrete_func = model_inference.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    with open(model_save_path / "best_model_int8.tflite", "wb") as f:
        f.write(tflite_model)


def main():
    args = parse_args()

    model = Conv_Attn_Conv_Scaled(
        input_dim=args.feature_length,
        n_heads=args.heads,
        hidden=args.hidden_layers,
    )

    train_ds = create_sequence_dataset(
        args.base_dir / "train", batch_size=args.batch_size
    )

    val_ds = create_sequence_dataset(args.base_dir / "val", batch_size=args.batch_size)

    train(model, train_ds, val_ds, args)


if __name__ == "__main__":
    main()
