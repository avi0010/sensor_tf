import argparse
import json
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm, trange

from sensors.config import CLASSIFICATION_MODEL_OUTPUTS
from sensors.models.H3_classification import LinformerClassifier
from sensors.utils.dataset_tfRecord import create_tfrecord_dataset
from sensors.utils.lr_scheduler import LinearWarmupExponentialDecay
from sensors.utils.plotting import save_confusion_matrix_png


def parse_args():
    parser = argparse.ArgumentParser(description="Train a deep learning model.")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_layers", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--linformer_dim", type=int, default=64)
    parser.add_argument("--base_dir", type=Path, default="data_processed")
    parser.add_argument("--gamma", type=float, default=0.975)
    parser.add_argument("--save_dir", type=Path, default="classification_models")
    parser.add_argument("--class_weights", default="100,1,1,1,1,1,1")
    return parser.parse_args()


@tf.function
def train_step(model, x_batch, y_batch, optimizer, metrics, class_weights=None):
    with tf.GradientTape() as tape:
        logits = model(x_batch, training=True)

        if class_weights is not None:
            sample_weights = tf.gather(class_weights, y_batch)
            y_batch = tf.one_hot(tf.cast(y_batch, tf.int32), len(CLASSIFICATION_MODEL_OUTPUTS))
            loss = tf.keras.losses.categorical_crossentropy(y_true=y_batch, y_pred=logits, label_smoothing=0.05)
            loss = loss * sample_weights
        else:
            y_batch = tf.one_hot(tf.cast(y_batch, tf.int32), len(CLASSIFICATION_MODEL_OUTPUTS))
            loss = tf.keras.losses.categorical_crossentropy(y_true=y_batch, y_pred=logits, label_smoothing=0.05)

        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # Get predictions
    preds = tf.argmax(logits, axis=-1)

    # Update metrics
    metrics["loss"].update_state(loss)
    metrics["accuracy"].update_state(y_batch, preds)

    return preds


def train_one_epoch(model, train_ds, optimizer, class_weights, train_ds_length):
    metrics = create_metrics()

    # Collect all predictions and labels for confusion matrix
    all_y_true = []
    all_y_pred = []

    for x_batch, y_batch in tqdm(train_ds, leave=False, total=train_ds_length):
        y_batch = tf.cast(y_batch, tf.int32)  # Labels should be integers for multi-class
        preds = train_step(model, x_batch, y_batch, optimizer, metrics, class_weights)

        # Collect for confusion matrix
        all_y_true.append(y_batch.numpy())
        all_y_pred.append(preds.numpy())

    y_true_concat = np.concatenate(all_y_true)
    y_pred_concat = np.concatenate(all_y_pred)
    confusion_matrix = tf.math.confusion_matrix(
        y_true_concat, y_pred_concat, num_classes=len(CLASSIFICATION_MODEL_OUTPUTS)
    ).numpy()

    return calculate_epoch_metrics(metrics, confusion_matrix)


@tf.function
def val_step(model, x_batch, y_batch, metrics, class_weights=None):
    logits = model(x_batch, training=False)

    if class_weights is not None:
        sample_weights = tf.gather(class_weights, y_batch)
        y_batch = tf.one_hot(tf.cast(y_batch, tf.int32), len(CLASSIFICATION_MODEL_OUTPUTS))
        loss = tf.keras.losses.categorical_crossentropy(y_true=y_batch, y_pred=logits, label_smoothing=0.05)
        loss = loss * sample_weights
    else:
        y_batch = tf.one_hot(tf.cast(y_batch, tf.int32), len(CLASSIFICATION_MODEL_OUTPUTS))
        loss = tf.keras.losses.categorical_crossentropy(y_true=y_batch, y_pred=logits, label_smoothing=0.05)

    loss = tf.reduce_mean(loss)
    preds = tf.argmax(logits, axis=-1)

    # Update metrics
    metrics["loss"].update_state(loss)
    metrics["accuracy"].update_state(y_batch, preds)

    return preds


def validate_one_epoch(model, val_ds, class_weights, model_save_path, epoch):
    metrics = create_metrics()

    # Collect all predictions and labels for confusion matrix
    all_y_true = []
    all_y_pred = []

    for x_batch, y_batch in val_ds:
        y_batch = tf.cast(y_batch, tf.int32)
        preds = val_step(model, x_batch, y_batch, metrics, class_weights)

        all_y_true.append(y_batch.numpy())
        all_y_pred.append(preds.numpy())

    # Calculate confusion matrix
    y_true_concat = np.concatenate(all_y_true)
    y_pred_concat = np.concatenate(all_y_pred)
    confusion_matrix = tf.math.confusion_matrix(
        y_true_concat, y_pred_concat, num_classes=len(CLASSIFICATION_MODEL_OUTPUTS)
    ).numpy()

    confusion_matrix_path = model_save_path / 'confusion_matrix'
    confusion_matrix_path.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix_png(confusion_matrix, epoch, confusion_matrix_path, "val", CLASSIFICATION_MODEL_OUTPUTS)

    return calculate_epoch_metrics(metrics, confusion_matrix)


def calculate_epoch_metrics(metrics, confusion_matrix):
    """Calculate precision, recall, F1 from confusion matrix for multi-class"""

    # Calculate per-class metrics
    num_classes = confusion_matrix.shape[0]
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Calculate macro averages
    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)
    macro_f1 = sum(f1_scores) / len(f1_scores)

    return {
        "loss": metrics["loss"].result().numpy(),
        "accuracy": metrics["accuracy"].result().numpy(),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "per_class_f1": f1_scores,
    }


def create_metrics():
    return {
        "loss": tf.keras.metrics.Mean(name="loss"),
        "accuracy": tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
    }


def train(
        model: tf.keras.Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        args,
):
    model_save_path = Path(args.save_dir) / str(uuid.uuid4())

    # Save to JSON file
    args_dict = vars(args)
    args_file_path = model_save_path / "arguments.json"

    train_writer = tf.summary.create_file_writer(
        str(model_save_path / "results" / "train")
    )
    val_writer = tf.summary.create_file_writer(str(model_save_path / "results" / "val"))

    train_ds_length = sum(1 for _ in train_ds)

    with open(args_file_path, 'w') as f:
        json.dump(args_dict, f, indent=4, default=str)

    lr_schedule = LinearWarmupExponentialDecay(
        max_lr=args.learning_rate,
        warmup_epochs=10,
        total_epochs=args.epochs,
        steps_per_epoch=train_ds_length,
        gamma=args.gamma,
    )

    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)

    # Parse class weights if provided
    class_weights = None

    checkpoint_path = model_save_path / "best_model.keras"

    best_val_f1 = 0.0
    for epoch in trange(args.epochs):

        if args.class_weights:
            weights = [float(w) for w in args.class_weights.split(",")]
            weights[0] *= 1.1 ** epoch
            class_weights = tf.constant(weights)

        train_metrics = train_one_epoch(
            model,
            train_ds,
            optimizer,
            class_weights,
            train_ds_length,
        )

        val_metrics = validate_one_epoch(
            model,
            val_ds,
            class_weights,
            model_save_path,
            epoch,
        )

        # Logging
        with train_writer.as_default():
            tf.summary.scalar("loss", train_metrics["loss"], step=epoch + 1)
            tf.summary.scalar("accuracy", train_metrics["accuracy"], step=epoch + 1)
            tf.summary.scalar("macro_precision", train_metrics["macro_precision"], step=epoch + 1)
            tf.summary.scalar("macro_recall", train_metrics["macro_recall"], step=epoch + 1)
            tf.summary.scalar("macro_f1", train_metrics["macro_f1"], step=epoch + 1)
            tf.summary.scalar("learning_rate", optimizer.learning_rate.numpy(), step=epoch + 1)

            for class_idx in range(len(CLASSIFICATION_MODEL_OUTPUTS)):
                gas_id = CLASSIFICATION_MODEL_OUTPUTS[class_idx]
                with tf.summary.create_file_writer(
                        str(model_save_path / "results" / "train" / f"gas_{gas_id}")).as_default():
                    tf.summary.scalar(f"precision", train_metrics["per_class_precision"][class_idx],
                                      step=epoch + 1)
                    tf.summary.scalar(f"recall", train_metrics["per_class_recall"][class_idx],
                                      step=epoch + 1)
                    tf.summary.scalar(f"f1", train_metrics["per_class_f1"][class_idx], step=epoch + 1)

        with val_writer.as_default():
            tf.summary.scalar("loss", val_metrics["loss"], step=epoch + 1)
            tf.summary.scalar("accuracy", val_metrics["accuracy"], step=epoch + 1)
            tf.summary.scalar("macro_precision", val_metrics["macro_precision"], step=epoch + 1)
            tf.summary.scalar("macro_recall", val_metrics["macro_recall"], step=epoch + 1)
            tf.summary.scalar("macro_f1", val_metrics["macro_f1"], step=epoch + 1)

            for class_idx in range(len(CLASSIFICATION_MODEL_OUTPUTS)):
                gas_id = CLASSIFICATION_MODEL_OUTPUTS[class_idx]
                with tf.summary.create_file_writer(
                        str(model_save_path / "results" / "val" / f"gas_{gas_id}")).as_default():
                    tf.summary.scalar(f"precision", train_metrics["per_class_precision"][class_idx],
                                      step=epoch + 1)
                    tf.summary.scalar(f"recall", train_metrics["per_class_recall"][class_idx],
                                      step=epoch + 1)
                    tf.summary.scalar(f"f1", train_metrics["per_class_f1"][class_idx], step=epoch + 1)

        # Save best model based on macro F1
        model.save(model_save_path / f"epoch-{epoch + 1}_f-{val_metrics['macro_f1']}.keras")
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            model.save(checkpoint_path)

    # Model export remains the same...
    best_model = tf.keras.models.load_model(checkpoint_path)
    dummy_input = tf.random.uniform([1, 101, 27], dtype=tf.float32)
    _ = best_model(dummy_input)

    # onnx save
    input_signature = [tf.TensorSpec([1, 101, 27], tf.float32, name="x")]

    @tf.function(input_signature=input_signature)
    def model_inference(x):
        return best_model(x)

    # Convert to TFLite
    concrete_func = model_inference.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    converter.target_spec.supported_types = [tf.float32]
    converter._experimental_lower_tensor_list_ops = False
    converter.experimental_enable_resource_variables = False
    tflite_model = converter.convert()

    with open(
            model_save_path
            / f"H3-heads_{args.heads}-linformer_{args.linformer_dim}.tflite",
            "wb",
    ) as f:
        f.write(tflite_model)


def main():
    args = parse_args()

    model = LinformerClassifier(
        n_heads=args.heads,
        hidden=args.hidden_layers,
        linformer_dim=args.linformer_dim,
        num_classes=len(CLASSIFICATION_MODEL_OUTPUTS)
    )

    train_ds = create_tfrecord_dataset(
        args.base_dir / "train_class.tfrecord", batch_size=args.batch_size
    )

    val_ds = create_tfrecord_dataset(
        args.base_dir / "val_class.tfrecord", batch_size=args.batch_size, shuffle=False
    )

    train(model, train_ds, val_ds, args)


if __name__ == "__main__":
    main()
