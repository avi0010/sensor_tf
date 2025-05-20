import os

import numpy as np
import tensorflow as tf


def get_file_paths(data_dir):
    file_paths = []
    for exp in os.listdir(data_dir):
        exp_data_dir = os.path.join(data_dir, exp)
        if os.path.isdir(exp_data_dir):
            for fname in os.listdir(exp_data_dir):
                file_paths.append(os.path.join(exp_data_dir, fname))
    return file_paths


def load_npz_tf(file_path):
    def _load_npz(path):
        path = path.numpy().decode("utf-8")

        data = np.load(path)
        x, y = data["X"], data["y"]

        return x.astype(np.float32), y.astype(np.float32)

    x, y = tf.py_function(_load_npz, [file_path], [tf.float32, tf.float32])
    return x, y


def create_sequence_dataset(
        data_dir, batch_size=32, shuffle=True, prefetch=True,
):
    file_paths = get_file_paths(data_dir)
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)

    if shuffle:
        path_ds = path_ds.shuffle(path_ds.cardinality())

    ds = path_ds.map(load_npz_tf, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)

    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)


    return ds


if __name__ == "__main__":
    train_ds = create_sequence_dataset(
        "./data_processed/train/", batch_size=64
    ).repeat()

    for batch_x, batch_y in train_ds:
        print(batch_x.shape, batch_y.shape)
        pass
