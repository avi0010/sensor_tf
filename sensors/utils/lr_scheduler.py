import math

import tensorflow as tf


class LinearWarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        max_lr: float,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int,
        gamma: float = 0.975,
    ):
        super(LinearWarmupExponentialDecay, self).__init__()
        self.max_lr = max_lr
        self.warmup_steps = int(warmup_epochs * steps_per_epoch)
        self.total_steps = total_epochs * steps_per_epoch
        self.gamma = gamma

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # Linear warmup phase
        warmup_lr = self.max_lr * step / self.warmup_steps

        exponential_lr = self.max_lr * self.gamma**(step - self.warmup_steps)

        # Choose between warmup and cosine based on current step
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: exponential_lr,
        )

    def get_config(self):
        return {
            "max_lr": self.max_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "gamma": self.gamma,
        }
