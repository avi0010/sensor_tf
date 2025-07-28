import math

import tensorflow as tf


class LinearWarmupExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        max_lr: float,
        warmup_steps: int,
        total_steps: int,
        gamma: float = 0.975,
    ):
        super(LinearWarmupExponentialDecay, self).__init__()
        self.max_lr = max_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.gamma = gamma

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # Linear warmup phase
        warmup_lr = self.max_lr * step / self.warmup_steps

        # Cosine annealing phase
        # cosine_steps = step - self.warmup_steps
        # cosine_total_steps = self.total_steps - self.warmup_steps
        # cosine_decay = 0.5 * (1 + tf.cos(math.pi * cosine_steps / cosine_total_steps))
        # cosine_lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        exponential_lr = self.max_lr * self.gamma**step

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
