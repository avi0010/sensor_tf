import math


class PosWeightDecaySchedule:
    def __init__(
        self,
        initial_pos_weight: float,
        min_pos_weight: float,
        total_epochs: int,
        decay_type: str = "exponential",
        decay_rate: float = 0.90,
    ):
        self.initial_pos_weight = initial_pos_weight
        self.min_pos_weight = min_pos_weight
        self.total_epochs = total_epochs
        self.decay_type = decay_type
        self.decay_rate = decay_rate

    def __call__(self, epoch):
        if self.decay_type == "exponential":
            # Exponential decay: pos_weight = initial * (decay_rate ^ epoch)
            current_pos_weight = self.initial_pos_weight * (self.decay_rate**epoch)
        elif self.decay_type == "linear":
            # Linear decay: pos_weight decreases linearly from initial to min
            progress = epoch / self.total_epochs
            current_pos_weight = self.initial_pos_weight - progress * (
                self.initial_pos_weight - self.min_pos_weight
            )
        elif self.decay_type == "cosine":
            # Cosine decay: similar to cosine annealing for learning rate
            progress = epoch / self.total_epochs
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            current_pos_weight = (
                self.min_pos_weight
                + (self.initial_pos_weight - self.min_pos_weight) * cosine_factor
            )
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")

        # Ensure pos_weight doesn't go below minimum
        return max(current_pos_weight, self.min_pos_weight)
