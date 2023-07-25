import numpy as np
import jax.numpy as jnp


class MinMaxScaler:
    def __init__(self, data, min_=0, max_=1) -> None:
        self.data_min = np.min(data)
        self.data_max = np.max(data)
        self.min_ = min_
        self.max_ = max_

    def transform(self, x):
        d_diff = self.data_max - self.data_min + 1e-8
        s_diff = self.max_ - self.min_
        return (x - self.data_min) / d_diff * s_diff + self.min_

    def inverse_transform(self, x):
        d_diff = self.data_max - self.data_min + 1e-8
        s_diff = self.max_ - self.min_
        return (x - self.min_) / s_diff * d_diff + self.data_min


class StandardScaler:
    def __init__(self, data):
        self.std = jnp.std(data, axis=0)
        self.mean = jnp.mean(data, axis=0)

    def transform(self, x):
        return (x - self.mean) / (self.std + 1e-6)

    def inverse_transform(self, x):
        return x * self.std + self.mean


