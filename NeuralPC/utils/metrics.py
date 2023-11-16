import numpy as np


def mean_squared_error(true, pred):
    return np.mean(np.power(true - pred, 2))


def r2(true, pred):
    ss_res = np.sum(np.power(true - pred, 2))
    ss_tot = np.sum(np.power(true - np.mean(true), 2))
    return 1 - ss_res / ss_tot


def mape(true, pred):
    return np.abs(true - pred).mean() / np.abs(true).mean() * 100


def test_metrics(true, pred):
    mse = mean_squared_error(true, pred)
    R2 = r2(true, pred)
    MAPE = mape(true, pred)
    return mse, R2, MAPE


class min_max_scaler:
    def __init__(self, d_min, d_max, s_min=0, s_max=100) -> None:
        self.d_min = d_min
        self.d_max = d_max
        self.s_min = s_min
        self.s_max = s_max

    def transform(self, x):
        d_diff = self.d_max - self.d_min
        s_diff = self.s_max - self.s_min
        return (x - self.d_min) / d_diff * s_diff + self.s_min

    def inverse_transform(self, x):
        d_diff = self.d_max - self.d_min
        s_diff = self.s_max - self.s_min
        return (x - self.s_min) / s_diff * d_diff + self.d_min
