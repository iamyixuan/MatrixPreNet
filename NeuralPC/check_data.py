import numpy as np
import matplotlib.pyplot as plt

data = np.load(
    "../../datasets/Dirac/precond_data/config.l64-N32-b2.0-k0.276-unquenched-test.x.npy"
)

U1_field = np.exp(1j * data)
print(U1_field.shape)
