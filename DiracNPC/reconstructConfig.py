import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from NeuralPC.model.CNNs_flax import Encoder_Decoder
from NeuralPC.train.TrainFlax import init_train_state, train_val
from NeuralPC.utils.data import create_dataLoader, split_idx



data = np.load(
    "../../datasets/Dirac/precond_data/config.l8-N1600-b2.0-k0.276-unquenched.x.npy"
)
data = jnp.asarray(data)
data = jnp.transpose(data, [0, 2, 3, 1])

# expoential transform
data_exp = np.exp(1j * data)
# print(data[0])
dataReal = data_exp.real
dataImag = data.imag
dataComb = jnp.concatenate([dataReal, dataImag], axis=-1)

trainIdx, valIdx = split_idx(dataComb.shape[0], 42)
trainData = dataComb[trainIdx]
valData = dataComb[valIdx]

TrainLoader = create_dataLoader(np.array(trainData), 16, True)
ValLoader = create_dataLoader(np.array(valData), 16, False)


epochs = 10000
learning_rate = 1e-4
h_ch = 64
data = "config_exp"

model = Encoder_Decoder(4, 4, h_ch, (3, 3))
# train the autoecoder
run = wandb.init(
    # Set the project where this run will be logged
    project="reconstructDiracConfig",
    name="MoreData",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "h_ch": h_ch,
    },
)
key = jax.random.PRNGKey(121)
state = init_train_state(model, key, (1, 8, 8, 4), learning_rate)


final_state = train_val(TrainLoader, ValLoader, state, epochs, verbose=False)
