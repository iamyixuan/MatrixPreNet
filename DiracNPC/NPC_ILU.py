import numpy as np
import jax
import jax.numpy as jnp
import optax
import wandb
from NeuralPC.utils.data import split_idx, create_dataLoader
from NeuralPC.model.CNNs_flax import Encoder_Decoder
from NeuralPC.train.TrainFlax import train_val, init_train_state
from NeuralPC.utils.dirac import DDOpt
from functools import partial
from flax.training import orbax_utils
import orbax
import warnings
import pickle

# cwarnings.simplefilter('error')

with open("../../datasets/Dirac/precond_data/ILU.pickle", "rb") as f:
    data = pickle.load(f)

U1 = data["U1"]
M = data["M"]

data = jnp.asarray(U1)
M = jnp.asarray(M)
data = jnp.transpose(data, [0, 2, 3, 1])  # shape B, X, T, 2

trainIdx, valIdx = split_idx(data.shape[0], 42)
trainData = data[trainIdx]
valData = data[valIdx]

trainM = M[trainIdx]
valM = M[valIdx]

TrainLoader = create_dataLoader(
    data=[np.array(trainData), np.array(trainM)],
    batchSize=32,
    kappa=0.276,
    shuffle=True,
    dataset="ILU",
)
ValLoader = create_dataLoader(
    data=[np.array(valData), np.array(valM)],
    kappa=0.276,
    batchSize=32,
    shuffle=False,
    dataset="ILU",
)


epochs = 500
learning_rate = 1e-3
h_ch = 16

model = Encoder_Decoder(2, 4, h_ch, (3, 3))

key = jax.random.PRNGKey(121)
state = init_train_state(model, key, (1, 8, 8, 2), learning_rate)
LOG = True

if LOG:
    run = wandb.init(
        # Set the project where this run will be logged
        project="reconstructDiracConfig",
        name="NN-ConLinearOpt-ILU-2",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "h_ch": h_ch,
        },
    )

final_state = train_val(
    TrainLoader,
    ValLoader,
    state,
    epochs,
    diracOpt=DDOpt,
    model=model,
    verbose=True,
    log=LOG,
)

ckpt = {"model": final_state}
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
orbax_checkpointer.save(
    "/Users/yixuan.sun/Documents/projects/Preconditioners/MatrixPreNet/checkpoints/NN-PCG/model_ILU",
    ckpt,
    save_args=save_args,
)
