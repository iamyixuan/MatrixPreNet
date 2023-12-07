import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import wandb
from NeuralPC.utils.data import split_idx, create_dataLoader
from NeuralPC.model.EqxModel import EncoderDecoder, train, change_model_dtype, random_b
from NeuralPC.utils.dirac import DDOpt
from NeuralPC.utils.losses import PCG_loss, testLoss


data = np.load(
    "../../datasets/Dirac/precond_data/config.l8-N1600-b2.0-k0.276-unquenched.x.npy"
)
data = jnp.asarray(data)
data = jnp.transpose(data, [0, 2, 3, 1])  # shape B, X, T, 2


# expoential transform
data_exp = np.exp(1j * data)
# print(data[0])
dataReal = data_exp.real
dataImag = data.imag
# dataComb = jnp.concatenate([dataReal, dataImag], axis=-1)

trainIdx, valIdx = split_idx(data_exp.shape[0], 42)
trainData = data_exp[trainIdx]
valData = data_exp[valIdx]

TrainLoader = create_dataLoader(
    data=np.array(trainData), batchSize=256, kappa=0.276, shuffle=True
)
ValLoader = create_dataLoader(
    data=np.array(valData), kappa=0.276, batchSize=256, shuffle=False
)


epochs = 10
learning_rate = 1e-4

model = EncoderDecoder(key=jax.random.PRNGKey(0))
# ls = testLoss(model, data_exp)
# loss, grads = eqx.filter_value_and_grad(testLoss)(model, data_exp)
# print(loss, grads)

b = random_b(jax.random.PRNGKey(0), (1600, 8, 8, 2))
U1 = jnp.moveaxis(data_exp, -1, 1)
ls = PCG_loss(NN=model, U1=U1, b=b, steps=100, kappa=0.276, operator=DDOpt)
loss, grads = eqx.filter_value_and_grad(PCG_loss)(
    model, U1=U1, b=b, steps=100, kappa=0.276, operator=DDOpt
)
print(ls, loss)

# model = change_model_dtype(model, jnp.complex64)


# trainedModel = train(model=model,
#                      trainLoader=TrainLoader,
#                      valLoader=ValLoader,
#                      loss_fn=PCG_loss,
#                      epochs=epochs,
#                      key=jax.random.PRNGKey(1))


"""
run = wandb.init(
    # Set the project where this run will be logged
    project="reconstructDiracConfig",
    name='NN-PCG',
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "h_ch": h_ch,
    })
final_state = train_val(TrainLoader, ValLoader, state, epochs, diracOpt=DiracOperator, model=model, verbose=True, log=True)

ckpt = {'model': final_state}
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
orbax_checkpointer.save('/Users/yixuan.sun/Documents/projects/Preconditioners/MatrixPreNet/checkpoints/NN-PCG/model', ckpt, save_args=save_args)
"""
