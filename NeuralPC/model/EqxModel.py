import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from functools import partial
from NeuralPC.utils.dirac import DDOpt

class Encoder(eqx.Module):
    layers: list

    def __init__(self, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.layers = [
            eqx.nn.Conv2d(2, 4, kernel_size=3, key=k1),
            eqx.nn.PReLU(),
            eqx.nn.Conv2d(4, 8, kernel_size=3, key=k2),
            eqx.nn.PReLU(),
            eqx.nn.Conv2d(8, 16, kernel_size=3, key=k3),
            jnp.ravel,
            eqx.nn.Linear(64, 32, key=k4),
            eqx.nn.PReLU(),
            eqx.nn.Linear(32, 16, key=k5)
        ]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(eqx.Module):
    LinearLayers: list
    TransConvLayers: list
    def __init__(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.LinearLayers = [
            eqx.nn.Linear(16, 32, key=k1),
            eqx.nn.PReLU(),
            eqx.nn.Linear(32, 64, key=k2),
            eqx.nn.PReLU()
        ] 
        self.TransConvLayers = [
            eqx.nn.ConvTranspose2d(16, 8, kernel_size=3, key=k3),
            eqx.nn.PReLU(),
            eqx.nn.ConvTranspose2d(8, 4, kernel_size=3, key=k3),
            eqx.nn.PReLU(),
            eqx.nn.ConvTranspose2d(4, 2, kernel_size=3, key=k3),
        ]
    def __call__(self, x):
        for LinearL in self.LinearLayers:
            x = LinearL(x)
        
        x = jnp.reshape(x, (16, 2, 2))
        for ConvL in self.TransConvLayers:
            x = ConvL(x)
        return x


class EncoderDecoder(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    def __init__(self, key):
        encoderKey, decoderKey = jax.random.split(key)
        self.encoder = Encoder(encoderKey)
        self.decoder = Decoder(decoderKey)
    def __call__(self, x):
        """
        x: has shape X, T, 2
        Need to reshape to 2, X, T
        """
        x = jnp.angle(x)
        x = jnp.moveaxis(x, -1, 0)
        x = self.encoder(x)
        x = self.decoder(x)
        x = jnp.moveaxis(x, 0, -1)
        x = jnp.exp(1j * x)
        return x
    

def random_b(key, shape):
    # Generate random values for the real and imaginary parts
    real_part = 1 - jax.random.uniform(key, shape)
    imag_part = 1 - jax.random.uniform(jax.random.split(key)[1], shape)
    # Combine the real and imaginary parts
    complex_array = real_part + 1j * imag_part
    return complex_array


def train(model, trainLoader, valLoader, loss_fn, epochs, key):
    optimizer = optax.adam(learning_rate=0.001)
    optState = optimizer.init(eqx.filter(model, eqx.is_array))


    #@eqx.filter_jit
    def step(model,  optState, loss_fn=None):
        # params, static = eqx.partition(model, eqx.is_array)

        # def loss2(params, static):
        #     model = eqx.combine(params, static)
        #     return loss_fn(model)
        # ls = loss_fn(model)
        # print(ls)
        grads = eqx.filter_value_and_grad(loss_fn)(model)
        # loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        print( grads)
        updates, optState = optimizer.update(grads, optState, model)
        model = eqx.apply_updates(model, updates)
        return model, optState, #loss


    for ep in range(epochs):
        batchTrainLoss = []
        for batch in trainLoader:
            x, kappa = batch
            x = x.numpy()
            kappa = kappa[0].numpy()

            _, key = jax.random.split(key, 2)
            b = random_b(key, x.shape).astype(jnp.complex128)

            loss_fn_part = partial(loss_fn, U1=jnp.moveaxis(x, -1, 1), b=b,
                              kappa=kappa, steps=100, operator=DDOpt)
            
            model, optState, trainLossTmp = step(model, optState, loss_fn=loss_fn_part)
            batchTrainLoss.append(trainLossTmp)

        valEpochLoss = []
        for xVal, yVal in valLoader:
            valLossTmp = loss_fn(model, xVal, yVal)
            valEpochLoss.append(valLossTmp)

        trainLoss = jnp.mean(batchTrainLoss)
        valLoss = jnp.mean(valEpochLoss)
        print(f'Epoch {ep} Training Loss {trainLoss} Validation loss {valLoss}')
    return model

def to_dtype(x, new_dtype):
    if isinstance(x, jnp.ndarray):
        return x.astype(new_dtype)
    return x

def change_model_dtype(model, new_dtype):
    return jax.tree_map(lambda x: to_dtype(x, new_dtype), model)

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    model = EncoderDecoder(key)
    randomX = jax.random.uniform(shape=(10, 8, 8, 2), key=key).astype(jnp.complex128)
    out = jax.vmap(model)(randomX)
    print(out[0])