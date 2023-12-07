import jax
import jax.numpy as jnp
from jax.lax import conv_general_dilated


def sepRealImag(v):
    real = v.real
    imag = v.imag
    return jnp.concatenate([real, imag], axis=-1)


def mergRealImag(v):
    """
    The input v has shape (B, X, T, 4)
    the last two dims are real and imaginary part

    return shaep (B, X, T, 2), with complex entries
    that can be used in CG
    """
    v = v[..., :2] + 1j * v[..., -2:]
    return v


def linearConvOpt(input, kernels):
    """
    Apply a linear convolutional operator to the input.

    :param input: Input tensor of shape (B, X, T, 2)
    :param kernels: Convolution kernels of shape (B, K_X, K_T, 2, 2)
    :return: Output tensor of shape (B, X, T, 2)
    """
    # input has complex entries
    # separate the real and imag part to form (B, X, T, 4)
    # Therefore the kernel should have shape (B, K_X, K_T, 4, 4)
    # Define convolution parameters

    input = sepRealImag(input)
    strides = (1, 1)  # No stride
    padding = "SAME"  # Pad to keep input and output shape same

    # Perform depthwise convolution
    output = conv_general_dilated(
        lhs=input,  # input
        rhs=kernels,  # kernel
        window_strides=strides,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),  # Specify dimension order
    )
    out = output.squeeze()
    out = mergRealImag(out)
    return out


def linearOpt(input, w, b):
    """
    input has shape (1, 8, 8, 2)
    w shape [1*8*8*2,  1*8*8*2]
    b shape 1*8*8*2
    """
    input = input.ravel()
    out = input @ w + b
    return out.reshape((1, 8, 8, 2))


if __name__ == "__main__":
    # Example usage
    B, X, T = 10, 20, 20  # Example dimensions
    K_X, K_T = 3, 3  # Kernel size

    # Randomly initialize input and kernels for demonstration
    key = jax.random.PRNGKey(0)
    input = jax.random.normal(key, shape=(B, 1, X, T, 2))
    kernels = jax.random.normal(key, shape=(B, K_X, K_T, 2, 2))

    vLinearOpt = jax.vmap(linearConvOpt, in_axes=[0, 0])
    # Apply the operator
    output = vLinearOpt(input, kernels)
    print(output.shape)
