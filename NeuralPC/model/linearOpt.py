import jax
import jax.numpy as jnp
from jax.lax import conv_general_dilated

def linearConvOpt(input, kernels):
    """
    Apply a linear convolutional operator to the input.
    
    :param input: Input tensor of shape (B, X, T, 2)
    :param kernels: Convolution kernels of shape (B, K_X, K_T, 2, 2)
    :return: Output tensor of shape (B, X, T, 2)
    """
    # Define convolution parameters
    strides = (1, 1)  # No stride
    padding = 'SAME'  # Pad to keep input and output shape same

    # Perform depthwise convolution
    output = conv_general_dilated(
        lhs=input,  # input
        rhs=kernels,  # kernel
        window_strides=strides,
        padding=padding,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),  # Specify dimension order
    )

    return output.squeeze()

if __name__ == '__main__':
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