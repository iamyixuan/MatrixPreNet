import petsc4py

petsc4py.init()

import sys
import numpy as np
from petsc4py import PETSc


def generate_random_matrix(size):
    # Generate a random matrix of given size
    np.random.seed(123)
    return np.random.rand(size, size)


def jacobi_preconditioner(matrix):
    # Create a PETSc matrix and vector
    A = PETSc.Mat().createAIJ(matrix.shape)
    A.setFromOptions()
    A.setPreallocationNNZ([3] * matrix.shape[0])
    A.setUp()

    b = PETSc.Vec().createSeq(matrix.shape[0])
    b.setArray(np.random.rand(matrix.shape[0]))  # Set b to a vector with random values

    x = PETSc.Vec().createSeq(matrix.shape[0])

    # Create the preconditioner
    pc = PETSc.PC().create()
    pc.setType(PETSc.PC.Type.JACOBI)
    pc.setOperators(A)
    pc.setUp()

    # Assemble the matrix
    A.assemble()

    # Apply the preconditioner
    pc.apply(b, x)

    # Return the preconditioned vector
    return x.getArray()


if __name__ == "__main__":
    matrix_size = 500  # Size of the random matrix
    num_examples = 1000  # Number of training examples

    input_matrices = []  # List to store input matrices
    output_vectors = []  # List to store output vectors

    # Modify PETSc options to disable the allocation error check
    sys.argv.append("-mat_new_nonzero_allocation_err")
    sys.argv.append("false")

    # Initialize PETSc again with the modified options
    petsc4py.init(sys.argv)

    # Generate training examples
    for i in range(num_examples):
        # Generate a random matrix
        random_matrix = generate_random_matrix(matrix_size)

        # Compute the preconditioned vector
        preconditioned_vector = jacobi_preconditioner(random_matrix)

        # Print the input matrix and output vector
        print("Training Example", i + 1)
        print("Input Matrix:")
        print(random_matrix)
        print("Output Vector:")
        print(preconditioned_vector)
        print()

        # Append the input matrix and output vector to the respective lists
        input_matrices.append(random_matrix)
        output_vectors.append(preconditioned_vector)

    # Convert the lists to NumPy arrays
    input_matrices = np.array(input_matrices)
    output_vectors = np.array(output_vectors)

    # Save the input matrices and output vectors to a NumPy file
    np.savez(
        "training_data_500.npz",
        input_matrices=input_matrices,
        output_vectors=output_vectors,
    )

    print("Training examples saved to 'training_data.npz'.")
