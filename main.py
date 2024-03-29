import numpy as np


def power_method(A, x0, alpha, max_iter=1000, tol=1e-6):
    """
    Power method for finding the dominant eigenvalue and eigenvector of a matrix.

    Parameters:
        A (numpy.ndarray): The input matrix.
        x0 (numpy.ndarray): The initial guess for the eigenvector.
        alpha (float): The shift parameter.
        max_iter (int): Maximum number of iterations (default is 1000).
        tol (float): Tolerance for convergence (default is 1e-6).

    Returns:
        tuple: A tuple containing the dominant eigenvalue and eigenvector.
    """
    x = x0
    for _ in range(max_iter):
        x_new = np.dot(np.linalg.matrix_power(A, 2), x) + alpha * x
        x_new /= np.linalg.norm(x_new)
        lambda_new = np.dot(np.dot(x_new.T, A), x_new)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    return lambda_new, x_new


def shifted_higher_order_power_method(A, num_initial_points=100, max_iter=1000, tol=1e-6):
    """
    Shifted Higher Order Power Method for finding the Z-spectral radius of a tensor.

    Parameters:
        A (numpy.ndarray): The input tensor.
        num_initial_points (int): Number of initial points to use (default is 100).
        max_iter (int): Maximum number of iterations for power method (default is 1000).
        tol (float): Tolerance for convergence in power method (default is 1e-6).

    Returns:
        float: The Z-spectral radius of the tensor A.
    """
    lambda_list = []
    x0 = np.random.rand(A.shape[0])

    for _ in range(num_initial_points):
        lambda_i, _ = power_method(A, x0, 0, max_iter, tol)
        lambda_list.append(lambda_i)

    return max(lambda_list)


if __name__ == "__main__":
    # Tensor
    A = np.array([[1, 1 / np.sqrt(2), 1 / np.sqrt(3)],
                  [1 / np.sqrt(2), 1, 1 / np.sqrt(2)],
                  [1 / np.sqrt(3), 1 / np.sqrt(2), 1]])

    # Find the Z-spectral radius
    z_spectral_radius = shifted_higher_order_power_method(A)
    print("Z-spectral radius:", z_spectral_radius)
