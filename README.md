###Power Method and Shifted Higher Order Power Method

This Python script implements the Power Method and Shifted Higher Order Power Method algorithms for finding the dominant eigenvalue and eigenvector of a matrix, as well as the Z-spectral radius of a tensor, respectively.

###Overview
The script contains two main functions:

power_method: This function implements the Power Method for finding the dominant eigenvalue and eigenvector of a matrix. It takes as input a matrix A, an initial guess for the eigenvector x0, a shift parameter alpha, and optional parameters for maximum number of iterations (max_iter) and convergence tolerance (tol). It returns a tuple containing the dominant eigenvalue and eigenvector.
shifted_higher_order_power_method: This function implements the Shifted Higher Order Power Method for finding the Z-spectral radius of a tensor. It takes as input a tensor A, with optional parameters for the number of initial points to use (num_initial_points), maximum number of iterations for the power method (max_iter), and convergence tolerance (tol). It returns the Z-spectral radius of the tensor A.

###Usage
To use the script:

Import numpy library.
Call the shifted_higher_order_power_method function with the desired tensor A as input.
The Z-spectral radius of the tensor A will be printed.
