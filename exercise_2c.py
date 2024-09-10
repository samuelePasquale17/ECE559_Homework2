import numpy as np
import matplotlib.pyplot as plt


def MAT_step(matrix):
    """
    Function that given a matrix in input generates a new matrices applying the
    step function for each i-j value. The step function is defined as
    - step(x) = 1 for x >= 0
    - step(x) = 0 for x < 0

    :param matrix: Input matrix
    :return: step(x)
    """
    # Initialize new temporary matrix for the result
    result = []

    # Rows
    for row in matrix:
        new_row = []
        # Element of the row
        for element in row:
            # Step function
            if element < 0:
                new_row.append(0)
            else:
                new_row.append(1)
        # Add the new row
        result.append(new_row)

    # return the new matrix
    return result


def MAT_prod(A, B):
    """
    Function that given two matrices computes the product
    :param A: Matrix A
    :param B: Matrix B
    :return: A * B
    """
    # Get the dimensions of the matrices
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    # Check matrices dimensions (columns of A == rows of B)
    if cols_A != rows_B:
        raise ValueError("Number of columns in A must be equal to the number of rows in B.")

    # Result matrix full of zeros
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    # Calculate the product of matrices
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result


def MAT_sum(A, B):
    """
    Function that given two matrices computes the sum
    :param A: Matrix A
    :param B: Matrix B
    :return: A + B
    """
    # Get the dimensions of the matrices
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    # Check if the matrices have the same dimensions
    if rows_A != rows_B or cols_A != cols_B:
        raise ValueError("Matrices must have the same dimensions for addition.")

    # Result matrix full of zeros
    result = [[0 for _ in range(cols_A)] for _ in range(rows_A)]

    # Sum of matrices
    for i in range(rows_A):
        for j in range(cols_A):
            result[i][j] = A[i][j] + B[i][j]

    return result


def boolean_func_NN(X, W, b, U, C):
    """
    Function that given the Weight matrices, the bias vectors, and the inputs computes the result through
    the analytic equation
    :param X: Input vector
    :param W: first-layer Weight matrix
    :param b: first-layer Bias vector
    :param U: second-layer Weight matrix
    :param C: second-layer Bias vector
    :return: y
    """
    return MAT_step(MAT_sum(MAT_prod(U, MAT_step(MAT_sum(MAT_prod(W, X), b))), C))


def plot_points(points, color):
    """
    Function that given a set of points and a color, plots the points using matplotlib
    :param points: vector of points
    :param color: vector of either 0's or 1's
    :return: None
    """

    # Defining 0 as blue and 1 as red
    colors = ['blue' if c == 0 else 'red' for c in color]

    # plot
    plt.scatter(points[:, 0], points[:, 1], c=colors, alpha=0.6)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Scatter Plot of Random Points with Colors')
    plt.grid(True)
    plt.show()


def main():
    # matrices and vectors of weights and biases for the given
    W = [[1, -1],
         [-1, -1],
         [0, -1]]

    b = [[1],
         [1],
         [-1]]

    U = [[1, 1, -1]]

    C = [[-1.5]]

    # Sample 1,000 random points x from a uniform distribution over the square [-2, 2]^2
    num_points = 1000
    x_points = []

    # 1,000 points
    for _ in range(num_points):
        point = np.random.uniform(-2, 2, 2)
        x_points.append(point)

    x_points = np.array(x_points)

    # compute y by the NN
    y_res = []
    for point in x_points:
        x = [[point[0]],
             [point[1]]]
        y = boolean_func_NN(x, W, b, U, C)
        y_res.append(y[0][0])

    # plot
    plot_points(x_points, y_res)


main()