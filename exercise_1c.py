def boolean_func_homework(x1, x2, x3):
    """
    Function that computes the following boolean function (given in the Homework2):
    (x1 and x2 and not(x3)) or (not(x2) and x3)
    :param x1: Boolean input x1
    :param x2: Boolean input x2
    :param x3: Boolean input x3
    :return: Boolean result of the logical statement
    """
    return (x1 and x2 and not(x3)) or (not(x2) and x3)


def MAT_sign(matrix):
    """
    Function that given a matrix in input generates a new matrices applying the
    sign function for each i-j value. The sign function is defined as
    - sign(x) = 1 for x > 0
    - sign(x) = 0 for x = 0
    - sign(x) = -1 for x < 0

    :param matrix: Input matrix
    :return: sign(x)
    """
    # Initialize new temporary matrix for the result
    result = []

    # Rows
    for row in matrix:
        new_row = []
        # Element of the row
        for element in row:
            # Sign function
            if element < 0:
                new_row.append(-1)
            elif element > 0:
                new_row.append(1)
            else:
                new_row.append(0)
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

    # Eesult matrix full of zeros
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
    :return: None
    """
    return MAT_sign(MAT_sum(MAT_prod(U, MAT_sign(MAT_sum(MAT_prod(W, X), b))), C))


def print_boolean_func_homework():
    """
    Function that prints the boolean result of the logical statement
    :return: None
    """
    # all possible combinations given three inputs
    combinations = [(False, False, False), (False, False, True),
                    (False, True, False), (False, True, True),
                    (True, False, False), (True, False, True),
                    (True, True, False), (True, True, True)]

    # print truth table
    print("x1\t\tx2\t\tx3\t\ty")
    print("-" * 30)

    for x1, x2, x3 in combinations:
        y = boolean_func_homework(x1, x2, x3)
        print(f"{x1}\t{x2}\t{x3}\t{y}")


def print_NN_func_homework(W, b, U, C):
    """
    Function that prints the boolean result of the logical statement computed by the Neural Network
    :param W: first-layer Weight matrix
    :param b: first-layer Bias vector
    :param U: second-layer Weight matrix
    :param C: second-layer Bias vector
    :return: None
    """
    # all possible combinations given three inputs
    combinations = [(-1, -1, -1), (-1, -1, 1),
                    (-1, 1, -1), (-1, 1, 1),
                    (1, -1, -1), (1, -1, 1),
                    (1, 1, -1), (1, 1, 1)]

    # print truth table
    print(f"{'x1':<8}{'x2':<8}{'x3':<8}{'y':<8}")
    print("-" * 32)

    for x1, x2, x3 in combinations:
        y = boolean_func_NN([[x1], [x2], [x3]], W, b, U, C)
        formatted_x1 = f"{' 1' if x1 == 1 else x1:<8}"
        formatted_x2 = f"{' 1' if x2 == 1 else x2:<8}"
        formatted_x3 = f"{' 1' if x3 == 1 else x3:<8}"
        formatted_y = f"{' 1' if y[0][0] == 1 else y[0][0]:<8}"
        print(f"{formatted_x1}{formatted_x2}{formatted_x3}{formatted_y}")




def main():
    # matrices and vectors of weights and biases for the given boolean function
    W = [[1, 1, -1],
         [0, -1, 1]]

    b = [[-2],
         [-1]]

    U = [[1, 1]]

    C = [[1]]

    # print truth table for the given boolean function
    print("Print truth table for the given boolean function: ")
    print_boolean_func_homework()

    # print outcome of the Neural Network
    print("\n\nPrint the outcome of the Neural Network: ")
    print_NN_func_homework(W, b, U, C)

main()
