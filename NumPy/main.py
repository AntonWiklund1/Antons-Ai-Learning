import numpy as np

# Creating a 2D array representing a 3x3 matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


print("Matrix:\n", matrix)
# Accessing elements
print("Element at row 1, column 2:", matrix[0, 1])  # Access element (Remember: indexing starts at 0)

# Slicing
print("First row:", matrix[0, :])  # Get the first row
print("First column:", matrix[:, 0])  # Get the first column

# Operations
transposed_matrix = matrix.T  # Transpose the matrix
print("Transposed matrix:\n", transposed_matrix)
