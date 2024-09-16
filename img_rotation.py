def rotate_clockwise_inplace(matrix):
    n = len(matrix)  # Square matrix (n x n)
    
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    
    # Step 2: Reverse each row to rotate 90 degrees clockwise
    for i in range(n):
        matrix[i].reverse()
    
    # counter clockwise: 
    matrix.reverse()

# Example of square matrix (3x3)
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

# Rotate in place
rotate_clockwise_inplace(matrix)

def transpose_matrix(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

# Output the rotated matrix
print("Clockwise Rotated Matrix (In-place):")
for row in matrix:
    print(row)
