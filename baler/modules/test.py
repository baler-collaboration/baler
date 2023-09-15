print("Hello world")
import numpy as np

# A = np.arange(150000).reshape(60, 50, 50)

# total_size = 150000
# block = [5, 5, 5]

# B = A.reshape((total_size // (block[1] * block[2]), block[1], block[2]))
# print(B.shape)

A = np.arange(150000).reshape(60, 50, 50)

total_size = 150000
block = [5, 5, 5]

print("Test1 - ", np.sum(A[0]))

original_shape = [A.shape[0], A.shape[1], A.shape[2]]
B = A.reshape((total_size // (block[1] * block[2]), block[1], block[2]))
print(B.shape)

# Reconstruct
C = B.reshape(original_shape[0], original_shape[1], original_shape[2])
print(C.shape)

print(np.array(A.shape))

print("Test2 - ", np.sum(C[0]))
