import numpy as np

matrix1 = np.array([[1, 3], 
             		[5, 7]])
             
matrix2 = np.array([[2, 6], 
                    [4, 8]])

#Matrix addition
result = np.add(matrix1, matrix2)
print("matrix1 + matrix2: \n",result)

#Matrix subtraction
result = np.subtract(matrix1, matrix2)
print("matrix1 - matrix2: \n",result)

#Matrix multiplication
result = np.dot(matrix1, matrix2)
print("matrix1 x matrix2: \n",result)

#Matrix transposition
result = np.transpose(matrix1)
print("transpose matrix1: \n",result)

#Matrix inversion
result = np.linalg.inv(matrix1)
print("inverse matrix1: \n",result)

#Matrix pseudo-inversion
result = np.linalg.pinv(matrix1)
print("pseudo-inverse matrix1: \n",result)

#Matrix power
result = np.power(matrix1, 2)
print("matrix1 power 2: \n",result)

#Matrix vector-matrix multiplication
vector = np.array([10, 20]) 
result = np.matmul(matrix1, vector)
print("matrix1 and vector multiplication: \n",result)

#Matrix square root
result = np.sqrt(matrix1)
print("matrix1 square root: \n",result)

#Sumber:
#https://www.programiz.com/python-programming/numpy/matrix-operations
#https://numpy.org/doc/2.1/reference/generated/numpy.linalg.pinv.html
#https://www.geeksforgeeks.org/parallel-matrix-vector-multiplication-in-numpy/