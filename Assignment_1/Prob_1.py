'''
    File name: Prob_1.py
    Description: ECE-7650 Deep Learning HW#1 Problem #1
    Author: Junyao Pu
    Date created: Jan 29th, 2020
    Date last modified: Feb 2nd, 2020
    Python Version: 3.6
'''
import numpy as np
#define the function
def tensor_flow(matrix, vector, tensor):
    vector_size = len(vector)
    matrix_size = len(matrix)

    matrix_col = len(vector)
    matrix_row = int(matrix_size/vector_size)

    for i in range(matrix_row):
        tensor[i][(i*vector_size):(i+1)*vector_size] = vector
        tensor[i] = np.reshape(tensor[i], (matrix_row, matrix_col))


#define variable W x and a empty tensor
W = [-1, -2, -1, -2, -1,
    1, 2, 3, 4, 5,
    -1, -2, -1, -2, -1]

x=[1, 2, 3, 4, 5]

x_size = len(x)
W_size = len(W)

tensor=[[0 for i in range(W_size)] for j in range(3)]
#print(tensor)

tensor_flow(W,x,tensor)
print(tensor)
