import numpy as np
 
def parse(line):
    line = line.split()
    rows = int(line[0])
    cols = int(line[1])
    data = list(map(float, line[2:]))
    return np.matrix([data[i*cols : (i+1)*cols] for i in range(rows)])

def print_matrix(matrix):
    dims = matrix.shape
    print(dims[0], dims[1], end=' ')
    [print(matrix[i,j], end=' ') for i in range(dims[0]) for j in range(dims[1])]
    print('\n')

T = parse(input())
E = parse(input())
S = parse(input())

print_matrix(S*T*E)
