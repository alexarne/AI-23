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

A = parse(input())
B = parse(input())
pi = parse(input())

def forward(A, B, pi, sequence):
    N = A.shape[0] # num states
    alpha = np.matrix(np.zeros((len(sequence), N)))
    alpha[0, 0] = pi * B[: , sequence[0]]

    for t in range(1, len(sequence)):
        for i in range(N):
            alpha[t, i] = alpha[t-1].dot(A[:, i]) * B[i, sequence[t]]

    return alpha[-1, sequence[-1]]

    # p = 1.0
    # N = A.shape[0] # num states
    # alpha = np.matrix(np.zeros((len(sequence), N)))
    # alpha[0, 0] = pi * B[:, sequence[0]]
    # for t in range(1, len(sequence)):
    #     probability = 0.0
    #     for i in range(N):
    #         alpha[t, i] = alpha[t-1].dot(A[:, i]) * B[i, sequence[t]]
    #         probability += alpha[t, i]
    #     p *= probability
    # return alpha[-1].sum()

sequence = list(map(int,input().split()))
print(forward(A, B, pi, sequence[1:]))
