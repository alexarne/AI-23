#!/usr/bin/env python3

def parse(line):
    line = line.split()
    rows = int(line[0])
    cols = int(line[1])
    data = list(map(float, line[2:]))
    return [data[i*cols : (i+1)*cols] for i in range(rows)]

def matmul(A, B):
    return [[sum(i * j for i, j in zip(r, c)) for c in zip(*B)] for r in A]

def print_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    print(rows, cols, end=' ')
    [print(matrix[i][j], end=' ') for i in range(rows) for j in range(cols)]
    print()

# forward algorithm - alpha-pass algorithm
# A - Transition matrix
# B - Emission matrix
# pi - Initial state vector
# emissions - Sequence of observations
# Returns - Probability of observing the observation sequence
def forward(A, B, pi, emissions):
    N = len(A) # num states
    alpha = [[0 for _ in range(N)] for _ in range(len(emissions))] # zeros matrix
    alpha[0] = [pi[0][i] * B[i][emissions[0]] for i in range(N)]
    for t in range(1, len(emissions)):
        for i in range(N):
            alpha[t][i] = sum([alpha[t-1][j] * A[j][i] for j in range(N)]) * B[i][emissions[t]]
    return sum(alpha[-1])

# Viterbi algorithm
# A - Transition matrix
# B - Emission matrix
# pi - Initial state vector
# emissions - Sequence of observations
# Returns - Most likely sequence of states given the observations
def viterbi(A, B, pi, emissions):
    N = len(A)
    M = len(emissions)
    delta = [[0 for _ in range(N)] for _ in range(M)]
    delta[0] = [pi[0][i] * B[i][emissions[0]] for i in range(N)]
    idx = [[0 for _ in range(N)] for _ in range(M)]
    for t in range(1, M):
        for i in range(N):
            alternatives = [delta[t-1][j] * A[j][i] * B[i][emissions[t]] for j in range(N)]
            maxima = max(enumerate(alternatives), key=lambda x: x[1])
            delta[t][i] = maxima[1]
            idx[t][i] = maxima[0]
    path = [0 for _ in range(M)]
    path[-1] = max(enumerate(delta[-1]), key=lambda x: x[1])[0]
    for t in range(M-2, -1, -1):
        prev_idx = path[t+1]
        path[t] = idx[t+1][prev_idx]
    return path

# backward algorithm - beta-pass algorithm
# A - Transition matrix
# B - Emission matrix
# pi - Initial state vector
# emissions - Sequence of observations
# Returns - Estimated transition matrix and emissions matrix
def backward(A, B, pi, emissions):
    N = len(A)          # num states
    T = len(emissions)  # num observations
    print(A)
    print(B)
    print(pi)

    # Compute all

    # --------- alpha ---------
    alpha = [[0 for _ in range(N)] for _ in range(T)] # zeros matrix
    alpha[0] = [pi[0][i] * B[i][emissions[0]] for i in range(N)]
    for t in range(1, T):
        for i in range(N):
            alpha[t][i] = sum([alpha[t-1][j] * A[j][i] for j in range(N)]) * B[i][emissions[t]]
    
    # --------- beta ---------
    beta = [[0 for _ in range(N)] for _ in range(T)] # zeros matrix
    beta[-1] = [1 for i in range(N)]
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t][i] = sum([beta[t+1][j] * B[j][emissions[t+1]] * A[i][j] for j in range(N)])

    # --------- di-gamma ---------
    print(alpha[0])
    print(alpha[1])
    print(alpha[2])
    print(alpha[3])
    asum = sum([alpha[-1][k] for k in range(N)])
    dg = [[[alpha[t][i]*A[i][j]*B[j][emissions[t+1]]*beta[t+1][j] / asum 
            for j in range(N)] 
            for i in range(N)] 
            for t in range(T-1)]
    
    # --------- gamma ---------
    g = [[sum([dg[t][i][j] for j in range(N)]) for i in range(N)] for t in range(T-1)]
    
    # Re-estimate A, B, pi
    A2 = [[sum([dg[t][i][j] for t in range(T-1)]) / sum([g[t][i] for t in range(T-1)]) 
           for j in range(len(A[0]))] 
           for i in range(len(A))]
    B2 = [[sum([(1 if emissions[t] == k else 0)*g[t][j] for t in range(T-1)]) / sum([g[t][j] for t in range(T-1)]) 
           for k in range(len(B[0]))] 
           for j in range(len(B))]
    pi2 = [[g[0][i] for i in range(N)]]

    print(A2)
    print(B2)
    print(pi2)

    # Repeat until convergence

    return [[1, 2, 3], [2, 3, 4]], [[1, 2, 3], [2, 3, 4]]



A = parse(input())
B = parse(input())
pi = parse(input())

emissions = list(map(int,input().split()))
newA, newB = backward(A, B, pi, emissions[1:])
print_matrix(newA)
print_matrix(newB)
