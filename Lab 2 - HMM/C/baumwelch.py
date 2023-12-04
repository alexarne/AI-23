#!/usr/bin/env python3
import math
import sys
import random
EPSILON = sys.float_info.epsilon

def init_matrix(size_y, size_x):
    matrix = [[1/size_x + random.random()/10 for _ in range(size_x)] for _ in range(size_y)]
    for i in range(size_y):
        rowsum = sum(matrix[i])
        matrix[i] = [v / rowsum for v in matrix[i]]
    return matrix

def uniform_matrix(size_y, size_x):
    matrix = [[1/size_x for _ in range(size_x)] for _ in range(size_y)]
    for i in range(size_y):
        rowsum = sum(matrix[i])
        matrix[i] = [v / rowsum for v in matrix[i]]
    return matrix

def diagonal_matrix(size_y, size_x):
    matrix = [[0.0 for _ in range(size_x)] for _ in range(size_y)]
    for i in range(size_y):
        matrix[i][i] = 1.0
    for i in range(size_y):
        rowsum = sum(matrix[i])
        matrix[i] = [v / rowsum for v in matrix[i]]
    return matrix

# forward algorithm for validating our model (used when answering Q9)
def forward(A, B, pi, sequence):
    N = len(A) # num states
    alpha = [[0 for _ in range(N)] for _ in range(len(sequence))] # zeros matrix
    alpha[0] = [pi[0][i] * B[i][sequence[0]] for i in range(N)]
    for t in range(1, len(sequence)):
        for i in range(N):
            alpha[t][i] = sum([alpha[t-1][j] * A[j][i] for j in range(N)]) * B[i][sequence[t]]
    return sum(alpha[-1])

# backward algorithm - beta-pass algorithm
# A - Transition matrix
# B - Emission matrix
# pi - Initial state vector
# emissions - Sequence of observations
# Returns - New estimated transition matrix, emissions matrix, and initial state vector
PRINT = False
def baumwelch(A, B, pi, emissions):
    N = len(A)          # num states
    T = len(emissions)  # num observations

    oldLogProb = 0
    logProb = 1
    MAX_ITER = 100
    iter = 0
    while iter < MAX_ITER and abs(oldLogProb - logProb) > 1e-4:
        iter = iter + 1
        if PRINT:
            print("Iteration", iter)
        oldLogProb = logProb
        # Compute all (with alpha and beta normalized)
        norm = [1 for _ in range(T)]

        # --------- alpha (normalized) ---------
        alpha = [[0 for _ in range(N)] for _ in range(T)] # zeros matrix
        alpha[0] = [pi[0][i] * B[i][emissions[0]] for i in range(N)]
        norm[0] = sum(alpha[0]) + EPSILON
        alpha[0] = [pi[0][i] * B[i][emissions[0]] / norm[0] for i in range(N)]
        for t in range(1, T):
            for i in range(N):
                alpha[t][i] = sum([alpha[t-1][j] * A[j][i] for j in range(N)]) * B[i][emissions[t]]
            norm[t] = sum(alpha[t]) + EPSILON
            alpha[t] = [alpha[t][i] / norm[t] for i in range(N)]


        # --------- beta (normalized) ---------
        beta = [[0 for _ in range(N)] for _ in range(T)] # zeros mat-rix
        beta[-1] = [1 / norm[-1] for _ in range(N)]
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[t][i] = sum([beta[t+1][j] * B[j][emissions[t+1]] * A[i][j] for j in range(N)])
                beta[t][i] = beta[t][i] / norm[t]

        # --------- di-gamma ---------
        # Doesn't need normalization because alpha & beta are normalized
        # print(alpha[-1])
        dg = [[[alpha[t][i]*A[i][j]*B[j][emissions[t+1]]*beta[t+1][j] 
                for j in range(N)] 
                for i in range(N)] 
                for t in range(T-1)]
        
        # --------- gamma ---------
        g = [[sum([dg[t][i][j] for j in range(N)]) for i in range(N)] for t in range(T-1)]
        
        # Re-estimate A, B, pi
        A = [[sum([dg[t][i][j] for t in range(T-1)]) / (sum([g[t][i] for t in range(T-1)])+EPSILON)
            for j in range(len(A[0]))] 
            for i in range(len(A))]
        B = [[sum([(1 if emissions[t] == k else 0)*g[t][j] for t in range(T-1)]) / (sum([g[t][j] for t in range(T-1)])+EPSILON)
            for k in range(len(B[0]))] 
            for j in range(len(B))]
        pi = [g[0]]

        # Repeat until convergence
        logProb = sum([math.log(norm[i]) for i in range(len(norm))])
        # print("logProb", logProb, "oldLogProb", oldLogProb)
        if PRINT:
            print("logprob diff", abs(logProb-oldLogProb))
        # print("A:", A)
        # print("B:", B)
        # print("pi:", pi)

    return A, B, pi


emissions = list(map(int,input().split()))

# # Q7 initializations
# A = [[0.54, 0.26, 0.20],
#      [0.19, 0.53, 0.28],
#      [0.22, 0.18, 0.60]]
# B = [[0.50, 0.20, 0.11, 0.19],
#      [0.22, 0.28, 0.23, 0.27],
#      [0.19, 0.21, 0.15, 0.45]]
# pi = [[0.3, 0.2, 0.5]]

# # Q8 random initializations
# A = init_matrix(3, 3)
# B = init_matrix(3, 4)
# pi = init_matrix(1, 3)

# # Q8 manual initializations
# A = [[0.80, 0.05, 0.15],
#      [0.13, 0.73, 0.04],
#      [0.23, 0.25, 0.52]]
# B = [[0.69, 0.23, 0.08, 0.00],
#      [0.08, 0.42, 0.27, 0.23],
#      [0.03, 0.07, 0.18, 0.72]]
# pi = [[0.7, 0.15, 0.15]]

# Q9
for N in range(1,5):
    A = init_matrix(N, N)
    B = init_matrix(N, 4)
    pi = init_matrix(1, N)
    k = 5
    breakpoint = int(len(emissions)/k)
    for i in range(k):
        A1 = A
        B1 = B
        pi1 = pi
        training = (emissions[1+breakpoint*i:breakpoint*(i+1)+1])
        validation = (emissions[1:1+breakpoint*i] + emissions[breakpoint*(i+1)+1:])
        print("length of training set:",len(training))
        print("length of validation set:",len(validation))
        A2, B2, pi2 = baumwelch(A1, B1, pi1, training)
        alpha = forward(A2, B2, pi2, validation)
        print("N =",N,"=> alpha =",alpha)

# # Q10
# Uniform matrices: 
# A = uniform_matrix(3, 3)
# B = uniform_matrix(3, 4)
# # pi = uniform_matrix(1, 3)
# pi = [[0.5, 0.1, 0.4]]

# # Diagonal A and specific start state:
# A = diagonal_matrix(3, 3)
# print(A)
# B = init_matrix(3, 4)
# pi = [[0.0, 0.0, 1.0]]

# # Close to solution:
# A = [[0.72, 0.04, 0.24],
#      [0.13, 0.78, 0.09],
#      [0.21, 0.28, 0.51]]
# B = [[0.69, 0.22, 0.08, 0.01],
#      [0.08, 0.42, 0.27, 0.23],
#      [0.03, 0.07, 0.18, 0.72]]
# pi = [[0.9, 0.05, 0.05]]

A2, B2, pi2 = baumwelch(A, B, pi, emissions[1:])
print(A2)
print(B2)
print(pi2)