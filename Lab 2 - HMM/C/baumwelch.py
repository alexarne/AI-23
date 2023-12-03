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

# backward algorithm - beta-pass algorithm
# A - Transition matrix
# B - Emission matrix
# pi - Initial state vector
# emissions - Sequence of observations
# Returns - New estimated transition matrix, emissions matrix, and initial state vector
def baumwelch(A, B, pi, emissions):
    N = len(A)          # num states
    T = len(emissions)  # num observations

    oldLogProb = 0
    logProb = 1
    MAX_ITER = 1000
    iter = 0
    while iter < MAX_ITER and abs(oldLogProb - logProb) > 1e-5:
        iter = iter + 1
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
        beta = [[0 for _ in range(N)] for _ in range(T)] # zeros matrix
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
        print("logprob diff", abs(logProb-oldLogProb))

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

# Q8 manual initializations
A = [[0.80, 0.05, 0.15],
     [0.13, 0.73, 0.04],
     [0.23, 0.25, 0.52]]
B = [[0.69, 0.23, 0.08, 0.00],
     [0.08, 0.42, 0.27, 0.23],
     [0.03, 0.07, 0.18, 0.72]]
pi = [[0.7, 0.15, 0.15]]

A2, B2, pi2 = baumwelch(A, B, pi, emissions[1:])
print(A2)
print(B2)
print(pi2)