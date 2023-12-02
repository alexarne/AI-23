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

A = parse(input())
B = parse(input())
pi = parse(input())

def forward(A, B, pi, sequence):
    N = len(A) # num states
    alpha = [[0 for _ in range(N)] for _ in range(len(sequence))] # zeros matrix
    alpha[0] = [pi[0][i] * B[i][sequence[0]] for i in range(N)]
    for t in range(1, len(sequence)):
        for i in range(N):
            alpha[t][i] = sum([alpha[t-1][j] * A[j][i] for j in range(N)]) * B[i][sequence[t]]
    return sum(alpha[-1])

sequence = list(map(int,input().split()))
print(forward(A, B, pi, sequence[1:]))
