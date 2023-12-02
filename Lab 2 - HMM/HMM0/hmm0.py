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

T = parse(input())
E = parse(input())
S = parse(input())

print_matrix(matmul(S, matmul(T, E)))
