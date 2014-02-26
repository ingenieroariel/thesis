using MAT
using ParallelSparseRegression


function matrixlasso(dictionary, data, lambda=1)
    dimensions, components = size(dictionary)
    dimensions, samples = size(data)

    # [b1 b2]
    b = sparse(reshape(data, dimensions * samples))

    # [D 0]
    # [0 D]
    AA = kron(speye(samples), sparse(dictionary))

    # [c1 c2]
    x = lasso(AA, b, lambda)

    # [C]
    code = reshape(x, components, samples)
    return code
end


file = matopen("admm.mat")
A = read(file,"A")
B_cvx = read(file, "B")
X = read(file, "X")
cost = read(file, "cost")
close(file)

# ||AB - X||_2^2 + ||B||_1
B_julia = matrixlasso(A, X)

print("B_cvx[1, 1:5]","\n")
print(B_cvx[1, 1:5], "\n")
print("B_julia[1, 1:5]","\n")
print(B_julia[1,1:5], "\n")

assert(size(B_julia) == size(B_cvx))
assert(B_julia == B_cvx)
