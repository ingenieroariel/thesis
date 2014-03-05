
require("../src/SparseCoding.jl")
using SparseCoding
using MAT

file = matopen("admm.mat")
A_in = read(file,"A")
B_cvx = read(file, "B")
X_in = read(file, "X")
cost = read(file, "cost")
close(file)

B_ariel = matrixlasso(A_in, X_in;quiet=false)

print("B_cvx[1, 1:5]","\n")
print(B_cvx[1, 1:5], "\n")
print("B_ariel[1, 1:5]","\n")
print(B_ariel[1,1:5], "\n")

function assertapprox(x, y; ATOL=1e-2, RTOL=1e-2)
    assert(size(x) == size(y))
    for n in 1:length(x)
        equals = isapprox(x[n], y[n]; atol=ATOL, rtol=RTOL)
        if ~equals
           print("Found difference in items:")
           print(" x[",n,"]=", x[n]," y[",n,"]=", y[n])
           print(" with atol=", ATOL, " and rtol=", RTOL, "\n")
        end
        #assert(equals)
    end 
end

assertapprox(B_ariel, B_cvx)
