using MAT
using IterativeSolvers

file = matopen("admm.mat")
A_in = read(file,"A")
B_cvx = read(file, "B")
X_in = read(file, "X")
cost = read(file, "cost")
close(file)

dimensions, samples = size(X_in)
dimensions, components = size(A_in)

X = reshape(X_in, dimensions * samples)
A = kron(eye(samples), A_in)

samdim = size(X, 1)
samcom = size(A, 2)

rho = 1
maxIter = 4000
I = eye(samcom)
lambda = 1
quiet = false

ABSTOL, RELTOL = 5e-6, 1e-5

# Initialize B and it's dual variable C with zeroes.
B = C = zeros(samcom)

L = zeros(size(X))
sthresh(x, th) = sign(x) .* max(abs(x)-th,0)

cost = zeros(maxIter)

F = A'*A+rho*I

if ~quiet
    @printf("iter :\t%8s\t%8s\t%8s\t%8s\n", "r", "eps_pri", "s", "eps_dual")
end

AX = vec(A'*X)

tic()
for iter in 1:maxIter

    B_prev = copy(B)

    # Solve sub-problem to solve B
    lsqr!(B, F, AX + rho*C - L; maxiter=5)

    # Solve sub-problem to solve C
    C = sthresh(B + L / rho, lambda / rho);

    # Update the Lagrangian
    L = L + rho*(B - C);

    # get the current cost
    #cost[iter] = 0.5*norm(X - A*B, 2) + lambda*norm(B, 1);
    sqrtn = sqrt(samcom)

    eps_pri  = sqrtn*ABSTOL + RELTOL*max(norm(B), norm(C))
    eps_dual = sqrtn*ABSTOL + RELTOL*norm(B)

    #FIXME(Ariel): Fix calculation of prires and duares
    prires = norm(B - C)
    duares = rho*norm(B - B_prev)

    if ~quiet && (iter == 1 || mod(iter,10) == 0)
         @printf("%4d :\t%.2e\t%.2e\t%.2e\t%.2e\n", iter, prires, eps_pri, duares, eps_dual);
    end

    if iter > 2 && prires < eps_pri && duares < eps_dual
        if ~quiet
            @printf("total iterations: %d\n", iter);
            toc()
        end
        break;
    end
end

B_ariel = reshape(sthresh(B, 1e-10), components, samples)

print("B_cvx[1, 1:5]","\n")
print(B_cvx[1, 1:5], "\n")
print("B_ariel[1, 1:5]","\n")
print(B_ariel[1,1:5], "\n")

print(size(sparse(B_ariel)))

assert(size(B_ariel) == size(B_cvx))
assert(B_ariel == B_cvx)
