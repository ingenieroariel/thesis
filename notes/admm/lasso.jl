using IterativeSolvers

function lasso(A_in, X_in, lambda=1; rho=1, quiet=true, maxiter=5000, ABSTOL=1e-6, RELTOL=1e-5)
    dimensions, samples = size(X_in)
    dimensions, components = size(A_in)

    X = reshape(X_in, dimensions * samples)
    A = kron(eye(samples), A_in)

    samdim = size(X, 1)
    samcom = size(A, 2)

    I = eye(samcom)

    # Initialize B and it's dual variable C with zeroes.
    B = C = zeros(samcom)

    L = zeros(size(X))
    sthresh(x, th) = sign(x) .* max(abs(x)-th,0)

    F = A'*A+rho*I

    if ~quiet
        @printf("iter :\t%8s\t%8s\t%8s\t%8s\n", "r", "eps_pri", "s", "eps_dual")
    end

    AX = vec(A'*X)

    tic()
    for iter in 1:maxiter

        B_prev = copy(B)

        # Solve sub-problem to solve B
        lsqr!(B, F, AX + rho*C - L; maxiter=5)

        # Solve sub-problem to solve C
        C = sthresh(B + L / rho, lambda / rho);

        # Update the Lagrangian
        L = L + rho*(B - C);

        # get the current cost
        sqrtn = sqrt(samcom)

        #FIXME(Ariel): Are these two the right eps?
        eps_pri  = sqrtn*ABSTOL + RELTOL*max(norm(B), norm(C))
        eps_dual = sqrtn*ABSTOL + RELTOL*norm(B)

        #FIXME(Ariel): Fix calculation of prires and duares.
        # The values below are incorrect placeholders that work by chance.
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

    return reshape(sthresh(B, ABSTOL), components, samples)

end


using MAT

file = matopen("admm.mat")
A_in = read(file,"A")
B_cvx = read(file, "B")
X_in = read(file, "X")
cost = read(file, "cost")
close(file)

B_ariel = lasso(A_in, X_in;quiet=false)

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
