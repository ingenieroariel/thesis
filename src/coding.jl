export sparse_code, dict_update!, dictionary_learning

function sparse_code(D, x,lambda1=1, lambda2=1, z=nothing; kwargs ...)
    code = lasso(D, x, lambda1; kwargs...)
end

function dict_update!(D, x, a)
    # Calculate the residuals
    R = -D * a
    R += x
    
    components = length(a)
    
    for k in 1:components
        D[:, k] = R * a[k, :]'

        l2_square = norm(D[:, k], 2)

        if l2_square < 1e-20
            # If the value is too small, let's replace it with random info
            # but set the coef to zero.
            D[:, k] = randn()
            a[k] = 0.0
            l2_square = sqrt(norm(D[:, k], 2))
        end

        # Normalize the dictionary
        D[:, k] /= sqrt(l2_square)

        # R <- 1.0 * U_k * V_k^T + R
        BLAS.ger!(1.0, D[:, k], [a[k]], R[:,:])
    end 
end

function dictionary_learning(X, components; quiet=true)
    # Calculate a dictionary that produces a sparse code
    # for X with the given number of components.
    
    # Find an initial dictionary.
    U, S, V = svd(X);
    
    D_svd = U .* reshape(S, 1, length(S));

    w, h = size(D_svd);

    if components <= h
        D = D_svd[:, 1:components];
    else
        D = [D_svd zeros(w, components - h)];
    end;
    
    # Iterate over the samples
    for i in 1:length(X)
        x = X[:,i]
        # Find a sparse code with a given data 
        a = sparse_code(D, x; quiet=quiet)
        # Perform the dictionary update in place for efficiency
        dict_update!(D, x, a)
    end

    return D
end
