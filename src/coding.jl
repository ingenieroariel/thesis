export sparse_code, dict_update!, dictionary_learning, online_dictionary_learning

function sparse_code(D, x,lambda1=1, lambda2=1, z=nothing; kwargs ...)

    if length(size(x)) > 1
        coding_function = matrixlasso
    else
        coding_function = lasso
    end

    code = coding_function(D, x, lambda1; kwargs...)

    return sign(code) .* max(abs(code) - 1e-5, 0)
end


function dict_update!(D, x, a)
    # Calculate the residuals
    R = -D * a
    R += x
    
    dimensions, components = size(D)
    
    for k in 1:components
        D[:, k] = R * a[k, :]'

        l2_square = norm(D[:, k], 2)

        if l2_square < 1e-20
            # If the value is too small, let's replace it with random info
            # but set the coef to zero.
            D[:, k] = randn(dimensions)
            a[k, :] = 0.0
            l2_square = sqrt(norm(D[:, k], 2))
        end

        # Normalize the dictionary
        D[:, k] /= sqrt(l2_square)

        # R <- 1.0 * U_k * V_k^T + R
        BLAS.ger!(1.0, D[:, k], vec(a[k, :]), R[:,:])
    end 
end

function init_dictionary(X, components)
    
    # Find an initial dictionary.
    U, S, V = svd(X);
    
    D_svd = U .* reshape(S, 1, length(S));

    w, h = size(D_svd);

    if components <= h
        D = D_svd[:, 1:components];
    else
        D = [D_svd zeros(w, components - h)];
    end;

    return D
end 

function online_dictionary_learning(X, components; quiet=true)
    # Calculate a dictionary that produces a sparse code
    # for X with the given number of components.
    D = init_dictionary(X, components)   

    # Iterate over the samples
    for i in 1:size(X, 2)
        if ~quiet
            print("Iterating over sample #", i, "/",size(X,2), "\n")
        end
        x = X[:,i]
        # Find a sparse code with a given data 
        a = sparse_code(D, x; quiet=quiet, rtol=5e-3, atol=1e-2)
        # Perform the dictionary update in place for efficiency
        dict_update!(D, x, a)
    end

    return D
end

function dictionary_learning(X, components; quiet=true, maxiter=5)
    # Calculate a dictionary that produces a sparse code
    # for X with the given number of components.
    D = init_dictionary(X, components)   
    A = spzeros(components, size(X, 2))

    # Alternate between solving the sparse code and updating the dictionary.
    for i in 1:maxiter
        if ~quiet
           print("Iteration #", i, "/",maxiter, "\n")
        end
        # Find a sparse code with a given data 
        A = sparse_code(D, X, A; quiet=quiet)
        # Perform the dictionary update in place for efficiency
        dict_update!(D, X, A)
    end

    return D

end
