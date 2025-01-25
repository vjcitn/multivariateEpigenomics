using Random

#There are packages in julia (e.g., NMF.jl) that are more comprehensive, but they do not allow for GPU acceleration. Here I'm using the fast HALS algorithm and writing it to be GPU compatible, greatly speeding up the large number of matrix multiplications required for NMF.

#Paper Reference for fast HALS: Cichocki, Andrzej, and P. H. A. N. Anh-Huy. Fast local algorithms for large scale nonnegative matrix and tensor factorizations. IEICE transactions on fundamentals of electronics, communications and computer sciences 92.3: 708-721 (2009).

#Flushes out messages on cluster for printing
function message(x...)
    println(now(), " ", join(x, " ")...)
    flush(stdout)
end

#Not needed for NMF, but a useful function to check matrix sparsity
sparsity(M::AbstractArray{T}) where T = count(M .≈ zero(T)) / length(M)

# X ≈ WHᵀ
#minimize ½||Xʲ - wⱼhⱼᵀ||² + α||hⱼ||₁
#α is the sparsity constraint on H
#j is the column index
mutable struct NMFCache{M,V,T}
    #Input
    X::M

    #Factorization
    W::M
    H::M

    #Estimate
    WHᵀ::M

    #Temporary variables for updating H
    XᵀW::M 
    WᵀW::M
    HWᵀWⱼ::V

    #Temporary variables for updating W
    XH::M
    HᵀH::M
    WHᵀHⱼ::V

    #Regularization Variables (sparsity)
    α::T
end

#########################################
# Cache Initialization
#########################################

#k is the number of factors to reduce to

#CPU or GPU compatible
function NMFCache(X::AbstractMatrix{T}, k; α=zero(T)) where T

    n,m = size(X)

    #Initialize with random matrices
    #The "similar |> rand!" paradigm is general enough to handle matrices on CPU or GPU
    #Can be read as "create a matrix similar to X with the given dimensions and randomize its values"
    W = similar(X,n,k) |> rand!
    H = similar(X,m,k) |> rand!

    #L2 norm of each column needs to be one
    foreach(normalize!,eachcol(W))
    foreach(normalize!,eachcol(H))

    WHᵀ = W*H'

    XᵀW = X'W
    WᵀW = W'W
    HWᵀWⱼ = similar(X,m) |> rand!

    XH = X*H
    HᵀH = H'H
    WHᵀHⱼ = similar(X,n) |> rand!

    return NMFCache(X, W, H, WHᵀ, XᵀW, WᵀW, HWᵀWⱼ, XH, HᵀH, WHᵀHⱼ, α)
end


#########################################
# Main update function
#########################################

function solveNMF(nmf::NMFCache; verbose=false, maxiter=200)
    

    for step in 1:maxiter
        
        verbose ? message("Updating H...") : nothing
        updateH!(nmf)
        verbose ? message("Updating W...") : nothing
        updateW!(nmf)


        if verbose
            message("Iterations: $step \n error: $(residual(nmf))  \n __________________________________________________________________")
        end
    end

    zeroout!(nmf.H)
    zeroout!(nmf.W)

    mul!(nmf.WHᵀ, nmf.W, nmf.H')

    return nmf
end

#Calculate the difference between X and WHᵀ
#Normalize by ||X||
function residual(nmf)
    mul!(nmf.WHᵀ, nmf.W, nmf.H')
    return norm(nmf.WHᵀ - nmf.X) / norm(nmf.X)
end

#Convert all values less than machine percision to zero
function zeroout!(M::AbstractArray{T}) where T

    M[M .≤ eps(T)] .= zero(T)

    return nothing
end

#########################################
# Update H
#########################################

function updateH!(nmf)
    #Unpack the cached matrices/vectors
    X = nmf.X
    W = nmf.W
    H = nmf.H
    XᵀW = nmf.XᵀW
    WᵀW = nmf.WᵀW
    HWᵀWⱼ = nmf.HWᵀWⱼ
    α = nmf.α

    #Get the smallest positive value for our matrix type
    ϵ = eps(eltype(X))

    #These can be updated out of the loop
    mul!(nmf.XᵀW, X', W)
    mul!(nmf.WᵀW, W', W)


    # hⱼ ← max(ϵ, hⱼ + [XᵀW]ⱼ - H[WᵀW]ⱼ - α⋅1ₖ)
    @views begin
        for j in axes(H,2)
            #Calculate the jth vector of H*WᵀW
            mul!(HWᵀWⱼ, H, WᵀW[:,j])

            #Calculate the update
            @. H[:,j] = max(ϵ, H[:,j] + XᵀW[:,j] - HWᵀWⱼ - α)
        end
    end   
end

#########################################
# Update W
#########################################

function updateW!(nmf)
    #Unpack the cached matrices/vectors
    X = nmf.X
    W = nmf.W
    H = nmf.H
    XH = nmf.XH
    HᵀH = nmf.HᵀH
    WHᵀHⱼ = nmf.WHᵀHⱼ

    #These can be updated out of the loop
    mul!(XH, X, H)
    mul!(HᵀH, H', H)

    #Get the smallest positive value for our matrix type
    ϵ = eps(eltype(X))

    @views begin
        for j in axes(W,2)
            mul!(WHᵀHⱼ, W, HᵀH[:,j])

            #Option 1 (only good for CPU)
            # @. W[:,j] = max(ϵ, W[:,j] * HᵀH[j,j] + XH[:,j] - WHᵀHⱼ)
            
            #Option 2 (re-calculates HᵀH[j,j] but avoids scalar indexing on GPU)
            HᵀHⱼⱼ = sum(abs2, H[:,j]) # == HᵀH[j,j]
            @. W[:,j] = max(ϵ, W[:,j]*HᵀHⱼⱼ + XH[:,j] - WHᵀHⱼ)


            normalize!(W[:,j])
        end
    end
end
