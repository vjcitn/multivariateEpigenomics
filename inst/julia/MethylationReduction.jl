module MethylationReduction

using LinearAlgebra, SparseArrays, Random  # Base Packages
using CUDA                                 # Allow calculations on GPU
using Dates                                # Get the current time (for logging) 


include("custom_nmf.jl")

export 
    message,
    sparsity,
    residual,
    NMFCache,
    solveNMF

end
