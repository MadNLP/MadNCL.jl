
function symul!(y::AbstractVector, A::AbstractMatrix, x::AbstractVector, alpha::Number, beta::Number)
    return mul!(y, Symmetric(A, :L), x, alpha, beta)
end

#=
    Kernels for GPU computation
=#

@kernel function _transfer_to_map!(dest, to_map, src)
    k = @index(Global, Linear)
    @inbounds begin
        Atomix.@atomic dest[to_map[k]] += src[k]
    end
end

@kernel function _remove_diagonal_kernel!(y, Ap, Ai, Az, x, alpha)
    j = @index(Global, Linear)
    @inbounds for k in Ap[j]:Ap[j+1]-1
        if Ai[k] == j
            y[j] -= alpha * Az[k] * x[j]
            break
        end
    end
end


