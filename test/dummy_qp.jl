
using Random
using NLPModels

struct DenseDummyQP{
    T,
    VT <: AbstractVector{T},
    MT <: AbstractMatrix{T},
    VI <: AbstractVector{Int}
    } <: NLPModels.AbstractNLPModel{T,VT}
    meta::NLPModels.NLPModelMeta{T, VT}
    P::MT # primal hessian
    A::MT # constraint jacobian
    q::VT
    buffer::VT
    hrows::VI
    hcols::VI
    map_hess::VI
    jrows::VI
    jcols::VI
    map_jac::VI
    counters::NLPModels.Counters
end

function NLPModels.jac_structure!(qp::DenseDummyQP, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, qp.jrows)
    copyto!(J, qp.jcols)
    return (I, J)
end
function NLPModels.hess_structure!(qp::DenseDummyQP, I::AbstractVector{T}, J::AbstractVector{T}) where T
    copyto!(I, qp.hrows)
    copyto!(J, qp.hcols)
    return (I, J)
end

function NLPModels.obj(qp::DenseDummyQP{T}, x::AbstractVector{T}) where T
    mul!(qp.buffer, qp.P, x)
    return 0.5 * dot(x, qp.buffer) + dot(qp.q, x)
end

function NLPModels.grad!(qp::DenseDummyQP, x::AbstractVector, g::AbstractVector)
    mul!(g, qp.P, x)
    g .+= qp.q
    return g
end

function NLPModels.cons!(qp::DenseDummyQP, x::AbstractVector, c::AbstractVector)
    mul!(c, qp.A, x)
    return c
end

function NLPModels.jac_coord!(qp::DenseDummyQP, x::AbstractVector, J::AbstractVector)
    J .= qp.A[qp.map_jac]
    return J
end

function NLPModels.jprod!(qp::DenseDummyQP, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
    mul!(jv, qp.A, v)
    return jv
end

function NLPModels.jtprod!(qp::DenseDummyQP, x::AbstractVector, v::AbstractVector, jv::AbstractVector)
    mul!(jv, qp.A', v)
    return jv
end

function NLPModels.hess_coord!(qp::DenseDummyQP{T},x, l, hess::AbstractVector; obj_weight=1.0) where T
    hess .= obj_weight .* qp.P[qp.map_hess]
end

function DenseDummyQP(
    x0::AbstractVector{T} = zeros(100);
    m=10, fixed_variables=similar(x0,Int,0), equality_cons=similar(x0,Int,0)
    ) where {T}

    n = length(x0)

    if m >= n
        error("The number of constraints `m` should be less than the number of variable `n`.")
    end

    Random.seed!(1)
    # Generate random values.
    # N.B.: we need to allocate the matrix P_ right after the vector
    # q_ if we want to obtain a deterministic behavior: the seed is not working
    # if we call the function `randn` after allocating vectors on the device.
    q_ = randn(n)
    P_ = randn(n, n)

    y0 = fill!(similar(x0, m), zero(T))
    q = copyto!(similar(x0, n), q_)
    buffer = similar(x0, n)

    # Bound constraints
    xl = fill!(similar(x0, n), zero(T))
    xu = fill!(similar(x0, n), one(T))
    gl = fill!(similar(x0, m), zero(T))
    gu = fill!(similar(x0, m), one(T))

    # Update gu to load equality constraints
    gu[equality_cons] .= zero(T)
    xl[fixed_variables] .= @view(xu[fixed_variables])

    # Build QP problem 0.5 * x' * P * x + q' * x
    P = copyto!(similar(x0, n , n), P_)
    P = P*P' # P is symmetric
    P += T(100.0) * I

    # Build constraints gl <= Ax <= gu
    A = fill!(similar(x0, m, n), zero(T))
    A[1:m+1:m^2] .= one(T)
    A[m+1:m+1:m^2+m] .=-one(T)

    nnzh = div(n * (n + 1), 2)
    Ih, Jh = [i for i in 1:n for j in 1:i], [j for i in 1:n for j in 1:i]
    hrows = copyto!(similar(x0, Int, nnzh), Ih)
    hcols = copyto!(similar(x0, Int, nnzh), Jh)
    map_hess = [i + n*(j-1) for (i, j) in zip(Ih, Jh)]
    map_hess_device = copyto!(similar(x0, Int, nnzh), map_hess)

    nnzj = n * m
    Ij, Jj = [j for i in 1:n for j in 1:m], [i for i in 1:n for j in 1:m]
    jrows = copyto!(similar(x0, Int, nnzj), Ij)
    jcols = copyto!(similar(x0, Int, nnzj), Jj)
    map_jac = [i + m*(j-1) for (i, j) in zip(Ij, Jj)]
    map_jac_device = copyto!(similar(x0, Int, nnzj), map_jac)

    return DenseDummyQP(
        NLPModels.NLPModelMeta(
            n,
            ncon = m,
            nnzj = nnzj,
            nnzh = nnzh,
            x0 = x0,
            y0 = y0,
            lvar = xl,
            uvar = xu,
            lcon = gl,
            ucon = gu,
            minimize = true
        ),
        P,A,q,buffer,
        hrows,hcols,map_hess_device,
        jrows,jcols,map_jac_device,
        NLPModels.Counters()
    )
end

