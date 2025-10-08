
using Revise
using Test
using LinearAlgebra
using MadNLP
using MadNCL
using NLPModels
using FiniteDiff

include("hs15.jl")

function test_kkt_system(kkt, cb)
    # Getters
    n = MadNLP.num_variables(kkt)
    (m, p) = size(kkt)
    # system should be square
    @test m == p

    # Interface
    MadNLP.initialize!(kkt)

    # Update internal structure
    x0 = NLPModels.get_x0(cb.nlp)
    y0 = NLPModels.get_y0(cb.nlp)
    # Update Jacobian manually
    jac = MadNLP.get_jacobian(kkt)
    MadNLP._eval_jac_wrapper!(cb, x0, jac)
    MadNLP.compress_jacobian!(kkt)
    # Update Hessian manually
    hess = MadNLP.get_hessian(kkt)
    MadNLP._eval_lag_hess_wrapper!(cb, x0, y0, hess)
    MadNLP.compress_hessian!(kkt)

    # N.B.: set non-trivial dual's bounds to ensure
    # l_lower and u_lower are positive. If not we run into
    # an issue inside SparseUnreducedKKTSystem, which symmetrize
    # the system using the values in l_lower and u_lower.
    fill!(kkt.l_lower, 1e-3)
    fill!(kkt.u_lower, 1e-3)

    # Update diagonal terms manually.
    MadNLP._set_aug_diagonal!(kkt)

    # Factorization
    MadNLP.build_kkt!(kkt)
    MadNLP.factorize!(kkt.linear_solver)

    # Backsolve
    x = MadNLP.UnreducedKKTVector(kkt)
    fill!(MadNLP.full(x), 1.0)  # fill RHS with 1
    out1 = MadNLP.solve!(kkt, x)
    @test out1 === x

    println(x.values)
    y = copy(x)
    fill!(MadNLP.full(y), 0.0)
    out2 = mul!(y, kkt, x)
    @test out2 === y
    @test MadNLP.full(y) ≈ ones(length(x))

    if MadNLP.is_inertia(kkt.linear_solver)
        ni, mi, pi = MadNLP.inertia(kkt.linear_solver)
        @test MadNLP.is_inertia_correct(kkt, ni, mi, pi)
    end

    prim_reg, dual_reg = 1.0, 1.0
    MadNLP.regularize_diagonal!(kkt, prim_reg, dual_reg)

    return
end

nlp = HS15Model()
# Test we recover original solution for ρ ≫ 1
n, m = NLPModels.get_nvar(nlp), NLPModels.get_ncon(nlp)
ncl = MadNCL.RegularizedModel(nlp)

solver = MadNLP.MadNLPSolver(
    ncl;
    linear_solver=LapackCPUSolver,
    kkt_system=MadNCL.K1sAuglagKKTSystem,
    # kkt_system=MadNLP.SparseKKTSystem,
    print_level=MadNLP.DEBUG,
    richardson_max_iter=1,
    # dual_initialized=true,
    max_iter=10,
)
MadNLP.solve!(solver)
# linear_solver = MadNLP.LapackCPUSolver

# ind_cons = MadNLP.get_index_constraints(ncl)

# cb = MadNLP.create_callback(
#     MadNLP.SparseCallback, ncl,
# )

# kkt = MadNLP.create_kkt_system(
#     MadNCL.K1sAuglagKKTSystem,
#     cb,
#     ind_cons,
#     linear_solver;
# )
# test_kkt_system(kkt, cb)

