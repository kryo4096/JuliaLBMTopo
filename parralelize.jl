# this is a parralel implementation of ibm.jl
const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(AMDGPU, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end


const c = [0 0; 1 0; 0 1; -1 0; 0 -1; 1 1; -1 1; -1 -1; 1 -1]
const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]

# @parallel_indices (ix, iy) function compute_flux!(qDx, qDy, Pf, k_ηf_dx, k_ηf_dy, _1_θ_dτ)
#     nx, ny = size(Pf)
#     if (ix <= nx - 1 && iy <= ny) qDx[ix+1, iy] -= (qDx[ix+1, iy] + k_ηf_dx * @d_xa(Pf)) * _1_θ_dτ end
#     if (ix <= nx && iy <= ny - 1) qDy[ix, iy+1] -= (qDy[ix, iy+1] + k_ηf_dy * @d_ya(Pf)) * _1_θ_dτ end
#     return nothing
# end


# Equation 16
@parallel_indices (i,j) function f_equilibrium!(rho, u, v)
    f_eq = @zeros(9)

    for i in 1:9
        cu = c[i, 1] * u + c[i, 2] * v
        f_eq[i] = w[i] * rho * (1 + 3 * cu + 9 / 2 * cu^2 - 3 / 2 * (u^2 + v^2))
    end

    return nothing
end

rho = @zeros(100, 100)
u = @zeros(100, 100)
v = @zeros(100, 100)

f_equilibrium!(rho, u, v)

