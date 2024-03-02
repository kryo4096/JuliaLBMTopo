using ParallelStencil


module LBM

const c = [0 0; 1 0; 0 1; -1 0; 0 -1; 1 1; -1 1; -1 -1; 1 -1]
const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]


## PARAMETERS
const L_x = 1
const L_y = 1

const resolution = 100

const nu = 0.01
const k = 0.01



## Dependent Parameters

const n_x = L_x * resolution
const n_y = L_y * resolution

const dx = 1 / resolution

const tau_f = 3 * nu / dx + 0.5
const tau_g = 3 * k / dx + 0.5

# Equation 16
function f_equilibrium(rho, u, v)
    f_eq = zeros(Float64, 9)

    for i in 1:9
        cu = c[i, 1] * u + c[i, 2] * v
        f_eq[i] = w[i] * rho * (1 + 3 * cu + 9 / 2 * cu^2 - 3 / 2 * (u^2 + v^2))
    end

    return f_eq
end

# Equation 17
function g_equilibrium(T, u, v)
	g_eq = zeros(Float64, 9)

	for i in 1:9
		cu = c[i, 1] * u + c[i, 2] * v
		g_eq[i] = w[i] * T * (1 + 3 * cu)
	end

	return g_eq
end

function f_relax!(f, tau_f, rho, u, v)
    f .-= 1/tau_f * f_equilibrium(rho, u, v)
    return f
end

function compute_moments!(f, g, rho, u, v, T)
    for ix in 1:n_x
        for iy in 1:n_y
            rho[ix, iy] = 0
            u[ix, iy] = 0
            v[ix, iy] = 0
            T[ix, iy] = 0
            for i in 1:9
                rho[ix, iy] += f[i, ix, iy]
                u[ix, iy] += c[i, 1] * f[i, ix, iy]
                v[ix, iy] += c[i, 2] * f[i, ix, iy]
                T[ix, iy] += g[i, ix, iy]
            end
        end
    end
end

function stream!(pop_old, pop_new)
    for ix in 1:n_x
        for iy in 1:n_y
            for i in 1:9
                ix_new = mod1(ix + c[i, 1], n_x)
                iy_new = mod1(iy + c[i, 2], n_y)

                pop_new[i, ix, iy] = pop_old[i, ix_new, iy_new]
            end
        end
    end
end

function slip_condition!(pop)

end



