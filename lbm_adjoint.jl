module ALBM

using Plots
using CoherentNoise
using .Threads

const c = [  0   0; 
             1   0;
             0   1;
            -1   0; 
             0  -1; 
             1   1; 
            -1   1; 
            -1  -1; 
             1  -1]

const w = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]

const resolution = 128
const dx = 1 / resolution
const dt = dx

const output_interval = 32


# Equation 32
const alpha_min = 0.0
const alpha_max = 2e2
const ramp_alpha = 0.1

const beta_min = 0.0
const beta_max = 0.1
const ramp_beta = 0.1

const t_end = 100.0
const n_t = Int(t_end / dt)

# Equation 68, 69
const epsilon_u = 1e-4
const epsilon_q = 1e-4

const L_x = 1
const L_y = 2

const n_x = L_x * resolution
const n_y = L_y * resolution

const u_0 = 0.00
const v_0 = 0.1

# Page 369
const rho_0 = 1.0

const g_x = 0.0
const g_y = 0.0

const tau_f = 0.8

const Pr = 6
const nu = (1.0 / 3.0) * (tau_f - 0.5) * dx
const alpha_t =  nu / Pr

const inlet_width = 0.33

const T_0 = 0.0
const T_in = 0.0

function pow(x,y)
    return x^y
end

# Equation 5
function f_equilibrium!(f_eq, rho, u, v)
    for i in 1:9
        cu = c[i, 1] * u + c[i, 2] * v
        f_eq[i] = w[i] * rho * (1 + 3 * cu + 9 / 2 * cu^2 - 3 / 2 * (u^2 + v^2))
    end
end

# Equation 6
function g_equilibrium!(g_eq, T, u, v)
	for i in 1:9
		cu = c[i, 1] * u + c[i, 2] * v
		g_eq[i] = w[i] * T * (1 + 3 * cu)
	end
end

# Equation 22, 23
function init_pops!(f, g, rho, u, v, T)
    f_eqs = [zeros(Float32, 9) for i in 1:Threads.nthreads()]
    g_eqs = [zeros(Float32, 9) for i in 1:Threads.nthreads()]

    Threads.@threads for ix in 1:n_x
        for iy in 1:n_y
            f_equilibrium!(f_eqs[Threads.threadid()], rho[ix, iy], u[ix, iy], v[ix, iy]) # 99% sure this works
            g_equilibrium!(g_eqs[Threads.threadid()], T[ix, iy], u[ix, iy], v[ix, iy])

            f[:,ix,iy] .= f_eqs[Threads.threadid()]
            g[:,ix,iy] .= g_eqs[Threads.threadid()]
        end
    end
end

# Equation 3, 4
function relax!(f, g, p, rho, u, v, T, tau_f, tau_g)
    f_eq_thread_local = [zeros(Float32, 9) for i in 1:Threads.nthreads()]
    g_eq_thread_local = [zeros(Float32, 9) for i in 1:Threads.nthreads()]

 

    Threads.@threads for ix in 1:n_x
        for iy in 1:n_y

            ti = Threads.threadid()
            
            f_equilibrium!(f_eq_thread_local[ti], rho[ix,iy], u[ix,iy], v[ix,iy])
            g_equilibrium!(g_eq_thread_local[ti], T[ix,iy], u[ix,iy], v[ix,iy])

            for i in 1:9
                f[i,ix,iy] = f[i,ix,iy] - 1/tau_f * (f[i,ix,iy] - f_eq_thread_local[ti][i])
                g[i,ix,iy] = g[i,ix,iy] - 1/tau_g * (g[i,ix,iy] - g_eq_thread_local[ti][i])
            end
        end
    end
end

# Equation 20, 21
function force!(f, g, F, Q_t)
    Threads.@threads for ix in 1:n_x
        for iy in 1:n_y
            for i in 1:9
                f[i,ix,iy] = f[i,ix,iy] + 3.0 * dx*w[i]*(c[i,1]*F[1,ix,iy] + c[i,2]*F[2,ix,iy])
                g[i,ix,iy] = g[i,ix,iy] + dx * w[i]*Q_t[ix,iy] 
            end
        end
    end
end

# equation 8, 9, 10, 11, 12
function compute_moments!(rho, u, v, p, q_t, T, f, g)
    Threads.@threads for ix in 1:n_x
        for iy in 1:n_y
            rho[ix, iy] = 0
            u[ix, iy] = 0
            v[ix, iy] = 0
            p[ix, iy] = 0
            q_t[:, ix, iy] .= 0
            T[ix, iy] = 0

            for i in 1:9
                rho[ix, iy] += f[i, ix, iy]
                u[ix, iy] += c[i, 1] * f[i, ix, iy]
                v[ix, iy] += c[i, 2] * f[i, ix, iy]
                
                T[ix, iy] += g[i, ix, iy]
            end
            
            # Equation 11
            for i in 1:9
                q_t[1, ix, iy] = c[i, 1] * g[i, ix, iy] - T[ix, iy] * u[ix, iy]
                q_t[2, ix, iy] = c[i, 2] * g[i, ix, iy] - T[ix, iy] * v[ix, iy]
            end
            p[ix,iy] = rho[ix,iy] / 3.0
            u[ix, iy] /= rho[ix, iy]
            v[ix, iy] /= rho[ix, iy] 
        end
    end
end

# Equation 32
function compute_alpha_gamma(gamma)
    return @. alpha_max + (alpha_min - alpha_max) * ( gamma * (1 + ramp_alpha)) / (gamma + ramp_alpha)
end

# Equation 36
function compute_beta_gamma(gamma)
    return @. beta_max + (beta_min - beta_max) * ( gamma * (1 + ramp_beta)) / (gamma + ramp_beta)
end


# Equation 16
function compute_tau_f()
    return 3 * nu / dx + 0.5
end


function compute_K_gamma(gamma)
    return @. K_f + (K_s - K_f) * ramp_k  * ( gamma * (1 + ramp_k)) / (gamma + ramp_k)
end

function compute_tau_g()
    # Equation 17
    return 3 * alpha_t / dx + 0.5
    # return @. 3 * K_gamma / dx + 0.5
end

# Equation 31
function compute_F!(F, u, v, alpha_gamma)
    F[1, :, :] .= - alpha_gamma .* u .+ g_x
    F[2, :, :] .= - alpha_gamma .* v .+ g_y
end

# Equation 35
function compute_Q_t!(Q_t, T, beta_gamma)
    Q_t .= beta_gamma .* (1 .- T)
end

function advect_f!(f)
    f_new = zero(f) 

    Threads.@threads for ix in 1:n_x
        for iy in 1:n_y
            for i in 1:9
                ix_new = ix - c[i, 1]
                iy_new = iy - c[i, 2]

                if ix_new < 1 || ix_new > n_x || iy_new < 1 || iy_new > n_y
                    f_new[i, ix, iy] = 0.0
                else
                    f_new[i, ix, iy] = f[i, ix_new, iy_new]
                end
            end
        end
    end

    ## Boundary Conditions

    x_0 = Int(floor((0.5 - 0.5 * inlet_width) * n_x))
    x_1 = Int(floor((0.5 + 0.5 * inlet_width) * n_x))

    # Bounce Back Left 
    ix = 1
    Threads.@threads for iy in 1:n_y
        f_new[6,ix,iy] = f[8,ix,iy]
        f_new[2,ix,iy] = f[4,ix,iy]
        f_new[9,ix,iy] = f[7,ix,iy]
    end

    # Bounce Back Right
    ix = n_x
    Threads.@threads for iy in 1:n_y
        f_new[8,ix,iy] = f[6,ix,iy]
        f_new[4,ix,iy] = f[2,ix,iy]
        f_new[7,ix,iy] = f[9,ix,iy]
    end

    # Bounce Back Bottom
    iy = 1
    Threads.@threads for ix in 1:n_x
        f_new[3,ix,iy] = f[5,ix,iy]
        f_new[6,ix,iy] = f[8,ix,iy]
        f_new[7,ix,iy] = f[9,ix,iy]
    end

    # Bounce Back Top
    iy = n_y
    Threads.@threads for ix in 1:n_x
        f_new[5,ix,iy] = f[3,ix,iy]
        f_new[8,ix,iy] = f[6,ix,iy]
        f_new[9,ix,iy] = f[7,ix,iy]
    end
    
    # Inlet 
    iy = 1
    Threads.@threads for ix in x_0:x_1
        _f = f_new[:, ix, iy]
        rho = (_f[1] + _f[2] + _f[4] + 2 * (_f[5] + _f[8] + _f[9])) / (1 - v_0)
        f_new[3, ix, iy] = _f[5] + (2. / 3.) * rho * v_0 
        f_new[6, ix, iy] = _f[8] + (1. / 6.) * rho * v_0 - 0.5 * (_f[2] - _f[4])
        f_new[7, ix, iy] = _f[9] + (1. / 6.) * rho * v_0 + 0.5 * (_f[2] - _f[4])
    end

    # Outlet
    iy = n_y
    Threads.@threads for ix in x_0:x_1
        _f = f_new[:, ix, iy]
        v = (1 - (_f[1] + _f[2] + _f[4] + 2 * (_f[3] + _f[7] + _f[6]))) / (rho_0)
        f_new[5, ix, iy] = _f[3] + (2. / 3.) * rho_0 * v 
        f_new[8, ix, iy] = _f[6] + (1. / 6.) * rho_0 * v + 0.5 * (_f[2] - _f[4])
        f_new[9, ix, iy] = _f[7] + (1. / 6.) * rho_0 * v - 0.5 * (_f[2] - _f[4])
    end

    
    f .= f_new
end


function advect_g!(g, f)
    g_new = zero(g) 

    Threads.@threads for ix in 1:n_x
        for iy in 1:n_y
            for i in 1:9
                ix_new = ix - c[i, 1]
                iy_new = iy - c[i, 2]

                if ix_new < 1 || ix_new > n_x || iy_new < 1 || iy_new > n_y
                    g_new[i, ix, iy] = 0.0
                else
                    g_new[i, ix, iy] = g[i, ix_new, iy_new]
                end
            end
        end
    end

    # Boundary Conditions

    x_0 = Int(floor((0.5 - 0.5 * inlet_width) * n_x))
    x_1 = Int(floor((0.5 + 0.5 * inlet_width) * n_x))

    # Adiabatic Left 
    ix = 1
    Threads.@threads for iy in 1:n_y
        _f =  f[:,ix,iy]
        v = 1 - (_f[5] + _f[1] + _f[3] + 2 * (_f[7] +  _f[4] + _f[8])) / rho_0
        T = 6 * (g_new[8,ix,iy] + g_new[4,ix,iy] + g_new[7,ix,iy]) / (1.0 - 3*v)
        g_new[6,ix,iy] = (1.0 / 36.0) * T * (1 + 3*v)
        g_new[2,ix,iy] = (1.0 / 9.0) * T * (1 + 3*v)
        g_new[9,ix,iy] = (1.0 / 36.0) * T * (1 + 3*v)
    end

    # Adiabatic Right
    ix = n_x
    Threads.@threads for iy in 1:n_y
        _f =  f[:,ix,iy]
        v = 1 - (_f[5] + _f[1] + _f[3] + 2 * (_f[6] + _f[2] + _f[9])) / rho_0
        T = 6 * (g_new[6,ix,iy] + g_new[2,ix,iy] + g_new[9,ix,iy]) / (1.0 - 3*v)
        g_new[8,ix,iy] = (1.0 / 36.0) * T * (1 + 3*v)
        g_new[4,ix,iy] = (1.0 / 9.0) * T * (1 + 3*v)
        g_new[7,ix,iy] = (1.0 / 36.0) * T * (1 + 3*v)
    end

    # Adiabatic Bottom
    iy = 1
    Threads.@threads for ix in 1:n_x
        _f =  f[:,ix,iy]
        v = 1 - (_f[1] + _f[2] + _f[4] + 2 * (_f[5] + _f[8] + _f[9])) / rho_0
        T = 6 * (g_new[8,ix,iy] + g_new[5,ix,iy] + g_new[9,ix,iy]) / (1.0 - 3*v)
        g_new[6,ix,iy] = (1.0 / 36.0) * T * (1 + 3*v)
        g_new[3,ix,iy] = (1.0 / 9.0) * T * (1 + 3*v)
        g_new[7,ix,iy] = (1.0 / 36.0) * T * (1 + 3*v)
    end

    # Adiabatic Top
    iy = n_y
    Threads.@threads for ix in 1:n_x
        _f = f[:,ix,iy]
        v = 1 - (_f[1] + _f[2] + _f[4] + 2 * (_f[7] + _f[3] + _f[6])) / rho_0
        T = 6 * (g_new[6,ix,iy] + g_new[3,ix,iy] + g_new[7,ix,iy]) / (1.0 - 3*v)
        g_new[8,ix,iy] = (1.0 / 36.0) * T * (1 + 3*v)
        g_new[5,ix,iy] = (1.0 / 9.0) * T * (1 + 3*v)
        g_new[9,ix,iy] = (1.0 / 36.0) * T * (1 + 3*v)
    end

    # Inlet Dirichlet
    iy = 1
    Threads.@threads for ix in x_0:x_1
        _f = @view f[:,ix,iy]
        _g = @view g_new[:, ix, iy]
        v = 1 - (_f[1] + _f[2] + _f[4] + 2 * (_f[5] + _f[8] + _f[9])) / rho_0
        T = 6*(T_in - (_g[1]+_g[2]+_g[4]+_g[5]+_g[8]+_g[9])) / (1 + 3*v)
        g_new[6, ix, iy] = 1/36*T*(1+3*v)
        g_new[3, ix, iy] = 1/9*T*(1+3*v)
        g_new[7, ix, iy] = 1/36*T*(1+3*v)
    end
    
   
 
    g .= g_new
end



function run_forward!(gamma, rho, u, v, p, q_t, T, f, g, alpha_gamma, beta_gamma, tau_f, tau_g, F, Q_t)
    x = range(0, stop=L_x, length=n_x)
    y = range(0, stop=L_y, length=n_y)

    init_pops!(f, g, rho, u, v, T)
    
    for t in 1:n_t


        if mod1(t, output_interval) == 1

            index = Int(floor(t/output_interval))

            index = lpad(index,3,"0")

            T_hm = heatmap(y, x, T)
            savefig("run/step_$(index)_T.png")
            u_hm = heatmap(y, x, sqrt.(u.^2 + v.^2))
            savefig("run/step_$(index)_u.png")
            F_hm = heatmap(y, x, sqrt.(F[1,:,:].^2 + F[2,:,:].^2))
            savefig("run/step_$(index)_F.png")
            rho_hm = heatmap(y, x, rho)
            savefig("run/step_$(index)_rho.png")
            gamma_hm = heatmap(y, x, gamma)
            savefig("run/step_$(index)_gamma.png")
            heatmap(y,x,p)
            savefig("run/step_$(index)_p.png")

            #println("Time: $ti")
    
        end

        relax!(f, g, p, rho, u, v, T, tau_f, tau_g)

        compute_Q_t!(Q_t, T, beta_gamma)
        compute_F!(F, u, v, alpha_gamma)

        force!(f, g, F, Q_t)

        advect_f!(f)
        advect_g!(g, f)

        compute_moments!(rho, u, v, p, q_t, T, f, g)
        

        print("Step: $(t), Time: $(t * dt)\r")
    end
end


function create_gamma_from_noise!(gamma)
    noise = opensimplex2_2d()
    Threads.@threads for i in 1:n_x
        for j in 1:n_y
            xl = (i - 1 + 0.5) * dx
            yl = (j - 1 + 0.5) * dx
            
            if yl > 0.5 && yl < 1.5
                if sample(noise, 10*xl, 10*yl) > 0.7
                    gamma[i, j] = 0.0
                end
            end
        end
    end
end


function initial_conditions!(rho, u, v, T)
    Threads.@threads for i in 1:n_x
        for j in 1:n_y
            rho[i, j] = rho_0
            u[i, j] = 0
            v[i, j] = 0
            T[i,j] = T_0
        end
    end
end

function run_adjoint(f_i, g_i, u_i, v_i, T_i, m_i, q_i, alpha_gamma, beta_gamma, u, v)

end


function main()
    `rm run/"*"`
    # allocation
    gamma = ones(Float32, n_x, n_y)
    rho = zeros(Float32, n_x, n_y)
    u = zeros(Float32, n_x, n_y)
    v = zeros(Float32, n_x, n_y)
    p = zeros(Float32, n_x, n_y)
    q_t = zeros(Float32, 2, n_x, n_y)
    T = zeros(Float32, n_x, n_y)
    alpha_gamma = zeros(Float32, n_x, n_y)
    beta_gamma = zeros(Float32, n_x, n_y)

    f = zeros(Float32, 9, n_x, n_y)
    g = zeros(Float32, 9, n_x, n_y)

    F = zeros(Float32, 2, n_x, n_y)
    Q_t = zeros(Float32, n_x, n_y)

    # initial conditions
    create_gamma_from_noise!(gamma)

    # tau_f = compute_tau_f()$
    tau_f = 0.8
    tau_g = compute_tau_g()

    alpha_gamma = compute_alpha_gamma(gamma)
    beta_gamma = compute_beta_gamma(gamma)

    initial_conditions!(rho, u, v, T)

    # run forward
    run_forward!(gamma, rho, u, v, p, q_t, T, f, g, alpha_gamma, beta_gamma, tau_f, tau_g, F, Q_t)

    # adjoint
    # run_adjoint!()

    # sensitivity computation
end

end