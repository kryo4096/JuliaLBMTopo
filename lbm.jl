module LBM

using Plots
using .Threads
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
const USE_GPU = false
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

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

## PARAMETERS
const L_x = 1
const L_y = 1

const Ht = 150e-6
const Htfactor = 2e5
const ramp_p = 1e-6
const ramp_k = 1e-6
const K_f = 0.0001
const K_s = 0.05
const K_sub = 0.005
const kappa_cs = 0.005


const resolution = 100

const nu = 0.0001

const t_end = 10.0

## Dependent Parameters

const n_x = L_x * resolution
const n_y = L_y * resolution

const dx = 1 / resolution

const tau_f = 3 * nu / dx + 0.5

print(tau_f)

# const tau_g = 3 * K / dx + 0.5
const tau_g_s = 3 * K_sub / dx + 0.5

const dt = dx

const n_t = Int(t_end / dt)

function pow(x,y)
    return x^y
end

# Equation 16
 function f_equilibrium!(rho, u, v, f_eq)
    for i in 1:9
        cu = c[i, 1] * u + c[i, 2] * v
        f_eq[i] = w[i] * rho * (1 + 3 * cu + 9 / 2 * cu^2 - 3 / 2 * (u^2 + v^2))

        #f_eq[i] = w[i] * rho * (2 - sqrt(1 + 3 * u^2) ) * (2-sqrt(1+3*v^2)) * pow((2*u+ sqrt(1 + 3*u^2)) / (1-u), c[i,1]) * pow((2 * v + sqrt(1+ 3 * v^2)) / (1-v), c[i,2]);
    end
end

# Equation 17
function g_equilibrium!(T, u, v, g_eq)
	for i in 1:9
		cu = c[i, 1] * u + c[i, 2] * v
		g_eq[i] = w[i] * T * (1 + 3 * cu)
	end
end

@parallel_indices (ix,iy) function init_pops!(f, g_c, g_s, rho, u, v, T_c, T_s)
    
    # Threads.@threads for ix in 1:n_x
    #     for iy in 1:n_y
    if ix >= 1 && ix <= n_x && iy >= 1 && iy <= n_y
            f_eq = zeros(Float64, 9)
            g_c_eq = zeros(Float64, 9)
            g_s_eq = zeros(Float64, 9)

            f_equilibrium!(rho[ix, iy], u[ix, iy], v[ix, iy], f_eq)
            g_equilibrium!(T_c[ix, iy], u[ix, iy], v[ix, iy], g_c_eq)
            g_equilibrium!(T_s[ix, iy], 0.0, 0.0, g_s_eq)

            for i in 1:9
                f[i, ix, iy] = f_eq[i]
                g_c[i, ix, iy] = g_c_eq[i]
                g_s[i, ix, iy] = g_s_eq[i]
            end
    end
    return nothing
end

@parallel_indices (ix, iy) function f_relax!(f, tau_f, rho, u, v)

    # f_eqs = [zeros(Float64, 9) for i in 1:Threads.nthreads()]
    f_eqs = zeros(Float64,9)
    if ix >= 1 && ix <= n_x && iy >= 1 && iy <= n_y
            f_equilibrium!(rho[ix,iy], u[ix,iy], v[ix,iy], f_eqs)


            for i in 1:9
                f[i,ix,iy] -= 1/tau_f * (f[i,ix,iy] - f_eqs[i])
            end
    end
    return nothing
end

# equation 20
function apply_damping!(f, rho, u, v, alpha_gamma)
    Threads.@threads for ix in 1:n_x
        for iy in 1:n_y

            #u[ix,iy] /= (1 + alpha_gamma[ix, iy]*dt*3)
            #v[ix,iy] /= (1 + alpha_gamma[ix, iy]*dt*3)

            for i in 1:9
                c_alpha_gamma_u = (c[i, 1] * u[ix, iy] + c[i, 2] * v[ix, iy]) * alpha_gamma[ix, iy]
                f[i, ix, iy] -= 3 * dx * w[i] * c_alpha_gamma_u
            end
        end
    end
end

function g_c_relax!(g, tau_g, T, u, v, T_ref)

    g_c_eqs = [zeros(Float64, 9) for i in 1:Threads.nthreads()]

    Threads.@threads for ix in 1:n_x
        for iy in 1:n_y
            g_equilibrium!(T[ix,iy], u[ix,iy], v[ix,iy], g_c_eqs[Threads.threadid()])

            for i in 1:9
                g[i,ix,iy] -= 1/tau_g[ix,iy] * (g[i,ix,iy] - g_c_eqs[Threads.threadid()][i])
            end
        end
    end
end

function g_s_relax!(g, T, T_ref)

    g_s_eqs = [zeros(Float64, 9) for i in 1:Threads.nthreads()]

    Threads.@threads for ix in 1:n_x
        for iy in 1:n_y
            g_eq = g_equilibrium!(T[ix,iy], 0, 0, g_s_eqs[Threads.threadid()])

            for i in 1:9
                g[i,ix,iy] -= 1/tau_g_s * (g[i,ix,iy] - g_s_eqs[Threads.threadid()][i])
            end
        end
    end
end

@parallel_indices (ix,iy) function compute_moments!(f, g_c, g_s, rho, u, v, P, T_c, T_s)
    if ix >= 1 && ix <= n_x && iy >= 1 && iy <= n_y
            rho[ix, iy] = 0
            u[ix, iy] = 0
            v[ix, iy] = 0
            P[ix, iy] = 0
            T_c[ix, iy] = 0
            T_s[ix, iy] = 0

            for i in 1:9
                rho[ix, iy] += f[i, ix, iy]
                u[ix, iy] += c[i, 1] * f[i, ix, iy]
                v[ix, iy] += c[i, 2] * f[i, ix, iy]
                P[ix, iy] += 3 * (c[i, 1] * u[ix, iy] + c[i, 2] * v[ix, iy])^2 * f[i, ix, iy]
                T_c[ix, iy] += g_c[i, ix, iy]
                T_s[ix, iy] += g_s[i, ix, iy]
            end

            u[ix, iy] /= rho[ix, iy]
            v[ix, iy] /= rho[ix, iy]
    end
    return nothing
end

@parallel_indices (ix, iy) function apply_heat_source!(rho, T, source)
    if ix >= 1 && ix <= n_x && iy >= 1 && iy <= n_y
            T[ix, iy] += source[ix,iy] * dt

    end
    return nothing
end

function bindex(i)
    j = i
    if j == 3
        j = 5
    elseif j == 5
        j = 3
    elseif j > 5
        j = ((j - 6) + 2) % 4 + 6
    end

    return j
end

function stream!(pop_old, pop_new)
    Threads.@threads for ix in 1:n_x
        for iy in 1:n_y
            for i in 1:9
                
                iy_new = iy - c[i, 2]
                
                
                if false #iy_new < 1 || iy_new > n_y
                    pop_new[i, ix, iy] = pop_old[bindex(i), ix, iy]
                else
                    iy_new = mod1(iy - c[i, 2], n_y)
                    ix_new = mod1(ix - c[i, 1], n_x)
                    pop_new[i, ix, iy] = pop_old[i, ix_new, iy_new]
                end
                    
            end
        end
    end
end


function alpha_gamma(gamma)
    return @. Htfactor * ramp_p * ( 1 - gamma ) / (ramp_p + gamma)
end

function K_gamma(gamma)
    return @. K_f + (K_s - K_f) * ramp_k * ( 1 - gamma ) / (ramp_k + gamma)
end

function tau_g(K_gamma)
    return @. 3 * K_gamma / dx + 0.5
end

x = range(0, stop=L_x, length=n_x)
y = range(0, stop=L_y, length=n_y)



# Initial condition
rho = ones(Float64, n_x, n_y)
gamma = ones(Float64, n_x, n_y)

u = zeros(Float64, n_x, n_y)
v = zeros(Float64, n_x, n_y)
T_c = zeros(Float64, n_x, n_y)
T_s = zeros(Float64, n_x, n_y)
P = zeros(Float64, n_x, n_y)
# Reference temperature field

T_ref = zeros(Float64, n_x, n_y)

Q_s = zeros(Float64, n_x, n_y)
Q_c = zeros(Float64, n_x, n_y)

Threads.@threads for i in 1:n_x
    for j in 1:n_y
        xl = (i - 1 + 0.5) * dx
        yl = (j - 1 + 0.5) * dx

        u[i, j] = 0
        v[i, j] = 0.1 #xl < 0.5*L_x ? 0.1 : -0.1
        T_c[i, j] = 0 #exp(-((x[i] - 0.5)^2 + (y[j] - 0.25)^2) * 1000.0)
        T_s[i, j] = 0
        # central gamma obstacle
        #if (xl - 0.5)^2 + (yl - 0.25)^2 < 0.1^2
        gamma[i, j] = (1 .- 1.0*exp(-((x[i] - 0.5)^2 + (y[j] - 0.5)^2) * 1000.0))

        Q_s[i, j] = exp(-((x[i] - 0.5)^2 + (y[j] - 0.1)^2) * 10000.0) * 10.0 - exp(-((x[i] - 0.5)^2 + (y[j] - 0.9)^2) * 10000.0) * 10.0
    end
end

@parallel_indices (i,j) function cs_coupling!(T_c, T_s, dQ)
    if i >= 1 && i <= n_x && j >= 1 && j <= n_y
        dQ[i, j] = kappa_cs * (T_c[i, j] - T_s[i, j])
    end
    return nothing
end

function memcpy(dst, src)
    Threads.@threads for i in 1:n_x
        for j in 1:n_y
            for k in 1:9
                dst[k, i, j] = src[k, i, j]
            end
        end
    end
end

function main() 

    `rm run/'*'`

    ENV["GKSwstype"]="nul";

    println("Number of threads: $(Threads.nthreads())")

    f = @zeros(9, n_x, n_y)
    g_c = @zeros(9, n_x, n_y)
    g_s = @zeros(9, n_x, n_y)

    f_dash = @zeros(9, n_x, n_y)
    g_c_dash = @zeros(9, n_x, n_y)
    g_s_dash = @zeros(9, n_x, n_y)

    dQ = @zeros(n_x, n_y)

    @parallel init_pops!(f, g_c, g_s, rho, u, v, T_c, T_s)

    ag = alpha_gamma(gamma)
    tg = tau_g(K_gamma(gamma))


    for t in 1:n_t
        @parallel compute_moments!(f, g_c, g_s, rho, u, v, P, T_c, T_s)

        #v .+= 0.2 * dt
        @parallel cs_coupling!(T_c, T_s, dQ)
        @parallel apply_heat_source!(rho, T_s, Q_s + dQ)
        @parallel apply_heat_source!(rho, T_c, -dQ)

        @parallel f_relax!(f, tau_f, rho, u, v)

        apply_damping!(f, rho, u, v, ag)
       
        g_c_relax!(g_c, tg , T_c, u, v, T_ref)
        g_s_relax!(g_s, T_s, T_ref)

        stream!(f, f_dash)
        stream!(g_c, g_c_dash)
        stream!(g_s, g_s_dash)
        

        memcpy(f, f_dash)
        memcpy(g_c, g_c_dash)
        memcpy(g_s, g_s_dash)
        
        print("Time: $(t * dt)\r")

        if t % 100 == 0
        
            
            T_c_hm = heatmap(x, y, transpose(T_c))
            T_s_hm = heatmap(x, y, transpose(T_s))
            v_hm = heatmap(x, y, transpose(v), clim = (0, 0.15))
            
         

            p = plot(T_c_hm, T_s_hm, v_hm, layout = (1, 3), size = (2000, 500))

            savefig("run/$(lpad(t, 4, '0')).png")
            # display(p)

            #println("Time: $ti")
            
        end
    end

    #gif(anim, "run/anim.gif", fps = 30)
end

end # module
LBM.main()
