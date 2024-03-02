
module LBM

using Plots


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
const Htfactor = 2e2
const ramp_p = 30.0

const resolution = 100

const nu = 0.0001
const k = 0.0001

const n_t = 1000

## Dependent Parameters

const n_x = L_x * resolution
const n_y = L_y * resolution

const dx = 1 / resolution

const tau_f = 3 * nu / dx + 0.5
const tau_g = 3 * k / dx + 0.5

const dt = dx


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

function init_pops!(f, g, rho, u, v, T)
    for ix in 1:n_x
        for iy in 1:n_y

            f_eq = f_equilibrium(rho[ix, iy], u[ix, iy], v[ix, iy])
            g_eq = g_equilibrium(T[ix, iy], u[ix, iy], v[ix, iy])

            for i in 1:9
                f[i, ix, iy] = f_eq[i]
                g[i, ix, iy] = g_eq[i]
            end
        end
    end
end

function f_relax!(f, tau_f, rho, u, v)
    for ix in 1:n_x
        for iy in 1:n_y
            f_eq = f_equilibrium(rho[ix,iy], u[ix,iy], v[ix,iy])

            for i in 1:9
                f[i,ix,iy] -= 1/tau_f * (f[i,ix,iy] - f_eq[i])
            end
        end
    end
end



function g_relax!(g, tau_g, T, u, v, T_ref)
    for ix in 1:n_x
        for iy in 1:n_y
            g_eq = g_equilibrium(T[ix,iy], u[ix,iy], v[ix,iy])

            for i in 1:9
                g[i,ix,iy] -= 1/tau_g * (g[i,ix,iy] - g_eq[i])
            end
        end
    end
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

            u[ix, iy] /= rho[ix, iy]
        end
    end
end

function source(x,y)
    return exp(-((x - 0.5)^2 + (y - 0.25)^2)*10000)
end

function apply_source_term!(rho, u, v, T)
    for ix in 1:n_x
        for iy in 1:n_y
            
            x = (ix - 1) * dx
            y = (iy - 1) * dx

            
            T[ix, iy] += source(x,y) * dt
        end
    end
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
    for ix in 1:n_x
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

x = range(0, stop=L_x, length=n_x)
y = range(0, stop=L_y, length=n_y)



# Initial condition
rho = ones(Float64, n_x, n_y)
gamma = ones(Float64, n_x, n_y)


u = zeros(Float64, n_x, n_y)
v = zeros(Float64, n_x, n_y)
T = zeros(Float64, n_x, n_y)

# Reference temperature field

T_ref = zeros(Float64, n_x, n_y)

for i in 1:n_x
    for j in 1:n_y
        xl = (i - 1) * dx
        yl = (j - 1) * dx

        u[i, j] = 0
        v[i, j] = 0.01 #xl < 0.5*L_x ? 0.1 : -0.1
        T[i, j] = 0 #exp(-((x[i] - 0.5)^2 + (y[j] - 0.25)^2) * 1000.0)
        # central gamma obstacle
        #if (xl - 0.5)^2 + (yl - 0.25)^2 < 0.1^2
        gamma[i, j] = exp(-((x[i] - 0.5)^2 + (y[j] - 0.25)^2) * 1000.0)
        #end
        T_ref[i, j] = exp(-((x[i] - 0.5)^2 + (y[j] - 0.25)^2) * 1000.0)
    end
end

function main() 
    ENV["GKSwstype"]="nul"; loadpath = "./run-better"; anim = Animation(loadpath,String[]); println("Animation directory: $(anim.dir)")

    f = zeros(Float64, 9, n_x, n_y)
    g = zeros(Float64, 9, n_x, n_y)

    f_dash = zeros(Float64, 9, n_x, n_y)
    g_dash = zeros(Float64, 9, n_x, n_y)

    init_pops!(f, g, rho, u, v, T)


    for t in 1:n_t
        compute_moments!(f, g, rho, u, v, T)
        apply_source_term!(rho, u, v, T)
        f_relax!(f, tau_f, rho, u, v)
        g_relax!(g, tau_g, T, u, v, T_ref)
        stream!(f, f_dash)
        stream!(g, g_dash)
        f .= f_dash
        g .= g_dash

        if t % 1 == 0
            ti = t * dt
            heatmap(x, y, transpose(T))
            savefig("$loadpath/T_$t.png")
            
            print("Time: $ti\r")
        end
    end
end

end