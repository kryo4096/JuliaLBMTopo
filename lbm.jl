module LBM

using Plots
using .Threads
using CoherentNoise
using HDF5



const c = [ 0   0;
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
const L_x = 2
const L_y = 1

const Ht = 150e-6
const Htfactor = 2e1
const ramp_p = 1e-6
const ramp_k = 1e-6
const K_f = 0.0001
const K_s = 0.0005
const K_sub = 0.0005
const kappa_cs = 1.0
const kappa_fs = 0.05
const kappa_ps = 1.0
const resolution = 200
const nu = 0.00001
const t_end = 200.0
const v_0 = 0.2

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

function pow(x, y)
	return x^y
end

# Equation 16
function f_equilibrium!(rho, u, v, f_eq)
	for i in 1:9
		#cu = c[i, 1] * u + c[i, 2] * v
		#f_eq[i] = w[i] * rho * (1 + 3 * cu + 9 / 2 * cu^2 - 3 / 2 * (u^2 + v^2))

		f_eq[i] = w[i] * rho * (2 - sqrt(1 + 3 * u^2)) * (2 - sqrt(1 + 3 * v^2)) * pow((2 * u + sqrt(1 + 3 * u^2)) / (1 - u), c[i, 1]) * pow((2 * v + sqrt(1 + 3 * v^2)) / (1 - v), c[i, 2])
	end
end

# Equation 17
function g_equilibrium!(T, u, v, g_eq)
	for i in 1:9
		cu = c[i, 1] * u + c[i, 2] * v
		g_eq[i] = w[i] * T * (1 + 3 * cu)
	end
end

function init_pops!(f, g_c, g_s, rho, u, v, T_c, T_s)

	Threads.@threads for ix in 1:n_x
		for iy in 1:n_y

			f_eq = zeros(Float32, 9)
			g_c_eq = zeros(Float32, 9)
			g_s_eq = zeros(Float32, 9)

			f_equilibrium!(rho[ix, iy], u[ix, iy], v[ix, iy], f_eq)
			g_equilibrium!(T_c[ix, iy], u[ix, iy], v[ix, iy], g_c_eq)
			g_equilibrium!(T_s[ix, iy], 0.0, 0.0, g_s_eq)

			for i in 1:9
				f[i, ix, iy] = f_eq[i]
				g_c[i, ix, iy] = g_c_eq[i]
				g_s[i, ix, iy] = g_s_eq[i]
			end
		end
	end
end

function f_relax!(f, tau_f, rho, u, v)

	f_eqs = [zeros(Float32, 9) for i in 1:Threads.nthreads()]

	Threads.@threads for ix in 1:n_x
		for iy in 1:n_y
			f_equilibrium!(rho[ix, iy], u[ix, iy], v[ix, iy], f_eqs[Threads.threadid()])

			#f_mirror = 2 * f_eq - f[:,ix,iy]

			for i in 1:9
				f[i, ix, iy] = f[i, ix, iy] - 1 / tau_f * (f[i, ix, iy] - f_eqs[Threads.threadid()][i])
			end
		end
	end
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

	g_c_eqs = [zeros(Float32, 9) for i in 1:Threads.nthreads()]

	Threads.@threads for ix in 1:n_x
		for iy in 1:n_y
			g_equilibrium!(T[ix, iy], u[ix, iy], v[ix, iy], g_c_eqs[Threads.threadid()])

			for i in 1:9
				g[i, ix, iy] -= 1 / tau_g[ix, iy] * (g[i, ix, iy] - g_c_eqs[Threads.threadid()][i])
			end
		end
	end
end

function g_s_relax!(g, T, T_ref)

	g_s_eqs = [zeros(Float32, 9) for i in 1:Threads.nthreads()]

	Threads.@threads for ix in 1:n_x
		for iy in 1:n_y
			g_eq = g_equilibrium!(T[ix, iy], 0, 0, g_s_eqs[Threads.threadid()])

			for i in 1:9
				g[i, ix, iy] -= 1 / tau_g_s * (g[i, ix, iy] - g_s_eqs[Threads.threadid()][i])
			end
		end
	end
end

function compute_moments!(f, g_c, g_s, rho, u, v, P, T_c, T_s)
	Threads.@threads for ix in 1:n_x
		for iy in 1:n_y
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
	end
end

function apply_heat_source!(rho, T, source)
	Threads.@threads for ix in 1:n_x
		for iy in 1:n_y

			x = (ix - 1) * dx
			y = (iy - 1) * dx

			T[ix, iy] += source[ix, iy] * dt

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
	Threads.@threads for ix in 1:n_x
		for iy in 1:n_y
			for i in 1:9

				iy_new = iy - c[i, 2]


				if iy_new < 1 || iy_new > n_y
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
	return @. Htfactor * ramp_p * (1 - gamma) / (ramp_p + gamma)
end

function K_gamma(gamma)
	return @. K_f + (K_s - K_f) * ramp_k * (1 - gamma) / (ramp_k + gamma)
end

function kappa_gamma(gamma)
	return @. kappa_fs + (kappa_cs - kappa_fs) * ramp_k * (1 - gamma) / (ramp_k + gamma)
end

function tau_g(K_gamma)
	return @. 3 * K_gamma / dx + 0.5
end

x = range(0, stop = L_x, length = n_x)
y = range(0, stop = L_y, length = n_y)



# Initial condition
rho = ones(Float32, n_x, n_y)
gamma = ones(Float32, n_x, n_y)

u = zeros(Float32, n_x, n_y)
v = zeros(Float32, n_x, n_y)
T_c = zeros(Float32, n_x, n_y)
T_s = zeros(Float32, n_x, n_y)
P = zeros(Float32, n_x, n_y)
# Reference temperature field

T_ref = zeros(Float32, n_x, n_y)

Q_s = zeros(Float32, n_x, n_y)
Q_c = zeros(Float32, n_x, n_y)

heat_mask = zeros(Float32, n_x, n_y)

noise = opensimplex2_2d()

Threads.@threads for i in 1:n_x
	for j in 1:n_y
		xl = (i - 1 + 0.5) * dx
		yl = (j - 1 + 0.5) * dx

		u[i, j] = 0.01 * randn()
		v[i, j] = 0.01 * randn()
		T_c[i, j] = 0 #exp(-((x[i] - 0.5)^2 + (y[j] - 0.25)^2) * 1000.0)
		T_s[i, j] = 0
		# central gamma obstacle
		#if (xl - 0.5)^2 + (yl - 0.25)^2 < 0.1^2

		radius = sqrt((xl - 1.0)^2 + (yl - 0.5)^2)

        if radius < 0.2
			#Q_s[i, j] = 10.0
			heat_mask[i, j] = 20.0
		end

        a = exp(-radius^2 * 100.0)

		if abs(xl - 1.0) < 0.3
			if sample(noise, 10xl,10yl) > 0.7 #mod(yl+0.05*sin(xl*2*pi),0.1) > 0.05
				gamma[i, j] = 0.0
			end
		end

        

		



	end
end

function cs_coupling!(T_c, T_s, dQ)
	Threads.@threads for i in 1:n_x
		for j in 1:n_y
			dQ[i, j] = (T_c[i, j] - T_s[i, j])
		end
	end
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

function fix_temp!(T, T_ref)
	Threads.@threads for i in 1:n_x
		for j in 1:n_y
			if (i - 1 + 0.5) * dx < 0.5
				T[i, j] = T_ref[i, j]
			end
		end
	end
end


function objective(gamma_)
	f = zeros(Float32, 9, n_x, n_y)
	g_c = zeros(Float32, 9, n_x, n_y)
	g_s = zeros(Float32, 9, n_x, n_y)

	f_dash = zeros(Float32, 9, n_x, n_y)
	g_c_dash = zeros(Float32, 9, n_x, n_y)
	g_s_dash = zeros(Float32, 9, n_x, n_y)

	dQ = zeros(Float32, n_x, n_y)
	dQ_p = zeros(Float32, n_x, n_y)

	init_pops!(f, g_c, g_s, rho, u, v, T_c, T_s)

	ag = alpha_gamma(gamma)
	tg = tau_g(K_gamma(gamma))
	kappa = kappa_gamma(gamma)

	it = 0

    pressure_gradient = 0.01

    vel = 0.1

	for t in 1:n_t
		compute_moments!(f, g_c, g_s, rho, u, v, P, T_c, T_s)

        max_vel = maximum(u)

		u .+= pressure_gradient * dt


		fix_temp!(T_c, 0.0)
		#fix_temp!(T_s, -5.0)

		cs_coupling!(T_c, T_s, dQ)
		cs_coupling!(T_s, Q_s, dQ_p)

		dQ .*= kappa

		apply_heat_source!(rho, T_s, dQ + heat_mask * dt)
		apply_heat_source!(rho, T_c, -dQ)

		f_relax!(f, tau_f, rho, u, v)

		apply_damping!(f, rho, u, v, ag)

		g_c_relax!(g_c, tg, T_c, u, v, T_ref)
		g_s_relax!(g_s, T_s, T_ref)

		stream!(f, f_dash)
		stream!(g_c, g_c_dash)
		stream!(g_s, g_s_dash)

		memcpy(f, f_dash)
		memcpy(g_c, g_c_dash)
		memcpy(g_s, g_s_dash)

		print("Time: $(t * dt), it=$it, max_temp=$(maximum(T_s)),  max_vel=$max_vel                                           \r")

		if t % 50 == 0

			it_text = lpad(it, 4, "0")

			T_c_hm = heatmap(x, y, transpose(T_c), size = (1000, 400), clim=(0, 1.5))
			savefig("run/T_c_$it_text.png")
			T_s_hm = heatmap(x, y, transpose(T_s), size = (1000, 400), clim=(0, 1.5))
			savefig("run/T_s_$it_text.png")
			u_hm = heatmap(x, y, transpose(sqrt.(u .^ 2 + v .^ 2)), size = (1000, 400), clim=(0,0.3))
			savefig("run/u_$it_text.png")
			gamma_hm = heatmap(x, y, transpose(gamma), size = (1000, 400))
			savefig("run/gamma_$it_text.png")




			# h5open("run/fields_$it.h5", "w") do file
			#     write(file, "T_c", T_c)
			#     write(file, "T_s", T_s)
			#     write(file, "u", u)
			#     write(file, "v", v)
			#     write(file, "rho", rho)
			#     write(file, "gamma", gamma)
			# end


			# display(p)

			#println("Time: $ti")

			it += 1
		end
	end
end

function main()

	ENV["GKSwstype"] = "nul"

	println("Number of threads: $(Threads.nthreads())")

	objective(1)

	#gif(anim, "run/anim.gif", fps = 30)
end

end # module 
