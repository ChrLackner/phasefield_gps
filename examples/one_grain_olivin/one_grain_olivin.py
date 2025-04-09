
import ngsolve as ngs
from phasefield_gps import *
import ngsolve.meshes as ngmeshes
import matplotlib.pyplot as plt
import random
import numpy as np

initial_temperature = 1700.15 # K
end_temperature = 1373.15 # K
cooling_rate = 1/3600 # K/s

conc_liquid = 0.18

# mesh
nel = 50
nx = nel
ny = nel
mesh = ngmeshes.MakeStructured2DMesh(
    quads=True, nx=nx, ny=ny,
    periodic_x=False, periodic_y=False,
    mapping = lambda x,y: (1e-3*x-5e-4, 1e-3*y - 5e-4))

# this fixes angle to 0:
r1 = 0
mat = np.array([[np.cos(r1), -np.sin(r1)], [np.sin(r1), np.cos(r1)]])
factor = 6 # 1e4
v1s = []
v2s = []

v1 = np.array([1, 0])
v2 = np.array([0, 1])
v3 = np.array([1, 1])
v4 = np.array([-1, 1])
v1s.append(ngs.CF(tuple(np.dot(mat, v1))))
v2s.append(ngs.CF(tuple(np.dot(mat, v2))))

directional_surface_energies = { tuple(np.dot(mat, v1)) : 1.00 * 2.45 * factor,
                                 tuple(np.dot(mat, v2)) : 0.50 * 2.45 * factor,
                                 tuple(np.dot(mat, v3)) : 0.83 * 2.45 * factor,
                                 tuple(np.dot(mat, v4)) : 0.83 * 2.45 * factor }
                                 
D0_sol = 10**(-8.27)
Ea = 226000
P = 1e5
V = 7e-6
R = 8.31446261815324 # J K^-1 mol^-1
Feterm = lambda concentration: 3*((1-concentration) - 0.14)


temp = ngs.Parameter(initial_temperature)

diff_coef_solid1 = lambda concentration: D0_sol * ngs.exp(-(Ea+(P-1e5)*V)/(R*temp)) *10**Feterm(concentration) * (ngs.OuterProduct(v1s[0], v1s[0]) + 6 * ngs.OuterProduct(v2s[0], v2s[0]))
#diff_coef_solid1 = lambda concentration: D0_sol * ngs.exp(-(Ea+(P-1e5)*V)/(R*temp)) *10**Feterm(concentration) * (ngs.OuterProduct(v1s[0], v1s[0]) + 6 * 3e-12 * ngs.OuterProduct(v2s[0], v2s[0]))
if_mobility = 1e-12 * ngs.exp(0 * temp)

liquid = Phase("liquid", diffusion_coefficient=3e-7)
solid = Phase("solid", diffusion_coefficient=diff_coef_solid1,
              surface_energies= { liquid: 2.45*
              factor, #directional_surface_energies,
                                  "kappa": 0.2 })

fosterite_liquid = "foL_1373-2273-10.tab"
fosterite_solid = "fo_1373-2273-10.tab"
fayalite_liquid = "faL_1373-2273-10.tab"
fayalite_solid = "fa_1373-2273-10.tab"

fosterite = IdealSolutionComponent(name="Fosterite",
                                   phase_energies={ liquid: fosterite_liquid,
                                                    solid: fosterite_solid })
fayalite = IdealSolutionComponent(name="Fayalite",
                                  phase_energies={ liquid: fayalite_liquid,
                                                   solid: fayalite_solid })

model = GrandPotentialSolver(mesh=mesh,
                             components=[fosterite, fayalite],
                             phases=[liquid, solid],
                             molar_volume=4.3e-5, # m**3/mol
                             interface_mobility=if_mobility, # 1e-12, # m**4/J/s
                             temperature=temp, # K
                             interface_width=5e-5) # m
model.order = 2

r = 10 * 1e-5
rx = ngs.sqrt(ngs.x**2 + ngs.y**2) - r

# compute energy diagrams for full liquid concentation
# and from there calculate the concentration a new solid grain would be
# created with (having parallel energy tangents)
model.set_initial_conditions({ liquid: ngs.CF(1),
                               solid: ngs.CF(0) },
                             { fosterite: { liquid: conc_liquid,
                                            solid: 0 },
                               fayalite: { liquid: 1 - conc_liquid,
                                           solid: 1}})
conc_solid = solid.calculate_parallel_energy_concentration(conc_liquid,
                                                           liquid,
                                                           initial_temperature,
                                                           [fosterite, fayalite])
print("calculated concentration in solid2:", conc_solid)
ic_concentrations : dict[Component, dict[Phase, ngs.CF |
                                         float]] = {
                                             fosterite: { liquid: conc_liquid,
                                                          solid: conc_solid }}
ic_concentrations[fayalite] = { liquid: 1-ic_concentrations[fosterite][liquid],
                                solid: 1-ic_concentrations[fosterite][solid] }

tanh = lambda x: ngs.sinh(x)/ngs.cosh(x)
ic_solid = 1-(0.5 * (1 + tanh(ngs.sqrt(model.m(False)/model.kappa(False)) * rx)))
ic_liquid = 1-ic_solid

model.set_initial_conditions({ liquid: ic_liquid,
                               solid: ic_solid },
                             components=ic_concentrations)

def cooling(time):
    temp = initial_temperature - (cooling_rate * time)
    if temp < end_temperature:
        temp = end_temperature
    return temp

for comp, conc in model.get_concentrations().items():
    ngs.Draw(conc, mesh, f"{comp.name}")
ngs.Draw(model.get_phase(solid), mesh, "solid")
ngs.Draw(model.get_phase(liquid), mesh, "liquid")

nump = 1001
parr = np.array([i/nump for i in range(nump+1)]) * 5e-4
x_dir = mesh(parr, 0)
y_dir = mesh(0, parr)
x_y_dir = mesh(parr/np.sqrt(2), parr/np.sqrt(2))

nump_e = 101
parr_e = np.array([i/nump_e for i in range(nump_e+1)]) * 5e-4
x_dir_e = mesh(parr_e, 0)
y_dir_e = mesh(0, parr_e)
x_y_dir_e = mesh(parr_e/np.sqrt(2), parr_e/np.sqrt(2))

dirs = { "x" : x_dir,
         "y" : y_dir,
         "sqrt(x^2+y^2)" : x_y_dir }

dirs_energy = { "x" : x_dir_e,
                "y" : y_dir_e,
                "sqrt(x^2+y^2)" : x_y_dir_e }

fosterite_plots = { "x" : [],
                    "y" : [],
                    "sqrt(x^2+y^2)" : [] }

import os
os.makedirs("results", exist_ok=True)

def plot_function_over_directions(func, name, value_dict):
    for key, lst in value_dict.items():
        lst.append(func(dirs[key]))
    fig, ax = plt.subplots(3,1)
    for i, (key, lst) in enumerate(value_dict.items()):
        for v in lst:
            ax[i].plot(parr, v)
        ax[i].set_xlabel(key)
        ax[i].set_ylabel(name)
        ax[i].set_title(f"{name} over {key}")
    plt.tight_layout()
    fig.savefig(os.path.join("results", f"{name}.png"))
    plt.close('all')

if os.path.exists(os.path.join("results", "energy")):
    for f in os.listdir(os.path.join("results", "energy")):
        os.remove(os.path.join("results", "energy", f))
os.makedirs(os.path.join("results", "energy"), exist_ok=True)

def plot_energy_over_directions(energy, concentration):
    fig = model.plot_energy_landscape(times=3)
    ax = fig.get_axes()
    for i, (key, d) in enumerate(dirs_energy.items()):
        v = energy(d)
        c = concentration(d)
        ax[i].plot(c, v, "o")
        ax[i].set_xlabel("Concentration")
        ax[i].set_ylabel("Energy")
        ax[i].set_title(f"{key}")
    plt.tight_layout()
    fig.savefig(os.path.join("results", "energy", f"time_{model.time:.2f}.png"))
    plt.close('all')

output_file = os.path.join("results", "output.csv")
with open(output_file, "w") as f:
    f.write("time; Temp; XMg1_x; XMg2_x; XMg3_x; XMg4_x; XMg1_y; XMg2_y; XMg3_y; XMg4_y; XMg1_xy; XMg2_xy; XMg3_xy; XMg4_xy; IF_Pos_x; IF_Pos_y; IF_Pos_xy; GGW_Liq; GGW_Sol; Pot_Liq; Pot_Sol; Energy\n")
def write_to_file():
    pvals = { }
    xmg = model.get_concentrations()[fosterite]
    if_pos = {}
    for key, d in dirs.items():
        solid_vals = model.get_phase(solid)(d).flatten()
        #i1 = np.where(solid_vals > 0.9)[0][-1]
        #i2 = np.where(solid_vals < 0.1)[0][0]
        i1 = np.where(solid_vals > 0.8)[0][-1]
        l_vals = np.where(solid_vals < 0.2)[0]
        if len(l_vals) == 0:
            i2 = -1
        else:
            i2 = l_vals[0]
        p = [d[0], d[i1], d[i2], d[-1]]
        pvals[key] = [xmg(p[i]) for i in range(4)]
        if_pos[key] = 0.5 * (i1 + i2)/len(d)
    with open(output_file, "a") as f:
        f.write(f"{model.time:.2f}; {model.T.Get():.5f}; ")
        for key, vals in pvals.items():
            for v in vals:
                f.write(f"{v:.4f}; ")
        for key, vals in if_pos.items():
            f.write(f"{vals:.4f}; ")
        cvals = np.linspace(0, 1, 1001)[1:-1]
        e_solid = [solid.get_chemical_energy({ fosterite: c, fayalite: 1-c }, model.T.Get(), use_ifpos=False) for c in cvals]
        e_liquid = [liquid.get_chemical_energy({ fosterite: c, fayalite: 1-c }, model.T.Get(), use_ifpos=False) for c in cvals]
        min_val_s = np.argmin(e_solid)/1001
        min_val_l = np.argmin(e_liquid)/1001
        f.write(f"{min_val_l:.4f}; ")
        f.write(f"{min_val_s:.4f}; ")
        potential_solid = model.gfw(p[0])
        potential_liquid = model.gfw(p[3])
        f.write(f"{potential_liquid:.4f}; ")
        f.write(f"{potential_solid:.4f}; ")
        f.write(f"{ngs.Integrate(model.get_energy(), mesh):.4f}; ")
        f.write(f"\n")

vol = ngs.Integrate(1, mesh)

def set_time():
    model.set_Temperature(cooling(model.time))

model.time_set_callback = set_time
timestep = 0
import time
def callback():
    global timestep
    timestep += 1
    if timestep % 3 == 0 and model.dt.Get() < 10000:
        model.set_timestep(model.dt.Get()*2)
        print("increase timestep to", model.dt.Get())

    if timestep % 10 == 1:
        plot_function_over_directions(model.get_concentrations()[fosterite], "fosterite", fosterite_plots)
    plot_energy_over_directions(model.Vm * model.get_chemical_energy(), model.get_concentrations()[fosterite])
    write_to_file()

    print(f"liquid/solid = {ngs.Integrate(model.get_phase(liquid), mesh)/vol} / {ngs.Integrate(model.get_phase(solid), mesh)/vol}")
    print(f"Time: {model.time}, energy: {ngs.Integrate(model.get_energy(), mesh)}")
    ngs.Redraw()

with ngs.TaskManager():
    plot_energy_over_directions(model.Vm * model.get_chemical_energy(), model.get_concentrations()[fosterite])
    model.set_timestep(0.01)
    model.do_timestep()
    callback()
    model.set_timestep(0.1)
    model.solve(1e6, callback=callback)

print("finish")
