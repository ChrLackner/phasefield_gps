import ngsolve as ngs
from ngsolve.meshes import *
from phasefield_gps import *

liquid = Phase("liquid", diffusion_coefficient=3e-6)    #3e-7
solid = Phase("solid", diffusion_coefficient=3e-8,     #3e-12
            surface_energies={ liquid : 2.451e4 }) # J/m**2



fosterite = IdealSolutionComponent(name="Fosterite",
                                   phase_energies={ liquid: -2612808,   #1373.15 = -2280010    #2173.15 = -2612808
                                                    solid: -2611896 })  #1373.15 = -2307279    #2173.15 = -2611896
fayalite = IdealSolutionComponent(name="Fayalite",
                                  phase_energies={ liquid: -2122255,    #1373.15 = -1699537    #2173.15 = -2122255
                                                   solid: -2075201 })   #1373.15 = -1706086    #2173.15 = -2075201
# mesh
nx = 100
ny = 10
mesh = MakeStructured2DMesh(quads=True, nx=nx, ny=ny,
                            periodic_x=False, periodic_y=False,
                            mapping = lambda x,y: (1e-3*x-0.5e-3, 0.1e-3 * y))

model = GrandPotentialSolver(mesh=mesh,
                             components=[fosterite, fayalite],
                             phases=[liquid, solid],
                             molar_volume=4.3e-5, # m**3/mol
                             interface_mobility=1e-13, # m**4/J/s
                             temperature=2173.15, # K
                             interface_width=5e-5) # m

# model.set_interface_anisotropy(theta0=0, epsilon=0.2)

model.mass_conservation = True

# initial conditions for phase
# wenn ...(x+0).., dann genau Mitte (-0.5e-3..0..+0.5e-3). x+0.4e-3.. = verschiebt IF nach links
tanh = lambda x: ngs.sinh(x)/ngs.cosh(x)
# shift = -0.45e-3
shift = 0 #0.4e-3

# ic_concentrations = { fayalite: { liquid: 0.999,
#                                    solid: 0.98 }}
ic_concentrations = { fayalite: { liquid: 0.4, solid: 0.6 }}

ic_concentrations[fosterite] = { liquid: 1-ic_concentrations[fayalite][liquid],
                                 solid: 1-ic_concentrations[fayalite][solid] }

print(model.m(False)/model.kappa(False))
ic_liquid = 0.5 * tanh(ngs.sqrt(model.m(False)/model.kappa(False)) * (ngs.x+shift)) + 0.5
model.set_initial_conditions({ liquid: ic_liquid,
                               solid: 1-ic_liquid },
                             components=ic_concentrations)

Draw(1e-4 * model.get_phase(liquid), mesh, "liquid")
Draw(1e-4 * model.get_phase(solid), mesh, "solid")
# print(Integrate(omega(gfetas, gfw), mesh))



concentrations = model.get_concentrations()
for comp, conc in concentrations.items():
    Draw(1e-4 * conc, mesh, f"{comp.name}")

#print("conc-fay = ", concentrations[fayalite])

potentials = model.gfw
Draw(potentials, mesh, "potential")

import numpy as np

x_vals = np.linspace(-0.5e-3, 0.5e-3, 100)
y_vals = 0.5e-4
mesh_pnts = mesh(x_vals, y_vals)

funcs_to_plot = { "potential" : potentials,
                 "fosterite" : concentrations[fosterite],
                   "liquid" : model.get_phase(liquid),
                   "solid" : model.get_phase(solid),
#                   "eta1" : gfetas[0],
#                   "eta2" : gfetas[1],
                   "fayalite" : concentrations[fayalite]}
                 
vals = { name : [func(mesh_pnts)] for name, func in funcs_to_plot.items() }
time = 0
time_vals = [time]

import matplotlib.pyplot as plt
interface_point = (0.4e-3, 0)
# vol = Integrate(1, mesh) 

counter = 0
def callback():
    global counter
    counter += 1
    if counter % 10 == 0:
        time_vals.append(time)
        for name, func in funcs_to_plot.items():
            vals[name].append(func(mesh_pnts))
            plt.clf()
            for t,c in zip(time_vals, vals[name]):
                plt.plot(x_vals, c, label=f"t {t}")
            # plt.legend()
            plt.savefig(f"{name}.png")
         
    
    print(f"Time: {model.time}, energy: {ngs.Integrate(model.get_energy(), mesh)}, e_mw = {ngs.Integrate(model.get_multiwell_energy(), mesh)}, e_grad = {ngs.Integrate(model.get_gradient_energy(), mesh)}, e_chem = {ngs.Integrate(model.get_chemical_energy(), mesh)}")
    ngs.Redraw()

# model.print_newton = True
with ngs.TaskManager():
    model.set_timestep(0.1)
    model.do_timestep()
    model.set_timestep(1)
    model.solve(100, callback=callback)
