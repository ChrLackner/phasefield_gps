import ngsolve as ngs
from ngsolve.meshes import *
from phasefield_gps import *

factor = 1e4
directional_surface_energies = { (1,0) : 2.45 * factor,
                                 (0,1) : 0.5 * 2.45 * factor,
                                 (1,1) : 0.83 * 2.45 * factor,
                                 (-1,1) : 0.83 * 2.45 * factor }
print("directional_surface_energies", directional_surface_energies)
liquid = Phase("liquid", diffusion_coefficient=3e-12)
solid = Phase("solid", diffusion_coefficient=3e-16,
              surface_energies= { liquid: directional_surface_energies,
                                  "kappa": 0.2 })

fosterite = IdealSolutionComponent(name="Fosterite",
                                   phase_energies={ liquid: -2498560,
                                                    solid: -2509411 })
fayalite = IdealSolutionComponent(name="Fayalite",
                                  phase_energies={ liquid: -1980885,
                                                   solid: -1951843 })
# mesh
nx = 50
ny = 50
mesh = MakeStructured2DMesh(quads=True, nx=nx, ny=ny,
                            periodic_x=False, periodic_y=False,
                            mapping = lambda x,y: (1e-3*x-5e-4, 1e-3*y - 5e-4))

model = GrandPotentialSolver(mesh=mesh,
                             components=[fosterite, fayalite],
                             phases=[liquid, solid],
                             molar_volume=4.3e-5, # m**3/mol
                             interface_mobility=1e-13, # m**4/J/s
                             interface_energy=2.45, # J/m**2
                             temperature=1923.15, # K
                             interface_width=5e-5) # m
model.order=2

model.mass_conservation = False

# initial conditions for phase
# wenn ...(x+0).., dann genau Mitte (-0.5e-3..0..+0.5e-3). x+0.4e-3.. = verschiebt IF nach links

ic_concentrations = { fosterite: { liquid: 0.73,
                                  solid: 0.9 }}
ic_concentrations[fayalite] = { liquid: 1-ic_concentrations[fosterite][liquid],
                                solid: 1-ic_concentrations[fosterite][solid] }

r = 10 * 1e-5
rx = ngs.sqrt(ngs.x**2 + ngs.y**2) - r
tanh = lambda x: ngs.sinh(x)/ngs.cosh(x)
ic_liquid = 0.5 * (1 + tanh(ngs.sqrt(model.m(False)/model.kappa(False)) * rx))
# ic_liquid = 0.5 * (1 + tanh(ngs.sqrt(model.m()/model.kappa()) * ngs.x))

Draw(ic_liquid, mesh, "ic_liquid")
model.set_initial_conditions({ liquid: ic_liquid,
                               solid: 1-ic_liquid },
                             components=ic_concentrations)

Draw(1e-4 * model.get_phase(liquid), mesh, "liquid")
Draw(1e-4 * model.get_phase(solid), mesh, "solid")
# print(Integrate(omega(gfetas, gfw), mesh))
concentrations = model.get_concentrations()
for comp, conc in concentrations.items():
    Draw(1e-4 * conc, mesh, f"{comp.name}")

import numpy as np

x_vals = np.linspace(-0.5e-3, 0.5e-3, 100)
y_vals = 0.5e-4
mesh_pnts = mesh(x_vals, y_vals)

# funcs_to_plot = { "concentration" : gfc,
#                   "liquid" : liquid,
#                   "solid" : solid,
#                   "eta1" : gfetas[0],
#                   "eta2" : gfetas[1],
#                   "potential" : gfw }
                  

# vals = { name : [func(mesh_pnts)] for name, func in funcs_to_plot.items() }
# time = 0
# time_vals = [time]

# import matplotlib.pyplot as plt
# interface_point = (0.4e-3, 0)
timestep = 0
def callback():
    global timestep
    timestep += 1
    if timestep == 10:
        model.set_timestep(0.1)
    print(f"Time: {model.time}, energy: {ngs.Integrate(model.get_energy(), mesh)}, e_c = {ngs.Integrate(model.get_chemical_energy(), mesh)}, e_mw = {ngs.Integrate(model.get_multiwell_energy(),mesh)}, e_grad = {ngs.Integrate(model.get_gradient_energy(), mesh)}")
    ngs.Redraw()

model.print_newton = True
model.set_timestep(0.001)
print("do initial timestep")
model.do_timestep()
callback()
model.set_timestep(0.01)
# model.print_newton = True
with ngs.TaskManager(10**9):
    # model.do_timestep()
    model.solve(5, callback=callback)
    print("done")
print("finish")


