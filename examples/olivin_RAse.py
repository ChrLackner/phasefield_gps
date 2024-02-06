import ngsolve as ngs
from ngsolve.meshes import *
from phasefield_gps import *

liquid = Phase("liquid", diffusion_coefficient=3e-7)
solid = Phase("solid", diffusion_coefficient=3e-12)

fosterite = IdealSolutionComponent(name="Fosterite",
                                   phase_energies={ liquid: -2498560,
                                                    solid: -2509411 })
fayalite = IdealSolutionComponent(name="Fayalite",
                                  phase_energies={ liquid: -1980885,
                                                   solid: -1951843 })
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
                             interface_energy=2.45, # J/m**2
                             temperature=1923.15, # K
                             interface_width=5e-5) # m

# model.set_interface_anisotropy(theta0=0, epsilon=0.2)

model.mass_conservation = False

# initial conditions for phase
# wenn ...(x+0).., dann genau Mitte (-0.5e-3..0..+0.5e-3). x+0.4e-3.. = verschiebt IF nach links
tanh = lambda x: ngs.sinh(x)/ngs.cosh(x)
# shift = -0.45e-3
shift = 0
ic_concentrations = { fayalite: { liquid: 0.999,
                                    solid: 0.98 }}
# ic_concentrations = { fayalite: { liquid: 0.5,
                                  # solid: 0.5 }}
ic_concentrations[fosterite] = { liquid: 1-ic_concentrations[fayalite][liquid],
                                 solid: 1-ic_concentrations[fayalite][solid] }

print(model.m()/model.kappa())
ic_liquid = 0.5 * tanh(ngs.sqrt(model.m()/model.kappa()) * (ngs.x-shift)) + 0.5
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

def callback():
    
    print(f"Time: {model.time}, energy: {ngs.Integrate(model.get_energy(), mesh)}")
    ngs.Redraw()

# model.print_newton = True
with ngs.TaskManager():
    model.set_timestep(0.1)
    model.do_timestep()
    model.set_timestep(1)
    model.solve(100, callback=callback)


