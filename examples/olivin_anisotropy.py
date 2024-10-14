import ngsolve as ngs
from ngsolve.meshes import *
from phasefield_gps import *
import numpy as np
import random

dim = 2
factor = 1e4
random.seed(1)
v1s = []
v2s = []
if dim == 3:
    factor = 1
    r1 = random.random() * 2 * pi
    r1 = 0
    r2 = random.random() * 2 * pi
    r2 = 30*pi/180
    mat = np.array([[cos(r1), -sin(r1), 0], [sin(r1), cos(r1), 0],
                    [0, 0, 1]])
    v1 = np.array([0,0,1])
    v2 = np.array([1,0,0])
    v3 = np.array([0,1,0])
    v4 = np.array([1,0,1])
    v5 = np.array([0,2,1])
    v6 = np.array([-1,0,1])
    v7 = np.array([0,-2,1])

    directional_surface_energies = { tuple(np.dot(mat, v1)) : 2.45 * factor,
                                     tuple(np.dot(mat, v2)) : 0.5 * 2.45 * factor,
                                     tuple(np.dot(mat, v3)) : 1.67 * 2.45 * factor,
                                     tuple(np.dot(mat, v4)) : 2.33 * 2.45 * factor,
                                     tuple(np.dot(mat, v5)) : 2.26 * 2.45 * factor,
                                     tuple(np.dot(mat, v6)) : 2.33 * 2.45 * factor,
                                     tuple(np.dot(mat, v7)) : 2.26 * 2.45 * factor
                                    }
    mat2 = np.array([[cos(r2), -sin(r2), 0], [sin(r2), cos(r2), 0],
                        [0, 0, 1]])
    directional_surface_energies2 = { tuple(np.dot(mat2, v1)) : 2.45 * factor,
                                      tuple(np.dot(mat2, v2)) : 0.5 * 2.45 * factor,
                                      tuple(np.dot(mat2, v3)) : 1.67 * 2.45 * factor,
                                      tuple(np.dot(mat2, v4)) : 2.33 * 2.45 * factor,
                                      tuple(np.dot(mat2, v5)) : 2.26 * 2.45 * factor,
                                      tuple(np.dot(mat2, v6)) : 2.33 * 2.45 * factor,
                                      tuple(np.dot(mat2, v7)) : 2.26 * 2.45 * factor
                                    }

# directional_surface_energies = { (0,0,1) : 2.45 * factor,
    #                                  (1,0,0) : 0.5 * 2.45 * factor,
    #                                  (0,1,0) : 1.67 * 2.45 * factor,
    #                                  (1,0,1) : 2.33 * 2.45 * factor,
    #                                  (0,2,1) : 2.26 * 2.45 * factor,
    #                                  (-1,0,1) : 2.33 * 2.45 * factor,
    #                                  (0,-2,1) : 2.26 * 2.45 * factor }
else:
    r1 = random.random() * 2 * pi
    r1 = 0
    mat = np.array([[cos(r1), -sin(r1)], [sin(r1), cos(r1)]])
    v1 = np.array([1, 0])
    v2 = np.array([0, 1])
    v3 = np.array([1, 1])
    v4 = np.array([-1, 1])
    v1s.append(ngs.CF(tuple(np.dot(mat, v1))))
    v2s.append(ngs.CF(tuple(np.dot(mat, v2))))

    directional_surface_energies = { tuple(np.dot(mat, v1)) : 2.45 * factor,
                                     tuple(np.dot(mat, v2)) : 0.5 * 2.45 * factor,
                                     tuple(np.dot(mat, v3)) : 0.83 * 2.45 * factor,
                                     tuple(np.dot(mat, v4)) : 0.83 * 2.45 * factor }
    r2 = random.random() * 2 * pi
    r2 = 30*pi/180
    mat2 = np.array([[cos(r2), -sin(r2)], [sin(r2), cos(r2)]])
    v1s.append(ngs.CF(tuple(np.dot(mat2, v1))))
    v2s.append(ngs.CF(tuple(np.dot(mat2, v2))))
    directional_surface_energies2 = { tuple(np.dot(mat2, v1)) : 2.45 * factor,
                                      tuple(np.dot(mat2, v2)) : 0.5 * 2.45 * factor,
                                      tuple(np.dot(mat2, v3)) : 0.83 * 2.45 * factor,
                                      tuple(np.dot(mat2, v4)) : 0.83 * 2.45 * factor }
    print("directional_surface_energies", directional_surface_energies)
    print("directional_surface_energies2", directional_surface_energies2)
    # directional_surface_energies = 2.45 * factor
    # directional_surface_energies2 = 2.45 * factor

initial_temperature = 1523.15 # 2173.15, # K
end_temperature = 1373.15 # 2173.15, # K
cooling_rate_slow = 0.01/3600
# cooling_rate_fast = 10/3600
cooling_rate_fast = 1

fosterite_liquid = "foL_1373-2273-10.tab"
fosterite_solid = "fo_1373-2273-10.tab"
fayalite_liquid = "faL_1373-2273-10.tab"
fayalite_solid = "fa_1373-2273-10.tab"

solid1_diff = 3e-12 * ngs.OuterProduct(v1s[0], v1s[0]) + 6 * 3e-12 * ngs.OuterProduct(v2s[0], v2s[0])
solid2_diff = 3e-12 * ngs.OuterProduct(v1s[1], v1s[1]) + 6 * 3e-12 * ngs.OuterProduct(v2s[1], v2s[1])

print("directional_surface_energies", directional_surface_energies)
liquid = Phase("liquid", diffusion_coefficient=3e-7)
solid = Phase("solid", diffusion_coefficient=solid1_diff,
              surface_energies= { liquid: directional_surface_energies,
                                  "kappa": 0.2 })
solid2 = Phase("solid", diffusion_coefficient=solid2_diff,
               surface_energies= { liquid: directional_surface_energies2,
                                   solid: 1e3,
                                   "kappa": 0.2 })

fosterite = IdealSolutionComponent(name="Fosterite",
                                   phase_energies={ liquid: fosterite_liquid,
                                                    solid: fosterite_solid,
                                                    solid2: fosterite_solid })
fayalite = IdealSolutionComponent(name="Fayalite",
                                  phase_energies={ liquid: fayalite_liquid,
                                                   solid: fayalite_solid,
                                                   solid2: fayalite_solid})

# mesh
nel = 50
if dim == 3:
    nel = 30
nx = nel
ny = nel


if dim == 3:
    nz = nel
    mesh = MakeStructured3DMesh(hexes=True, nx=nx, ny=ny, nz=nz,
                                mapping= lambda x,y,z: (1e-3 * x-5e-4,
                                                        1e-3 * y-5e-4,
                                                        1e-3 * z-5e-4))
else:
    mesh = MakeStructured2DMesh(quads=True, nx=nx, ny=ny,
                                periodic_x=False, periodic_y=False,
                                mapping = lambda x,y: (1e-3*x-5e-4, 1e-3*y - 5e-4))

model = GrandPotentialSolver(mesh=mesh,
                             components=[fosterite, fayalite],
                             phases=[liquid, solid, solid2],
                             molar_volume=4.3e-5, # m**3/mol
                             interface_mobility=1e-13, # m**4/J/s
                             temperature=initial_temperature, # K
                             interface_width=5e-5) # m
model.order=2

model.mass_conservation = True

# initial conditions for phase
# wenn ...(x+0).., dann genau Mitte (-0.5e-3..0..+0.5e-3). x+0.4e-3.. = verschiebt IF nach links

ic_concentrations = { fosterite: { liquid: 0.5,
                                   solid: 0.5,
                                   solid2: 0.5 }}
ic_concentrations[fayalite] = { liquid: 1-ic_concentrations[fosterite][liquid],
                                solid: 1-ic_concentrations[fosterite][solid],
                                solid2: 1-ic_concentrations[fosterite][solid2] }

r = 5 * 1e-5
if dim == 3:
    rx = ngs.sqrt((ngs.x+0.15e-3)**2 + ngs.y**2 + ngs.z**2) - r
    rx2 = ngs.sqrt((ngs.x-0.15e-3)**2 + ngs.y**2 + ngs.z**2) - r
else:
    rx = ngs.sqrt((ngs.x+0.15e-3)**2 + ngs.y**2) - r
    rx2 = ngs.sqrt((ngs.x-0.15e-3)**2 + ngs.y**2) - r
tanh = lambda x: ngs.sinh(x)/ngs.cosh(x)
ic_solid = 1-(0.5 * (1 + tanh(ngs.sqrt(model.m(False)/model.kappa(False)) * rx)))
# ic_liquid = 0.5 * (1 + tanh(ngs.sqrt(model.m()/model.kappa()) * ngs.x))
ic_solid2 = 1-(0.5 * (1 + tanh(ngs.sqrt(model.m(False)/model.kappa(False)) * rx2)))
ic_liquid = 1-(ic_solid + ic_solid2)
model.set_initial_conditions({ liquid: ic_liquid,
                               solid: ic_solid,
                               solid2: ic_solid2 },
                             components=ic_concentrations)
Draw(ic_liquid, mesh, "ic_liquid")

# print(Integrate(omega(gfetas, gfw), mesh))
concentrations = model.get_concentrations()
for comp, conc in concentrations.items():
    Draw(conc, mesh, f"{comp.name}")
Draw(model.get_phase(solid), mesh, "solid")
Draw(model.get_phase(solid2), mesh, "solid2")
Draw(model.get_phase(liquid), mesh, "liquid")



# x_vals = np.linspace(-0.5e-3, 0.5e-3, 100)
# y_vals = 0.5e-4
# mesh_pnts = mesh(x_vals, y_vals)

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
def cooling(time):
    temp = initial_temperature - (cooling_rate_fast * time)
    if temp < end_temperature:
        temp = end_temperature
    print("Temperature path: ", temp)
    return temp

timestep = 0
def callback():
    global timestep
    timestep += 1
    if timestep % 10 == 0:
        model.set_timestep(model.dt.Get()*2)
        print("set timestep to ", model.dt.Get())
    temperature = cooling(model.time)
    print("set temperature to ", temperature)
    model.set_Temperature(temperature)
    solid_part = ngs.Integrate(model.get_phase(solid), mesh) / ngs.Integrate(1, mesh)
    if solid_part < 1e-3:
        # create new grains
        model.set_initial_conditions({ liquid: ic_liquid,
                                       solid: ic_solid,
                                       solid2: ic_solid2 },
                                     components=ic_concentrations)
    print(f"Time: {model.time}, energy: {ngs.Integrate(model.get_energy(), mesh)}, e_c = {ngs.Integrate(model.get_chemical_energy(), mesh)}, e_mw = {ngs.Integrate(model.get_multiwell_energy(),mesh)}, e_grad = {ngs.Integrate(model.get_gradient_energy(), mesh)}")
    print("Solid part = ", solid_part)
    ngs.Redraw()

model.print_newton = True
# model.set_timestep(0.01)
model.set_timestep(0.01)
print("do initial timestep")
model.do_timestep()
callback()
model.set_timestep(0.1)
# model.print_newton = True
with ngs.TaskManager(10**9):
    # model.do_timestep()
    model.solve(500, callback=callback)
    print("done")
print("finish")


