
import ngsolve as ngs
from phasefield_gps import *
import ngsolve.meshes as ngmeshes
import random
import numpy as np

# this selects the seed for the random number generator
random.seed(1)

initial_temperature = 1700 # K
end_temperature = 1373.15 # K
cooling_rate = 0.1/3600 # K/s

conc_liquid = 0.5
conc_solid1 = 0.4

# mesh
nel = 50
nx = nel
ny = nel
mesh = ngmeshes.MakeStructured2DMesh(
    quads=True, nx=nx, ny=ny,
    periodic_x=False, periodic_y=False,
    mapping = lambda x,y: (1e-3*x-5e-4, 1e-3*y - 5e-4))

r1 = random.random() * 2 * ngs.pi
# this fixes angle to 0:
# r1 = 0
mat = np.array([[np.cos(r1), -np.sin(r1)], [np.sin(r1), np.cos(r1)]])
factor = 1e4
v1s = []
v2s = []

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

r2 = random.random() * 2 * np.pi
# this fixes angle to 30 degrees
# r2 = 30*np.pi/180
mat2 = np.array([[np.cos(r2), -np.sin(r2)], [np.sin(r2), np.cos(r2)]])
v1s.append(ngs.CF(tuple(np.dot(mat2, v1))))
v2s.append(ngs.CF(tuple(np.dot(mat2, v2))))
directional_surface_energies2 = { tuple(np.dot(mat2, v1)) : 2.45 * factor,
                                  tuple(np.dot(mat2, v2)) : 0.5 * 2.45 * factor,
                                  tuple(np.dot(mat2, v3)) : 0.83 * 2.45 * factor,
                                  tuple(np.dot(mat2, v4)) : 0.83 * 2.45 * factor }

D0_sol = 10**(-8.27)
Ea = 226000
P = 1e5
V = 7e-6
Feterm = lambda concentration: 3*((1-concentration) - 0.14)

temp = ngs.Parameter(initial_temperature)
R = 8.31446261815324 # J K^-1 mol^-1
diff_coef_solid1 = lambda concentration: D0_sol * ngs.exp(-(Ea+(P-1e5)*V)/(R*temp)) *10**Feterm(concentration) * (ngs.OuterProduct(v1s[0], v1s[0]) + 6 * 3e-12 * ngs.OuterProduct(v2s[0], v2s[0]))
diff_coef_solid2 = lambda concentration: D0_sol * ngs.exp(-(Ea+(P-1e5)*V)/(R*temp)) *10**Feterm(concentration) * (ngs.OuterProduct(v1s[1], v1s[1]) + 6 * 3e-12 * ngs.OuterProduct(v2s[1], v2s[1]))
if_mobility = 1e-13 * ngs.exp(0 * temp)

liquid = Phase("liquid", diffusion_coefficient=3e-7)
solid = Phase("solid", diffusion_coefficient=diff_coef_solid1,
              surface_energies= { liquid: directional_surface_energies,
                                  "kappa": 0.2 })
solid2 = Phase("solid", diffusion_coefficient=diff_coef_solid2,
               surface_energies= { liquid: directional_surface_energies2,
                                   solid: 1e3,
                                   "kappa": 0.2 })

fosterite_liquid = "foL_1373-2273-10.tab"
fosterite_solid = "fo_1373-2273-10.tab"
fayalite_liquid = "faL_1373-2273-10.tab"
fayalite_solid = "fa_1373-2273-10.tab"

fosterite = IdealSolutionComponent(name="Fosterite",
                                   phase_energies={ liquid: fosterite_liquid,
                                                    solid: fosterite_solid,
                                                    solid2: fosterite_solid })
fayalite = IdealSolutionComponent(name="Fayalite",
                                  phase_energies={ liquid: fayalite_liquid,
                                                   solid: fayalite_solid,
                                                   solid2: fayalite_solid})

model = GrandPotentialSolver(mesh=mesh,
                             components=[fosterite, fayalite],
                             phases=[liquid, solid, solid2],
                             molar_volume=4.3e-5, # m**3/mol
                             interface_mobility=1e-13, # m**4/J/s
                             temperature=temp, # K
                             interface_width=5e-5) # m


r = 5 * 1e-5
rx = ngs.sqrt((ngs.x+0.15e-3)**2 + ngs.y**2) - r
rx2 = ngs.sqrt((ngs.x-0.15e-3)**2 + ngs.y**2) - r


# compute energy diagrams for full liquid concentation
# and from there calculate the concentration a new solid grain would be
# created with (having parallel energy tangents)
model.set_initial_conditions({ liquid: ngs.CF(1),
                               solid: ngs.CF(0),
                               solid2: ngs.CF(0) },
                             { fosterite: { liquid: conc_liquid,
                                            solid: 0,
                                            solid2: 0 },
                               fayalite: { liquid: 1 - conc_liquid,
                                           solid: 1,
                                           solid2: 1}})
conc_solid2 = solid2.calculate_parallel_energy_concentration(conc_liquid,
                                                             liquid,
                                                             initial_temperature,
                                                             [fosterite, fayalite])
print("calculated concentration in solid2:", conc_solid2)
ic_concentrations : dict[Component, dict[Phase, ngs.CF |
                                         float]] = {
                                             fosterite: { liquid: conc_liquid,
                                                          solid: conc_solid1,
                                                          solid2: conc_solid2 }}
ic_concentrations[fayalite] = { liquid: 1-ic_concentrations[fosterite][liquid],
                                solid: 1-ic_concentrations[fosterite][solid],
                                solid2: 1-ic_concentrations[fosterite][solid2] }

tanh = lambda x: ngs.sinh(x)/ngs.cosh(x)
ic_solid = 1-(0.5 * (1 + tanh(ngs.sqrt(model.m(False)/model.kappa(False)) * rx)))
ic_solid2 = 1-(0.5 * (1 + tanh(ngs.sqrt(model.m(False)/model.kappa(False)) * rx2)))
ic_liquid = 1-(ic_solid + ic_solid2)

model.set_initial_conditions({ liquid: ic_liquid,
                               solid: ic_solid,
                               solid2: ic_solid2 },
                             components=ic_concentrations)

def cooling(time):
    temp = initial_temperature - (cooling_rate * time)
    if temp < end_temperature:
        temp = end_temperature
    print("Change temperature to:", temp)
    return temp

for comp, conc in model.get_concentrations().items():
    ngs.Draw(conc, mesh, f"{comp.name}")
ngs.Draw(model.get_phase(solid), mesh, "solid")
ngs.Draw(model.get_phase(solid2), mesh, "solid2")
ngs.Draw(model.get_phase(liquid), mesh, "liquid")

timestep = 0
def callback():
    global timestep
    timestep += 1
    if timestep % 10 == 0:
        model.set_timestep(model.dt.Get()*2)
        print("increase timestep to", model.dt.Get())
    model.set_Temperature(cooling(model.time))
    print(f"Time: {model.time}, energy: {ngs.Integrate(model.get_energy(), mesh)}")
    ngs.Redraw()


with ngs.TaskManager():
    model.set_timestep(0.01)
    model.do_timestep()
    callback()
    model.set_timestep(0.1)
    model.solve(500, callback=callback)

print("finish")
