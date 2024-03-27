import ngsolve as ngs
from ngsolve.meshes import *
from phasefield_gps import *

liquid = Phase("liquid", diffusion_coefficient=3e-7,
               site_variable=1.7892)    #3e-7
solid = Phase("solid", diffusion_coefficient=3e-12,     #3e-12
              surface_energies={ liquid : 2.451e4 },
              site_variable=2) # J/m**2

fosterite_liquid = { 1373.15: -2280010.,
                     1773.15: -2434149.,
                     2173.15: -2612808.,
                     3273.15: -3198056. }

fosterite_solid = {1373.15: -2307279.,
                   1773.15: -2450896.,
                   2173.15: -2611896.,
                   3273.15: -3123702. }


fosterite = IdealSolutionComponent(name="Fosterite",
                                   phase_energies={ liquid: fosterite_liquid,
                                                    solid: fosterite_solid })

fayalite_liquid = {1373.15: -1699537.,
                   1773.15: -1899806.,
                   2173.15: -2122255.,
                   3273.15: -2818967. }

fayalite_solid = { 1373.15: -1706086.,
                   1773.15: -1881116.,
                   2173.15: -2075201.,
                   3273.15: -2686725. }

fayalite = IdealSolutionComponent(name="Fayalite",
                                    phase_energies={ liquid: fayalite_liquid,
                                                         solid: -1881116 })
nx = 100
ny = 10
mesh = MakeStructured2DMesh(quads=True, nx=nx, ny=ny,
                            periodic_x=False, periodic_y=False,
                            mapping = lambda x,y: (1e-3*x-0.5e-3, 0.1e-3 * y))

model = GrandPotentialSolver(mesh=mesh,
                             components=[fosterite, fayalite],
                             phases=[liquid, solid],
                             molar_volume=4.3e-5, # m**3/mol
                             interface_mobility=1e-13,  # m**4/J/s
                             temperature=1500, # 2173.15, # K
                             interface_width=5e-5) # m

model.mass_conservation = True

# initial conditions for phase
# wenn ...(x+0).., dann genau Mitte (-0.5e-3..0..+0.5e-3). x+0.4e-3.. = verschiebt IF nach links
tanh = lambda x: ngs.sinh(x)/ngs.cosh(x)
# shift = -0.45e-3
shift = 0 #0.4e-3

# ic_concentrations = { fayalite: { liquid: 0.999,
#                                    solid: 0.98 }}
# ic_concentrations = { fayalite: { liquid: 0.4, solid: 0.6 }}
ic_concentrations = { fayalite: { liquid: 0.55, solid: 0.55 }}

ic_concentrations[fosterite] = { liquid: 1-ic_concentrations[fayalite][liquid],
                                 solid: 1-ic_concentrations[fayalite][solid] }

print(model.m(False)/model.kappa(False))
ic_liquid = 0.5 * tanh(ngs.sqrt(model.m(False)/model.kappa(False)) * (ngs.x+shift)) + 0.5
model.set_initial_conditions({ liquid: ic_liquid,
                               solid: 1-ic_liquid },
                             components=ic_concentrations)

fig = model.plot_energy_landscape()
fig.savefig("energy_landscape_initial.png")
Draw(model.get_phase(liquid), mesh, "liquid")
Draw(model.get_phase(solid), mesh, "solid")
# print(Integrate(omega(gfetas, gfw), mesh))



concentrations = model.get_concentrations()
for comp, conc in concentrations.items():
    Draw(conc, mesh, f"{comp.name}")

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
                   "fayalite" : concentrations[fayalite],
                   "energy" : model.get_energy(),
                   "chem_energy" : model.get_chemical_energy(),
                   "grad_energy" : model.get_gradient_energy(),
                   "mw_energy" : model.get_multiwell_energy() }
                 
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
    if counter % 3 == 0:
        time_vals.append(time)
        for name, func in funcs_to_plot.items():
            vals[name].append(func(mesh_pnts))
            plt.clf()
            for t,c in zip(time_vals, vals[name]):
                plt.plot(x_vals, c, label=f"t {t}")
            # plt.legend()
            plt.savefig(f"{name}.png")

    # the first time the solve passes 0.01 sec we change the temperature
    if model.time > 0.01 and model.T.Get() < 2000:
        model.set_Temperature(1900)
        fig = model.plot_energy_landscape()
        fig.savefig(f"energy_landscape_{model.T.Get()}.png")
        time_vals.append(str(time) + "_reset")
        for name, func in funcs_to_plot.items():
            vals[name].append(func(mesh_pnts))
            plt.clf()
            for t,c in zip(time_vals, vals[name]):
                plt.plot(x_vals, c, label=f"t {t}")
            # plt.legend()
            plt.savefig(f"{name}.png")
    draw_time_curves()
    # input()
         
    
    print(f"Time: {model.time}, energy: {ngs.Integrate(model.get_energy(), mesh)}, e_mw = {ngs.Integrate(model.get_multiwell_energy(), mesh)}, e_grad = {ngs.Integrate(model.get_gradient_energy(), mesh)}, e_chem = {ngs.Integrate(model.get_chemical_energy(), mesh)}")
    ngs.Redraw(blocking=True)

ngs.SetVisualization(deformation=True)
from ngsolve.internal import *
visoptions.scaledeform1 = 1e-14

energy_over_time = []
concentrations_over_time = [[], []]
all_times = []

def draw_time_curves():
    all_times.append(model.time)
    energy_over_time.append(ngs.Integrate(model.get_energy(), mesh))
    concentrations_over_time[0].append(ngs.Integrate(concentrations[fosterite], mesh))
    concentrations_over_time[1].append(ngs.Integrate(concentrations[fayalite], mesh))
    plt.clf()
    plt.plot(all_times, energy_over_time)
    plt.savefig("total_energy.png")
    plt.clf()
    plt.plot(all_times, concentrations_over_time[0], label="Fosterite")
    plt.plot(all_times, concentrations_over_time[1], label="Fayalite")
    plt.savefig("concentrations.png")

def numdiff_omega():
    domega_deta = ngs.GridFunction(model.fes.components[2].components[0])
    domega_deta.vec[:] = 0
    eta = model.gfetas.components[0]
    for i in range(len(eta.vec)):
        eta.vec[i] += 1e-8
        domega_deta.vec[i] = ngs.Integrate(model.get_energy(), mesh)
        eta.vec[i] -= 2e-8
        domega_deta.vec[i] -= ngs.Integrate(model.get_energy(), mesh)
        eta.vec[i] += 1e-8
        domega_deta.vec[i] /= 2e-8
    return domega_deta
            
        

Draw(model.get_gradient_energy(), mesh, "grad_energy")
Draw(model.get_chemical_energy(), mesh, "chem_energy")
Draw(model.get_multiwell_energy(), mesh, "mw_energy")
Draw(model.get_energy(), mesh, "energy")
Draw(-model.L * model.get_energy().Diff(model.gfetas), mesh, "energy_diff")
Draw(liquid.get_chemical_potential(model.components, model.gfw, model.T),
     mesh, "chem_potential_liquid")
Draw(solid.get_chemical_potential(model.components, model.gfw, model.T),
     mesh, "chem_potential_solid")
# model.print_newton = True
with ngs.TaskManager():
    # domega = numdiff_omega()
    # Draw(domega, mesh, "domega_numdif")
    # input("wait")
    # draw_time_curves()
    model.set_timestep(0.002)
    # model.do_timestep()
    # callback()
    # input("wait")
    # model.set_timestep(0.1)
    model.solve(1e9, callback=callback)
