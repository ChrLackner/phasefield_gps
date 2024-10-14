from typing import Callable
import ngsolve as ngs

def argmax2(x,f):
    return ngs.IfPos(ngs.Norm(x[1])-1e-6-ngs.Norm(x[0]),
                     ngs.CF((x[1],f[1])),
                     ngs.CF((x[0],f[0])))

def argmax(x,f):
    res = argmax2(x[0:2], f[0:2])
    for i in range(2, len(x)):
        res = argmax2(ngs.CF((res[0], x[i])), ngs.CF((res[1], f[i])))
    return res

# Gas constant
R = 8.31446261815324 # J/mol/K

class Phase:
    def __init__(self, name: str, diffusion_coefficient: float | Callable,
                 surface_energies=None, site_variable=1.):
        self.name = name
        self.D = diffusion_coefficient
        self.surface_energies = surface_energies
        self.site_variable = site_variable

    def get_surface_energy(self, other_phase, interface_normal) -> float | ngs.CF:
        assert self.surface_energies is not None, "Surface energies not defined!"
        if isinstance(self.surface_energies, dict):
            phase_se = self.surface_energies[other_phase]
            if isinstance(phase_se, dict):
                if interface_normal is None:
                    raise Exception("Interface normal must be given for multi-component systems!")
                if interface_normal == False:
                    return list(phase_se.values())[0]
                nif = ngs.Norm(interface_normal)
                if_normal = ngs.IfPos(nif, 1/( nif * (1+1e-6)) * interface_normal, ngs.CF((1,0) if interface_normal.dim == 2 else (1,0,0)))
                ips = [ngs.InnerProduct(if_normal, 1/ngs.Norm(ngs.CF(direction)) * ngs.CF(direction)) for direction in phase_se.keys()]
                energies = list(phase_se.values())
                amax = argmax(ips, energies)
                theta = ngs.acos(amax[0])
                return ngs.IfPos(nif-100, amax[1] * ngs.sqrt(ngs.sin(theta)**2 + self.surface_energies["kappa"]**2 * ngs.cos(theta)**2), energies[0])
            return phase_se
        return self.surface_energies

    def get_concentrations(self, components, potentials, T):
        assert len(components) < 3, "TODO: implement for more than two components"
        concentrations = {}
        sum_concentrations = 0
        for i,component in enumerate(components[:-1]):
            assert isinstance(component, type(components[-1])), \
                "All components must be of same type!"
            solvent_energy = components[-1].phase_energies[self]
            my_energy = component.phase_energies[self]
            eps = my_energy - solvent_energy
            alpha = ngs.exp((potentials[i] - eps)/(self.site_variable*R*T))
            concentrations[component] = alpha/(1+alpha)
            sum_concentrations += concentrations[component]
        concentrations[components[-1]] = (1-sum_concentrations)
        return concentrations

    def get_potentials(self, concentrations, solvent, T):
        assert len(concentrations) == 2, "TODO: implement for more than two components"
        potentials = {}
        for comp, conc in concentrations.items():
            if comp == solvent:
                continue
            solvent_energy = solvent.phase_energies[self]
            my_energy = comp.phase_energies[self]
            eps = my_energy - solvent_energy
            potentials[comp] = eps + self.site_variable * R * T * ngs.IfPos(conc, ngs.IfPos(1-conc, ngs.log(conc/(1-conc)), 0), 0)
        return potentials

    def get_chemical_energy_from_potential(self, components, potentials, T):
        concentrations = self.get_concentrations(components, potentials, T)
        return self.get_chemical_energy(concentrations, T)

    def get_chemical_energy(self, concentrations, T, use_ifpos=True):
        omega = 0
        for component, concentration in concentrations.items():
            omega += component.phase_energies[self] * concentration
        if use_ifpos:
            omega += self.site_variable * R * T * sum([c * ngs.IfPos(c, ngs.log(c), 0) for c in concentrations.values()])
        else:
            omega += self.site_variable * R * T * sum([c * ngs.log(c) for c in concentrations.values()])
        return omega

    def get_chi(self, components, potentials, T):
        concentrations = self.get_concentrations(components, potentials, T)
        assert len(components) == 2, "TODO: implement for more than two components"
        c = concentrations[components[0]]
        c = ngs.IfPos(c-1e-9, c, 1e-9)
        return 1/(self.site_variable * R * T) * c * (1-c)

    def get_D(self, components, potentials, T):
        concentrations = self.get_concentrations(components, potentials, T)
        assert len(components) == 2, "TODO: implement for more than two components"
        c = concentrations[components[0]]
        c = ngs.IfPos(c-1e-9, c, 1e-9)
        if callable(self.D) and not isinstance(self.D, ngs.CF):
            D = self.D(c)
        else:
            D = self.D
        return D

    def get_D_times_chi(self, components, potentials, T):
        concentrations = self.get_concentrations(components, potentials, T)
        assert len(components) == 2, "TODO: implement for more than two components"
        c = concentrations[components[0]]
        c = ngs.IfPos(c-1e-9, c, 1e-9)
        if callable(self.D) and not isinstance(self.D, ngs.CF):
            D = self.D(c)
        else:
            D = self.D
        return D * 1/(self.site_variable * R * T) * c * (1-c)
