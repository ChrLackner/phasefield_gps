import ngsolve as ngs

# Gas constant
R = 8.314 # J/mol/K

class Phase:
    def __init__(self, name: str, diffusion_coefficient: float):
        self.names = name
        self.D = diffusion_coefficient

    def get_concentrations(self, components, potentials, T):
        assert len(components) < 3, "TODO: implement for more than two components"
        concentrations = {}
        for i,component in enumerate(components[:-1]):
            assert isinstance(component, type(components[-1])), \
                "All components must be of same type!"
            solvent_energy = components[-1].phase_energies[self]
            my_energy = component.phase_energies[self]
            eps = my_energy - solvent_energy
            alpha = ngs.exp((potentials[i] - eps)/(R*T))
            concentrations[component] = alpha/(1+alpha)
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
            potentials[comp] = eps + R * T * ngs.log(conc/(1-conc))
        return potentials

    def get_chemical_potential(self, components, potentials, T):
        concentrations = self.get_concentrations(components, potentials, T)
        omega = 0
        sum_concentrations = 0
        for component, concentration in concentrations.items():
            omega += component.phase_energies[self] * concentration
            sum_concentrations += concentration
        omega += components[-1].phase_energies[self] * (1-sum_concentrations)
        return omega

    def get_chi(self, components, potentials, T):
        concentrations = self.get_concentrations(components, potentials, T)
        assert len(components) == 2, "TODO: implement for more than two components"
        return concentrations[components[0]] * (1-concentrations[components[0]])
