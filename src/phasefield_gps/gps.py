
import ngsolve as ngs
from .phase import Phase
from .component import Component

class NewtonNotConverged(Exception):
    def __init__(self):
        super().__init__("Newton method did not converge!")

class GrandPotentialSolver:
    def __init__(self, mesh: ngs.Mesh, components: list[Component],
                 phases: list[Phase], molar_volume: float,
                 interface_mobility: float,
                 interface_energy: float,
                 temperature: float,
                 interface_width: float):
        """
See https://link.aps.org/doi/10.1103/PhysRevE.98.023309 for details.
        
Parameters
----------
mesh : ngs.Mesh
    The mesh on which to solve the model
components : list of Component
    The components of the system
Vm : float
    Molar volume
mu : float
    Interface mobility
sigma : float
    Interfacial energy
T : float
    Temperature
l : float
    Interfacial thickness
        """
        self.mesh = mesh
        self.components = components
        self.phases = phases
        self.phase_indices = { phase : i for i, phase in enumerate(self.phases) }
        self.component_indices = { component : i for i, component in enumerate(self.components) }
        self.dt = ngs.Parameter(0.01)
        self.order = 2
        self.mass_conservation = True
        self.gamma = ngs.Matrix(len(self.phase_indices), len(self.phase_indices))
        self.gamma[:] = 1.5
        for i in range(len(self.phase_indices)):
            self.gamma[i,i] = 0
        self.sigma = interface_energy
        self.T = temperature
        self.l = interface_width
        self.Vm = molar_volume
        self.mu = interface_mobility
        self.time = 0
        self.print_newton = False
        self.min_timestep = 1e-7

    @property
    def kappa(self) -> float:
        for i in range(len(self.phase_indices)):
            for j in range(i+1, len(self.phase_indices)):
                if self.gamma[i,j] != 1.5:
                    raise NotImplementedError("TODO: implement for gamma != 1.5, gamma = " + str(self.gamma))
        return 3/4 * self.sigma * self.l

    def set_gamma(self, phase1: Phase, phase2: Phase, gamma: float):
        self.gamma[self.phase_indices[phase1], self.phase_indices[phase2]] = gamma
        self.gamma[self.phase_indices[phase2], self.phase_indices[phase1]] = gamma

    @property
    def L(self) -> float:
        return 4/3 * self.mu/self.l

    def get_phase(self, phase) -> ngs.CoefficientFunction:
        return self.h(self.gfetas)[self.phase_indices[phase]]

    def h(self, etas) -> list[ngs.CoefficientFunction]:
        return [eta**2 / sum(etai**2 for etai in etas) for eta in etas]

    @property
    def m(self) -> float:
        return 6 * self.sigma/self.l

    def get_multiwell_energy(self, etas : list[ngs.CoefficientFunction]) \
            -> ngs.CoefficientFunction:
        return ngs.CF(self.m * (sum((eta**4/4-eta**2/2) for eta in etas) + sum(sum(self.gamma[i,j] * eta1**2 * eta2**2 for j,eta1 in enumerate(etas) if j > i) for i,eta2 in enumerate(etas)) + 0.25))

    def get_gradient_energy(self, etas: list[ngs.CoefficientFunction]) \
            -> ngs.CoefficientFunction:
        if isinstance(etas, tuple):
            return ngs.CF(self.kappa/2 * sum(ngs.InnerProduct(ngs.grad(eta), ngs.grad(eta)) for eta in etas))
        return self.kappa/2 * ngs.InnerProduct(ngs.grad(etas), ngs.grad(etas))

    def get_chemical_energy(self, etas: list[ngs.CoefficientFunction],
                            potentials: list[ngs.CoefficientFunction]) \
                            -> ngs.CoefficientFunction:
        omega_chem = ngs.CF(0)
        hs = self.h(etas)
        for phase, h in zip(self.phases, hs):
            omega = phase.get_chemical_potential(self.components, potentials, self.T)
            omega_chem += h * omega
        return 1/self.Vm * omega_chem

    def get_energy(self, etas: list[ngs.CoefficientFunction] | None = None,
                   potentials: list[ngs.CoefficientFunction] | None = None) \
                   -> ngs.CoefficientFunction:
        if etas is None:
            etas = self.gfetas
        if potentials is None:
            potentials = self.gfw
        return self.get_multiwell_energy(etas) + \
            self.get_gradient_energy(etas) + \
            self.get_chemical_energy(etas, potentials)

    def get_chi(self, etas: list[ngs.CoefficientFunction],
                potentials: list[ngs.CoefficientFunction]) \
                -> ngs.CoefficientFunction:
        hs = self.h(etas)
        chi = ngs.CF(0)
        for phase, h in zip(self.phases, hs):
            chi += h * phase.get_chi(self.components, potentials, self.T)
        return chi

    def get_D_times_chi(self, etas, potentials) -> ngs.CoefficientFunction:
        hs = self.h(etas)
        Dchi = ngs.CF(0)
        for phase, h in zip(self.phases, hs):
            Dchi += h * phase.D * phase.get_chi(self.components, potentials, self.T)
        return Dchi

    def _get_concentrations(self, etas: list[ngs.CoefficientFunction],
                           potentials: list[ngs.CoefficientFunction]) \
            -> dict[Component, ngs.CoefficientFunction]:
        c = {comp : ngs.CF(0) for comp in self.components[:-1]}
        for phase, h in zip(self.phases, self.h(etas)):
            concentrations = phase.get_concentrations(self.components,
                                                      potentials, self.T)
          
            for comp, concentration in concentrations.items():
                c[comp] += h * concentration
        return c

    def _setup(self):
        n_p = len(self.phases)
        n_c = len(self.components)

        h1 = ngs.H1(self.mesh, order=self.order)
        fes_comp = h1
        if self.mass_conservation:
            fes_comp = h1*h1
        fes_comps = fes_comp**(n_c-1)
        fes_phase = h1**n_p
        self.fes = fes_comps * fes_phase
        w, etas = self.fes.TrialFunction()
        dw, detas = self.fes.TestFunction()
        if self.mass_conservation:
            c = [w[i][0] for i in range(n_c-1)]
            w = [w[i][1] for i in range(n_c-1)]
            dc = [dw[i][1] for i in range(n_c-1)]
            dw = [dw[i][0] for i in range(n_c-1)]

        self.gf = ngs.GridFunction(self.fes)
        self.gfw = self.gf.components[0]
        self.gfetas = self.gf.components[1]
        if self.mass_conservation:
            self.gfcs = [gf.components[1] for gf in self.gfw.components]
            self.gfw = [gf.components[0] for gf in self.gfw.components]
        self.gf_old = ngs.GridFunction(self.fes)
        gfw_old = self.gf_old.components[0]
        gfetas_old = self.gf_old.components[1]
        if self.mass_conservation:
            gfcs_old = [gf.components[1] for gf in gfw_old.components]
            gfw_old = [gf.components[0] for gf in gfw_old.components]

        omega = self.get_energy(etas, w)

        self.a = ngs.BilinearForm(self.fes)
        self.a += 1/self.dt * (etas-gfetas_old) * detas * ngs.dx
        self.a += self.L * (omega * ngs.dx).Diff(etas, detas)
        # for eta, eta_old, deta in zip(etas, gfetas_old, detas):
        #     self.a += 1/self.dt * (eta - eta_old) * deta * ngs.dx
        #     self.a += self.L * (omega * ngs.dx).Diff(eta, deta)

        # TODO: Implement for multiple components
        list_etas = [etai for etai in etas]
        concentrations = self._get_concentrations(list_etas, w)
        if self.mass_conservation:
            self.a += 1/self.dt * (c[0]-gfcs_old[0]) * dc[0] * ngs.dx
            Dchi = self.get_D_times_chi(etas, w)
            for wi, dci in zip(w, dc):
                self.a += self.Vm * Dchi * ngs.InnerProduct(ngs.grad(wi),ngs.grad(dci)) * ngs.dx
            self.a += (concentrations[self.components[0]] - c[0]) * dw * ngs.dx
        else:
            chi = self.get_chi(etas, w)
            Dchi = self.get_D_times_chi(etas, w)
            self.a += self.Vm * chi * 1/self.dt * (w-gfw_old) * dw * ngs.dx
            self.a += self.Vm * Dchi * ngs.InnerProduct(ngs.grad(w),ngs.grad(dw)) * ngs.dx
            rho = 1/self.Vm * concentrations[self.components[0]]
            for etai in list_etas:
                etai.MakeVariable()
            self.a += self.Vm * sum(rho.Diff(etai) * 1/self.dt * (etai - etai_old) for etai, etai_old in zip(list_etas, gfetas_old.components)) * dw * ngs.dx

    def get_concentrations(self) -> dict[Component,ngs.CoefficientFunction]:
        if self.mass_conservation:
            cons = { comp : self.gfcs[self.component_indices[comp]] for comp in self.components[:-1] }
        else:
            cons = self._get_concentrations(self.gfetas, self.gfw)
        cons[self.components[-1]] = ngs.CF(1) - sum(cons.values())
        return cons

    def set_initial_conditions(self, \
        phases: dict[Phase, ngs.CoefficientFunction] | None = None,
        components: dict[Component,
                         dict[Phase, ngs.CoefficientFunction | float]] | None = None):
        """
Initial conditions for phase. For components, for each phase an initial condition must be given.
        """
        if not hasattr(self, "gfetas"):
            self._setup()
        if phases is not None:
            for phase, ic in phases.items():
                self.gfetas.components[self.phase_indices[phase]].Set(ic)
        if components is not None:
            tmp = self.gfw.vec.CreateVector()
            tmp[:] = 0
            if self.mass_conservation:
                tmp2 = self.gfcs[0].vec.CreateVector()
                tmp2[:] = 0
            hs = self.h(self.gfetas)
            # TODO: Implement for multiple components
            for h, phase in zip(hs, self.phases):
                concentrations = { comp: phase_concentration[phase] \
                                   for comp, phase_concentration in components.items() }
                potentials = phase.get_potentials(concentrations,
                                                 self.components[-1],
                                                 self.T)
                self.gfw.Set(h * potentials[self.components[0]])
                tmp += self.gfw.vec
                if self.mass_conservation:
                    self.gfcs[0].Set(h * concentrations[self.components[0]])
                    tmp2 += self.gfcs[0].vec
            self.gfw.vec.data = tmp
            if self.mass_conservation:
                self.gfcs[0].vec.data = tmp2

    def do_timestep(self):
        if not hasattr(self, "a"):
            self._setup()
        self.gf_old.vec.data = self.gf.vec
        def callback(it, err):
            import math
            if math.isnan(err):
                raise NewtonNotConverged()
        solver = ngs.nonlinearsolvers.NewtonSolver(self.a, self.gf,
                                                   inverse="pardiso")
        try:
            solver.Solve(maxerr=1e-9, printing=self.print_newton, callback=callback)
            self.time += self.dt.Get()
        except NewtonNotConverged as e:
            if self.dt.Get() > self.min_timestep:
                self.dt.Set(self.dt.Get()/2)
                self.gf.vec.data = self.gf_old.vec
                print("Newton not converged, try smaller timestep:", self.dt.Get())
                self.do_timestep()
            else:
                raise e
        
    def solve(self, endtime, callback=None):
        with ngs.TaskManager():
            while self.time < endtime * (1-1e-6):
                self.do_timestep()
                if callback is not None:
                    callback()
                self.time += self.dt.Get()


