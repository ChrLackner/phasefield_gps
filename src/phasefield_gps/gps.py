
import ngsolve as ngs
from .phase import Phase
from .component import Component, IdealSolutionComponent

class CGWrapper:
    def __init__(self, a, *args, **kwargs):
        self.a = a
        self.args = args
        self.kwargs = kwargs
        self.cg = None

    def ensure_cg(self):
        if self.cg is None:
            self.cg = ngs.krylovspace.CGSolver(self.a.mat, *self.args, **self.kwargs)

    def __getattr__(self, __name: str):
        self.ensure_cg()
        return self.cg.__getattribute__(__name)

    def __mul__(self, other):
        self.ensure_cg()
        return self.cg * other

class NonlinearException(Exception):
    pass

class NewtonNotConverged(NonlinearException):
    def __init__(self):
        super().__init__("Newton method did not converge!")
class ErrorTooLarge(NonlinearException):
    def __init__(self):
        super().__init__("Timestepping error too large!")

class GrandPotentialSolver:
    def __init__(self, mesh: ngs.Mesh, components: list[Component],
                 phases: list[Phase], molar_volume: float,
                 interface_mobility: float,
                 temperature: float | ngs.Parameter,
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
        self._user_timestep = None
        self.order = 2
        self.mass_conservation = True
        self.gamma = ngs.Matrix(len(self.phase_indices), len(self.phase_indices))
        self.gamma[:] = 1.5
        for i in range(len(self.phase_indices)):
            self.gamma[i,i] = 0
        self.T = temperature if isinstance(temperature, ngs.Parameter) else ngs.Parameter(temperature)
        self.l = interface_width
        self.Vm = molar_volume
        self.mu = interface_mobility
        self.time = 0
        self.print_newton = False
        self.min_timestep = 1e-7
        self.anisotropy = None
        # self.timestepping_tolerance = 1e-4
        self.timestepping_tolerance = None

    def set_timestep(self, dt: float):
        self.dt.Set(dt)
        self._user_timestep = dt

    def kappa(self, etas=None) -> ngs.CF | float:
        for i in range(len(self.phase_indices)):
            for j in range(i+1, len(self.phase_indices)):
                if self.gamma[i,j] != 1.5:
                    raise NotImplementedError("TODO: implement for gamma != 1.5, gamma = " + str(self.gamma))
        return 3/4 * self.sigma(etas) * self.l

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

    def sigma(self, etas: list[ngs.CoefficientFunction] | None = None) \
            -> ngs.CoefficientFunction | float:
        # TODO: implement for more than 2 phases
        sigma = self.phases[1].\
            get_surface_energy(self.phases[0],
                               # (None if etas is None else ngs.grad(etas)[0,:]))
                               # (None if etas is None else ngs.grad(self.gfetas)[0,:]))
                               False if etas == False else ngs.Grad(self.gfetas_old)[0,:])
        # if etas is not None:
        #     ngs.Draw(ngs.grad(self.gfetas)[0,:], self.mesh, "grad")
        # ngs.Draw(sigma, self.mesh, "sigma")
        return sigma

    def m(self, etas=None) -> ngs.CF | float:
        return 6 * self.sigma(etas)/self.l

    def set_Temperature(self, T, new_concentrations : None | dict[Phase, ngs.CF]=None):
        if hasattr(self, "gfetas"):
            print("Set temperature to", T)
            if new_concentrations is not None:
                components = new_concentrations
            else:
                potentials = self.gfw
                fes_tmp = ngs.H1(self.mesh, order=3)
                components = {
                    comp : {
                        phase : ngs.GridFunction(fes_tmp)
                        for phase in self.phases }
                    for comp in self.components }
                for phase in self.phases:
                    concentrations = phase.get_concentrations(self.components,
                                                              potentials, self.T)
                    for comp in self.components[:-1]:
                        components[comp][phase].Interpolate(concentrations[comp])

                        components[self.components[-1]][phase].\
                            Interpolate(ngs.CF(1)-\
                                        sum(concentrations[comp] \
                                            for comp in self.components[:-1]))
        # this also calls compute phase energies
        self.T.Set(T)
        if hasattr(self, "gfetas"):
            self.set_initial_conditions(phases=None,
                                        components=components)
            self._build_forms()

    def get_multiwell_energy(self, etas : list[ngs.CoefficientFunction] | None = None) \
            -> ngs.CoefficientFunction:
        if etas is None:
            etas = self.gfetas
        return ngs.CF(self.m(etas) * (sum((eta**4/4-eta**2/2) for eta in etas) + sum(sum(self.gamma[i,j] * eta1**2 * eta2**2 for j,eta1 in enumerate(etas) if j > i) for i,eta2 in enumerate(etas)) + 0.25))

    def get_gradient_energy(self, etas: list[ngs.CoefficientFunction] | None = None) \
            -> ngs.CoefficientFunction:
        if etas is None:
            etas = self.gfetas
        if isinstance(etas, tuple):
            return ngs.CF(self.kappa(etas)/2 * sum(ngs.InnerProduct(ngs.grad(eta), ngs.grad(eta)) for eta in etas))
        return self.kappa(etas)/2 * ngs.InnerProduct(ngs.grad(etas), ngs.grad(etas))

    def get_chemical_energy(self, etas: list[ngs.CoefficientFunction] | None = None,
                            potentials: list[ngs.CoefficientFunction] | None = None) \
                            -> ngs.CoefficientFunction:
        if etas is None:
            etas = self.gfetas
        if potentials is None:
            potentials = self.gfw
        omega_chem = ngs.CF(0)
        hs = self.h(etas)
        for phase, h in zip(self.phases, hs):
            omega = phase.get_chemical_energy_from_potential(self.components, potentials, self.T)
            omega_chem += h * omega
        return 1/self.Vm * omega_chem

    def get_energy(self, etas: list[ngs.CoefficientFunction] | None = None,
                   potentials: list[ngs.CoefficientFunction] | None = None) \
                   -> ngs.CoefficientFunction:
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
            Dchi += h * phase.get_D_times_chi(self.components, potentials, self.T)
        return Dchi

    def _get_concentrations(self, etas: list[ngs.CoefficientFunction],
                           potentials: list[ngs.CoefficientFunction]) \
            -> dict[Component, ngs.CoefficientFunction]:
        c = {comp : ngs.CF(0) for comp in self.components[:-1]}
        for phase, h in zip(self.phases, self.h(etas)):
            concentrations = phase.get_concentrations(self.components,
                                                      potentials, self.T)

            for comp in self.components[:-1]:
                c[comp] += h * concentrations[comp]
        return c

    def _setup(self):
        n_p = len(self.phases)
        n_c = len(self.components)

        h1 = ngs.H1(self.mesh, order=self.order)
        fes_comp = h1
        if self.mass_conservation:
            fes_comp = h1*h1
        if n_c > 2:
            fes_comp = fes_comp**(n_c-1)
        fes_phase = h1**n_p
        self.fes = fes_comp * fes_phase

        self.gf = ngs.GridFunction(self.fes)
        self.gfw = self.gf.components[:-1]
        self.gfetas = self.gf.components[-1]
        if self.mass_conservation:
            self.gfcs = self.gfw[1]
            self.gfw = self.gfw[0]
        else:
            self.gfw = self.gfw[0]
        self.gf_old = ngs.GridFunction(self.fes)
        self._gf_coarse = ngs.GridFunction(self.fes)
        self.gfw_old = self.gf_old.components[:-1]
        self._gfw_coarse = self._gf_coarse.components[:-1]
        self.gfetas_old = self.gf_old.components[-1]
        self._gfetas_coarse = self._gf_coarse.components[-1]
        if self.mass_conservation:
            self.gfcs_old = self.gfw_old[1]
            self.gfw_old = self.gfw_old[0]

    def _build_forms(self):
        n_p = len(self.phases)
        n_c = len(self.components)
        trial = self.fes.TrialFunction()
        test = self.fes.TestFunction()
        w, etas = trial[:-1], trial[-1]
        dw, detas = test[:-1], test[-1]
        if self.mass_conservation:
            c = [w[i*2+1] for i in range(n_c-1)]
            w = [w[i*2] for i in range(n_c-1)]
            dc = [dw[i*2+1] for i in range(n_c-1)]
            dw = [dw[i*2] for i in range(n_c-1)]
        omega = self.get_energy(etas, w)

        self.a = ngs.BilinearForm(self.fes)
        forms = 1/self.dt * (etas-self.gfetas_old) * detas * ngs.dx
        forms += (self.L * omega * ngs.dx).Diff(etas, detas)
        if self.mass_conservation:
            print("len c = ", n_c)
            assert n_c == 2, "more components still need to be implememented " + str(n_c)
            mu0 = self.phases[0].get_potentials(
                concentrations={ self.components[0] : c[0],
                                 self.components[1] : None },
                solvent=self.components[1], T=self.T)[self.components[0]]
            mu1 = self.phases[1].get_potentials(
                concentrations={ self.components[0] : c[0],
                                 self.components[1] : None },
                solvent=self.components[1], T=self.T)[self.components[0]]
            dmudphi = ngs.CF((mu0-mu1,mu1-mu0))
            chi = self.get_chi(etas, w)
            forms += 8.314 * self.T * self.L * w[0] * chi * dmudphi * detas * ngs.dx
        else:
            raise Exception("Not yet implemented!")

        # TODO: Implement for multiple components
        list_etas = [etai for etai in etas]
        concentrations = self._get_concentrations(list_etas, w)
        if self.mass_conservation:
            for ci, dci, gfci_old in zip(c, dc, [self.gfcs_old]):
                forms += 1/self.dt * (ci-gfci_old) * dci * ngs.dx
            Dchi = self.get_D_times_chi(etas, w)
            for wi, dci in zip(w, dc):
                # forms += self.Vm * Dchi * ngs.InnerProduct(ngs.grad(wi),ngs.grad(dci)) * ngs.dx
                forms += Dchi * ngs.InnerProduct(ngs.grad(wi),ngs.grad(dci)) * ngs.dx
            for ci, dwi in zip(c, dw):
                forms += (concentrations[self.components[0]] - ci) * dwi * ngs.dx
        else:
            chi = self.get_chi(etas, w)
            Dchi = self.get_D_times_chi(etas, w)
            # forms += self.Vm * chi * 1/self.dt * (w[0]-self.gfw_old[0]) * dw * ngs.dx
            # forms += self.Vm * Dchi * ngs.InnerProduct(ngs.grad(w[0]),ngs.grad(dw[0])) * ngs.dx
            forms += chi * 1/self.dt * (w[0]-self.gfw_old[0]) * dw * ngs.dx
            forms += Dchi * ngs.InnerProduct(ngs.grad(w[0]),ngs.grad(dw[0])) * ngs.dx
            rho = 1/self.Vm * concentrations[self.components[0]]
            for etai in list_etas:
                etai.MakeVariable()
            forms += self.Vm * sum(rho.Diff(etai) * 1/self.dt * (etai - etai_old) for etai, etai_old in zip(list_etas, self.gfetas_old.components)) * dw * ngs.dx
        self.a += forms.Compile()
        # self.c = ngs.Preconditioner(self.a, "bddc")

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
                self.gfetas.components[self.phase_indices[phase]].Set(ic, dual=True)
        if components is not None:
            self.compute_phase_energies(components)
            tmp = self.gfw.vec.CreateVector()
            tmp[:] = 0
            if self.mass_conservation:
                tmp2 = self.gfcs.vec.CreateVector()
                tmp2[:] = 0
            hs = self.h(self.gfetas)
            # TODO: Implement for multiple components
            for h, phase in zip(hs, self.phases):
                concentrations = { comp: phase_concentration[phase] \
                                   for comp, phase_concentration in components.items() }
                potentials = phase.get_potentials(concentrations,
                                                 self.components[-1],
                                                 self.T)
                self.gfw.Set(h * potentials[self.components[0]], dual=True)
                tmp += self.gfw.vec
                if self.mass_conservation:
                    self.gfcs.Set(h * concentrations[self.components[0]], dual=True)
                    tmp2.data += self.gfcs.vec
            self.gfw.vec.data = tmp
            if self.mass_conservation:
                self.gfcs.vec.data = tmp2
        self.gf_old.vec.data = self.gf.vec

    def compute_phase_energies(self, component_concentrations):
        assert len(self.components) == 2
        assert type(self.components[0]) == IdealSolutionComponent and \
            type(self.components[1]) == IdealSolutionComponent
        vol_domain = ngs.Integrate(ngs.CF(1), self.mesh)
        X = sum([ngs.Integrate(
            component_concentrations[self.components[0]][phase] * \
            self.get_phase(phase),
            self.mesh)/vol_domain for phase in self.phases])
        energies = [c.get_base_phase_energies_at_temperature(self.T.Get()) for c in self.components]
        phase_energy = lambda c, phase: (c * energies[0][phase] + (1-c) * energies[1][phase] + phase.site_variable * 8.314 * self.T.Get() * (c*ngs.log(c) + (1-c) * ngs.log(1-c)))
        import numpy as np
        from scipy.spatial import ConvexHull
        pts = np.arange(0,1,0.0001)[1:-1]
        gs = np.array([phase_energy(p, self.phases[0]) for p in pts])
        gl = np.array([phase_energy(p, self.phases[1]) for p in pts])
        # Minimization and common tangent
        g = np.min(np.vstack([gs, gl]), axis=0)
        hull = ConvexHull(np.array([pts, g]).T)
        k = hull.vertices
        k = np.sort(k)  # Ensure k is sorted if not already
        find = np.where(np.diff(k) > 1)[0]
        # print("X = ", X)
        check_in_between = 0
        for found in find:
            ind1 = k[found]
            ind2 = k[found+1]
            if X > pts[ind1] and X < pts[ind2]:
                break
            check_in_between += 1
        if check_in_between == len(find): # not found
            ind1 = np.where(pts < X)[0][-1]
            ind2 = ind1+1
        derivative = ((g[ind2] - g[ind1]) / (pts[ind2] - pts[ind1]))
        # print("derivative = ", derivative)
        self.components[0].phase_energies = { phase : energies[0][phase] - derivative for phase in self.phases }
        self.components[1].phase_energies = { phase : energies[1][phase] for phase in self.phases }
        # print("new energies = ", self.components[0].phase_energies)

    def plot_energy_landscape(self):
        assert len(self.components) == 2
        import matplotlib.pyplot as plt
        import numpy as np
        fig = plt.figure()
        ax = fig.add_subplot(111)
        c1, c2 = self.components
        assert type(c1) == IdealSolutionComponent and type(c2) == IdealSolutionComponent
        for phase in self.phases:
            c_vals = np.linspace(0, 1, 100)[1:-1]
            # e_vals = [c * c1.phase_energies[phase] + (1-c) * c2.phase_energies[phase] + phase.site_variable * 8.314 * self.T.Get() * (c*ngs.log(c) + (1-c) * ngs.log(1-c))for c in c_vals]
            e_vals = [phase.get_chemical_energy({ self.components[0] : c, self.components[1]: 1-c }, self.T.Get(), use_ifpos=False) for c in c_vals]
            ind = e_vals.index(min(e_vals))
            ax.plot(c_vals, e_vals, label=phase.name + " " + str(c_vals[ind]))
        # hard coded for now
        xvals = np.linspace(-0.5e-3, 0.5e-3, 11)
        pts = self.mesh(xvals, 0.)
        cvals = self.get_concentrations()[self.components[0]](pts)
        energies = self.Vm * self.get_chemical_energy()(pts)
        ax.plot(cvals, energies, "o", label="Concentrations")

        fig.legend()
        return fig

    @ngs.TimeFunction
    def do_timestep(self):
        if not hasattr(self, "a"):
            self._build_forms()
        # self.gf_old.vec.data = self.gf.vec
        def callback(it, err):
            import math
            if math.isnan(err):
                raise NewtonNotConverged()
        # lin_solver = CGWrapper(self.a, self.c)
        solver = ngs.nonlinearsolvers.NewtonSolver(self.a, self.gf,
                                                   # solver=lin_solver)
                                                   inverse="pardiso")
        try:
            store_gf_old = self.gf_old.vec.Copy()
            solver.Solve(maxerr=1e-9, printing=self.print_newton, callback=callback)
            if self.timestepping_tolerance is not None:
                self._gf_coarse.vec.data = self.gf.vec
                energy_coarse = self.get_energy(self._gfetas_coarse, self._gfw_coarse)
                print("Energy on coarse: ", ngs.Integrate(energy_coarse, self.mesh))
                self.dt.Set(self.dt.Get()/2)
                self.gf.vec.data = self.gf_old.vec
                solver.Solve(maxerr=1e-9, printing=self.print_newton, callback=callback)
                self.gf_old.vec.data = self.gf.vec
                solver.Solve(maxerr=1e-9, printing=self.print_newton, callback=callback)
                energy_fine = self.get_energy()
                tot_energy_fine = ngs.sqrt(ngs.Integrate(energy_fine**2, self.mesh))
                print("Energy on fine: ", ngs.Integrate(energy_fine, self.mesh))
                err = ngs.sqrt(ngs.Integrate((energy_coarse-energy_fine)**2, self.mesh))
                print("Error:", err/tot_energy_fine)
                self.dt.Set(self.dt.Get()*2)
                self.gf_old.vec.data = store_gf_old
                if err > self.timestepping_tolerance * tot_energy_fine:
                    raise ErrorTooLarge()
            self.time += self.dt.Get()
            self.gf_old.vec.data = self.gf.vec
            # if self._user_timestep is not None and self.dt.Get() < self._user_timestep:
            #     self.dt.Set(min(self.dt.Get()*2, self._user_timestep))
            #     print("Increase timestep to:", self.dt.Get())
        except NonlinearException as e:
            if self.dt.Get() > self.min_timestep:
                self.dt.Set(self.dt.Get()/2)
                self.gf.vec.data = self.gf_old.vec
                print(str(e) + ", try smaller timestep:", self.dt.Get())
                self.do_timestep()
            else:
                raise e

    @ngs.TimeFunction
    def solve(self, endtime, callback=None):
        while self.time < endtime * (1-1e-6):
            self.do_timestep()
            if callback is not None:
                callback()
