
from abc import abstractmethod
from .phase import Phase

class Component:
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_number_phases(self) -> int:
        pass

class IdealSolutionComponent(Component):
    def __init__(self, phase_energies: dict[Phase, float | dict[float,float] | str],
                 **kwargs):
        """
Phase energies is a dictionary with keys as Phase objects and values as the
energy of the phase. The energy of the phase can be a float or a dictionary
with keys as the temperature and values as the energy of the phase at that
temperature.
        """
        super().__init__(**kwargs)
        if isinstance(list(phase_energies.values())[0], str):
            self.base_phase_energies = self._read_from_file(phase_energies)
        else:
            self.base_phase_energies = phase_energies

    def _read_from_file(self, phase_filename: dict[Phase, str]) \
            -> dict[Phase, float | dict[float,float]]:
        energies = {}
        for phase, filename in phase_filename.items():
            with open(filename, 'r', encoding='ASCII') as f:
                found_start = False
                energies[phase] = {}
                for line in f:
                    values = line.strip().split()
                    if not found_start:
                        if len(values) > 2 and values[0].strip() == 'T(K)' and \
                                values[1].strip() == 'P(bar)':
                            found_start = True
                        continue
                    T = float(values[0].strip())
                    energy = float(values[2].strip())
                    energies[phase][T] = energy
        return energies

    def get_base_phase_energies_at_temperature(self, temperature: float) \
            -> dict[Phase,float]:
        energies = {}
        for phase, energy in self.base_phase_energies.items():
            if isinstance(energy, dict):
                last = None
                last_t = None
                for t, e in energy.items():
                    if temperature < t:
                        break
                    last = e
                    last_t = t
                if last is None or last_t is None:
                    energies[phase] = e
                elif last_t == t:
                    energies[phase] = e
                else:
                    energies[phase] = last + (temperature - last_t) * \
                                      (e - last) / (t - last_t)
            else:
                energies[phase] = energy
        return energies

    def get_number_phases(self) -> int:
        return len(self.base_phase_energies)

