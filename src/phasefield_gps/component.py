
from abc import abstractmethod
from .phase import Phase

class Component:
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_number_phases(self) -> int:
        pass

class IdealSolutionComponent(Component):
    def __init__(self, phase_energies: dict[Phase, float], **kwargs):
        super().__init__(**kwargs)
        self.phase_energies = phase_energies

    def get_number_phases(self) -> int:
        return len(self.phase_energies)


