from numpy.random import RandomState


class RandomCalculator:
    """Fake Calculator class."""
    def __init__(self):
        self.rng = RandomState(42)

    def get_forces(self, atoms):
        return self.rng.rand(len(atoms), 3)

    def get_dipole_moment(self, atoms):
        return self.rng.rand(3)
