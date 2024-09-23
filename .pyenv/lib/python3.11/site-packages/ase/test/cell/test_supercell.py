import pytest
import numpy as np
from ase.build import bulk, make_supercell
from ase.lattice import FCC, BCC
from ase.calculators.emt import EMT


a = 4.1


@pytest.fixture
def atoms():
    atoms = bulk('Au', a=a)
    return atoms


def test_supercell(atoms):
    assert atoms.cell.get_bravais_lattice().name == 'FCC'

    # Since FCC and BCC are reciprocal, their product is cubic:
    P = BCC(2.0).tocell()
    assert np.allclose(np.linalg.det(P), 4)

    cubatoms = make_supercell(atoms, P)
    assert np.allclose(cubatoms.cell, a * np.eye(3))

    # Also test some of the Cell object methods now that we are at it:
    assert len(cubatoms) == 4
    assert cubatoms.cell.orthorhombic
    assert np.allclose(cubatoms.cell.lengths(), a)
    assert cubatoms.cell.get_bravais_lattice().name == 'CUB'


def test_bcc_to_cub_transformation():
    bcc = bulk('Fe', a=a)
    P = FCC(2.0).tocell()
    assert np.allclose(np.linalg.det(P), 2)
    cubatoms = make_supercell(bcc, P)
    assert np.allclose(cubatoms.cell, a * np.eye(3))


def getenergy(atoms, eref=None):
    atoms.calc = EMT()
    e = atoms.get_potential_energy() / len(atoms)
    if eref is not None:
        err = abs(e - eref)
        print('natoms', len(atoms), 'err', err)
        assert err < 1e-12, err
    return e


def test_random_transformations(atoms):
    # Finally we want to check some random transformations.
    # This will produce some weird supercells, all of which should have
    # the same energy.

    rng = np.random.RandomState(44)

    e0 = getenergy(atoms)

    imgs = []

    i = 0
    while i < 10:
        P = rng.randint(-2, 3, size=(3, 3))
        detP = np.linalg.det(P)
        if detP == 0:
            continue
        elif detP < 0:
            P[0] *= -1
        bigatoms = make_supercell(atoms, P)
        imgs.append(bigatoms)
        getenergy(bigatoms, eref=e0)
        i += 1
    # from ase.visualize import view
    # view(imgs)
