from ase.calculators.emt import EMT
from ase.constraints import FixInternals
from ase.optimize.bfgs import BFGS
from ase.build import molecule
import copy
import pytest


# Convenience functions to compute linear combinations of internal coordinates
def get_bondcombo(atoms, bondcombo_def):
    return sum([defin[2] * atoms.get_distance(*defin[0:2]) for
                defin in bondcombo_def])


def get_anglecombo(atoms, anglecombo_def):
    return sum([defin[3] * atoms.get_angle(*defin[0:3]) for
                defin in anglecombo_def])


def get_dihedralcombo(atoms, dihedralcombo_def):
    return sum([defin[4] * atoms.get_dihedral(*defin[0:4]) for
                defin in dihedralcombo_def])


def setup_atoms():
    atoms = molecule('CH3CH2OH', vacuum=5.0)
    atoms.rattle(stdev=0.3)
    return atoms


def setup_fixinternals():
    atoms = setup_atoms()

    # Angles, Bonds, Dihedrals are built up with pairs of constraint
    # value and indices defining the constraint
    # Linear combinations of bond lengths are built up similarly with the
    # coefficients appended to the indices defining the constraint

    # Fix bond between atoms 1 and 2 to 1.4
    bond_def = [1, 2]
    target_bond = 1.4

    # Fix angle to whatever it was from the start
    angle_def = [6, 0, 1]
    target_angle = atoms.get_angle(*angle_def)

    # Fix this dihedral angle to whatever it was from the start
    dihedral_def = [6, 0, 1, 2]
    target_dihedral = atoms.get_dihedral(*dihedral_def)

    # Initialize constraint
    constr = FixInternals(bonds=[(target_bond, bond_def)],
                              angles_deg=[(target_angle, angle_def)],
                              dihedrals_deg=[(target_dihedral, dihedral_def)],
                              epsilon=1e-10)
    print(constr)
    return (atoms, constr, bond_def, target_bond, angle_def, target_angle,
            dihedral_def, target_dihedral)


def test_fixinternals():
    (atoms, constr, bond_def, target_bond, angle_def, target_angle,
     dihedral_def, target_dihedral) = setup_fixinternals()

    calc = EMT()

    opt = BFGS(atoms)

    previous_angle = atoms.get_angle(*angle_def)
    previous_dihedral = atoms.get_dihedral(*dihedral_def)

    print('angle before', previous_angle)
    print('dihedral before', previous_dihedral)
    print('bond length before', atoms.get_distance(*bond_def))
    print('target bondlength', target_bond)

    atoms.calc = calc
    atoms.set_constraint(constr)
    print('-----Optimization-----')
    opt.run(fmax=0.01)

    new_angle = atoms.get_angle(*angle_def)
    new_dihedral = atoms.get_dihedral(*dihedral_def)
    new_bondlength = atoms.get_distance(*bond_def)

    print('angle after', new_angle)
    print('dihedral after', new_dihedral)
    print('bondlength after', new_bondlength)

    err1 = new_angle - previous_angle
    err2 = new_dihedral - previous_dihedral
    err3 = new_bondlength - target_bond

    print('error in angle', repr(err1))
    print('error in dihedral', repr(err2))
    print('error in bondlength', repr(err3))

    assert err1 < 1e-11
    assert err2 < 1e-12
    assert err3 < 1e-12


def setup_combos():
    atoms = setup_atoms()

    # Fix linear combination of two bond lengths with atom indices 0-8 and
    # 0-6 with weighting coefficients 1.0 and -1.0 to the current value.
    # In other words, fulfil the following constraint:
    # 1.0 * atoms.get_distance(2, 1) + -1.0 * atoms.get_distance(2, 3) = const.
    bondcombo_def = [[2, 1, 1.0], [2, 3, -1.0]]
    target_bondcombo = get_bondcombo(atoms, bondcombo_def)

    # Fix linear combination of two angles
    # 1. * atoms.get_angle(7, 0, 8) + 1. * atoms.get_angle(7, 0, 6) = const.
    anglecombo_def = [[7, 0, 8, 1.], [7, 0, 6, 1]]
    target_anglecombo = get_anglecombo(atoms, anglecombo_def)

    # Fix linear combination of two dihedrals
    dihedralcombo_def = [[3, 2, 1, 4, 1.0], [2, 1, 0, 7, 1.0]]
    target_dihedralcombo = get_dihedralcombo(atoms, dihedralcombo_def)

    # Initialize constraint
    constr = FixInternals(bondcombos=[(target_bondcombo, bondcombo_def)],
                          anglecombos=[(target_anglecombo, anglecombo_def)],
                          dihedralcombos=[(target_dihedralcombo,
                                           dihedralcombo_def)], epsilon=1e-10)
    print(constr)
    return (atoms, constr, bondcombo_def, target_bondcombo, anglecombo_def,
            target_anglecombo, dihedralcombo_def, target_dihedralcombo)


@pytest.mark.xfail
def test_combos():
    # XXX https://gitlab.com/ase/ase/-/issues/868
    (atoms, constr, bondcombo_def, target_bondcombo, anglecombo_def,
     target_anglecombo, dihedralcombo_def,
     target_dihedralcombo) = setup_combos()

    ref_bondcombo = get_bondcombo(atoms, bondcombo_def)
    ref_anglecombo = get_anglecombo(atoms, anglecombo_def)
    ref_dihedralcombo = get_dihedralcombo(atoms, dihedralcombo_def)

    atoms.calc = EMT()
    atoms.set_constraint(constr)

    opt = BFGS(atoms)
    opt.run(fmax=0.01)

    new_bondcombo = get_bondcombo(atoms, bondcombo_def)
    new_anglecombo = get_anglecombo(atoms, anglecombo_def)
    new_dihedralcombo = get_dihedralcombo(atoms, dihedralcombo_def)

    err_bondcombo = new_bondcombo - ref_bondcombo
    err_anglecombo = new_anglecombo - ref_anglecombo
    err_dihedralcombo = new_dihedralcombo - ref_dihedralcombo

    print('error in bondcombo:', repr(err_bondcombo))
    print('error in anglecombo:', repr(err_anglecombo))
    print('error in dihedralcombo:', repr(err_dihedralcombo))

    for err in [err_bondcombo, err_anglecombo, err_dihedralcombo]:
        assert err < 1e-12


def test_index_shuffle():
    (atoms, constr, bond_def, target_bond, angle_def, target_angle,
     dihedral_def, target_dihedral) = setup_fixinternals()

    constr2 = copy.deepcopy(constr)

    # test no change, test constr.get_indices()
    assert all(a == b for a, b in zip(constr.get_indices(), (0, 1, 2, 6, 8)))
    constr.index_shuffle(atoms, range(len(atoms)))
    assert all(a == b for a, b in zip(constr.get_indices(), (0, 1, 2, 6, 8)))

    # test full constraint is not part of new slice
    with pytest.raises(IndexError):
        constr.index_shuffle(atoms, [0])

    # test correct shuffling
    constr2.index_shuffle(atoms, [1, 2, 0, 6])
    assert constr2.bonds[0][1] == [0, 1]
    assert constr2.angles[0][1] == [3, 2, 0]
    assert constr2.dihedrals[0][1] == [3, 2, 0, 1]


def test_combo_index_shuffle():
    (atoms, constr, bondcombo_def, target_bondcombo, anglecombo_def,
     target_anglecombo, dihedralcombo_def,
     target_dihedralcombo) = setup_combos()

    # test no change, test constr.get_indices()
    answer = (0, 1, 2, 3, 4, 6, 7, 8)
    assert all(a == b for a, b in zip(constr.get_indices(), answer))
    constr.index_shuffle(atoms, range(len(atoms)))
    assert all(a == b for a, b in zip(constr.get_indices(), answer))

    # test anglecombo not part of slice
    constr.index_shuffle(atoms, [1, 2, 3, 4, 0, 7])
    assert constr.bondcombos[0][1] == [[1, 0, 1.0], [1, 2, -1.0]]
    assert constr.dihedralcombos[0][1] == [[2, 1, 0, 3, 1.0], [1, 0, 4, 5, 1.0]]
