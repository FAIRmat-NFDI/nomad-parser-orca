import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
    RHL, MCL, MCLC, TRI, OBL, HEX2D, RECT, CRECT, SQR, LINE
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
    neighbor_k_search, calculate_weights, steepest_descent, md_min, \
    rotation_from_projection, Wannier


calc = pytest.mark.calculator


@pytest.fixture()
def rng():
    return np.random.RandomState(0)


@pytest.fixture(scope='module')
def _std_calculator_gpwfile(tmp_path_factory, factories):
    factories.require('gpaw')
    import gpaw
    atoms = molecule('H2', pbc=True)
    atoms.center(vacuum=3.)
    gpw_path = tmp_path_factory.mktemp('sub') / 'dumpfile.gpw'
    calc = gpaw.GPAW(gpts=(8, 8, 8), nbands=4, kpts=(2, 2, 2),
                     symmetry='off', txt=None)
    atoms.calc = calc
    atoms.get_potential_energy()
    calc.write(gpw_path, mode='all')
    return gpw_path


@pytest.fixture(scope='module')
def std_calculator(_std_calculator_gpwfile):
    import gpaw
    return gpaw.GPAW(_std_calculator_gpwfile, txt=None)


@pytest.fixture
def wan(rng, std_calculator):
    def _wan(gpts=(8, 8, 8),
             atoms=None,
             calc=None,
             nwannier=2,
             fixedstates=None,
             initialwannier='bloch',
             kpts=(1, 1, 1),
             file=None,
             rng=rng,
             full_calc=False,
             std_calc=True):
        if std_calc and calc is None and atoms is None:
            calc = std_calculator
        else:
            if calc is None:
                gpaw = pytest.importorskip('gpaw')
                calc = gpaw.GPAW(gpts=gpts, nbands=nwannier, kpts=kpts,
                                 symmetry='off', txt=None)
            if atoms is None and not full_calc:
                pbc = (np.array(kpts) > 1).any()
                atoms = molecule('H2', pbc=pbc)
                atoms.center(vacuum=3.)
            if not full_calc:
                atoms.calc = calc
                atoms.get_potential_energy()
        return Wannier(nwannier=nwannier,
                       fixedstates=fixedstates,
                       calc=calc,
                       initialwannier=initialwannier,
                       file=None,
                       rng=rng)
    return _wan


def bravais_lattices():
    return [CUB(1), FCC(1), BCC(1), TET(1, 2), BCT(1, 2),
            ORC(1, 2, 3), ORCF(1, 2, 3), ORCI(1, 2, 3),
            ORCC(1, 2, 3), HEX(1, 2), RHL(1, 110),
            MCL(1, 2, 3, 70), MCLC(1, 2, 3, 70),
            TRI(1, 2, 3, 60, 70, 80), OBL(1, 2, 110),
            HEX2D(1), RECT(1, 2), CRECT(1, 70), SQR(1),
            LINE(1)]


class Paraboloid:

    def __init__(self, pos=(10., 10., 10.), shift=1.):
        self.pos = np.array(pos, dtype=complex)
        self.shift = shift

    def get_gradients(self):
        return 2 * self.pos

    def step(self, dF, updaterot=True, updatecoeff=True):
        self.pos -= dF

    def get_functional_value(self):
        return np.sum(self.pos**2) + self.shift


def orthonormality_error(matrix):
    return np.abs(dagger(matrix) @ matrix - np.eye(len(matrix))).max()


def orthogonality_error(matrix):
    errors = []
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            errors.append(np.abs(matrix[i].T @ matrix[j]))
    return np.max(errors)


def normalization_error(matrix):
    old_matrix = matrix.copy()
    normalize(matrix)
    return np.abs(matrix - old_matrix).max()


def test_gram_schmidt(rng):
    matrix = rng.rand(4, 4)
    assert orthonormality_error(matrix) > 1
    gram_schmidt(matrix)
    assert orthonormality_error(matrix) < 1e-12


def test_lowdin(rng):
    matrix = rng.rand(4, 4)
    assert orthonormality_error(matrix) > 1
    lowdin(matrix)
    assert orthonormality_error(matrix) < 1e-12


def test_random_orthogonal_matrix(rng):
    dim = 4
    matrix = random_orthogonal_matrix(dim, rng=rng, real=True)
    assert matrix.shape[0] == matrix.shape[1]
    assert orthonormality_error(matrix) < 1e-12
    matrix = random_orthogonal_matrix(dim, rng=rng, real=False)
    assert matrix.shape[0] == matrix.shape[1]
    assert orthonormality_error(matrix) < 1e-12


def test_neighbor_k_search():
    kpt_kc = monkhorst_pack((4, 4, 4))
    Gdir_dc = [[1, 0, 0], [0, 1, 0], [0, 0, 1],
               [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    tol = 1e-4
    for d, Gdir_c in enumerate(Gdir_dc):
        for k, k_c in enumerate(kpt_kc):
            kk, k0 = neighbor_k_search(k_c, Gdir_c, kpt_kc, tol=tol)
            assert np.linalg.norm(kpt_kc[kk] - k_c - Gdir_c + k0) < tol


@pytest.mark.parametrize('lat', bravais_lattices())
def test_calculate_weights(lat):
    # Equation from Berghold et al. PRB v61 n15 (2000)
    tol = 1e-5
    cell = lat.tocell()
    g = cell @ cell.T
    w, G = calculate_weights(cell, normalize=False)

    errors = []
    for i in range(3):
        for j in range(3):
            errors.append(np.abs((w * G[:, i] @ G[:, j]) - g[i, j]))

    assert np.max(errors) < tol


def test_steepest_descent():
    tol = 1e-6
    step = 0.1
    func = Paraboloid(pos=np.array([10, 10, 10], dtype=float), shift=1.)
    steepest_descent(func=func, step=step, tolerance=tol, verbose=False)
    assert func.get_functional_value() == pytest.approx(1, abs=1e-5)


def test_md_min():
    tol = 1e-8
    step = 0.1
    func = Paraboloid(pos=np.array([10, 10, 10], dtype=complex), shift=1.)
    md_min(func=func, step=step, tolerance=tol, verbose=False)
    assert func.get_functional_value() == pytest.approx(1, abs=1e-5)


def test_rotation_from_projection(rng):
    proj_nw = rng.rand(6, 4)
    assert orthonormality_error(proj_nw[:int(min(proj_nw.shape))]) > 1
    U_ww, C_ul = rotation_from_projection(proj_nw, fixed=2, ortho=True)
    assert orthonormality_error(U_ww) < 1e-10, 'U_ww not unitary'
    assert orthogonality_error(C_ul.T) < 1e-10, 'C_ul columns not orthogonal'
    assert normalization_error(C_ul) < 1e-10, 'C_ul not normalized'
    U_ww, C_ul = rotation_from_projection(proj_nw, fixed=2, ortho=False)
    assert normalization_error(U_ww) < 1e-10, 'U_ww not normalized'


def test_save(tmpdir, wan):
    wanf = wan(nwannier=4, fixedstates=2, initialwannier='bloch')
    picklefile = tmpdir.join('wanf.pickle')
    f1 = wanf.get_functional_value()
    wanf.save(picklefile)
    wanf.initialize(file=picklefile, initialwannier='bloch')
    assert pytest.approx(f1) == wanf.get_functional_value()


# The following test always fails because get_radii() is broken.
@pytest.mark.parametrize('lat', bravais_lattices())
def test_get_radii(lat, std_calculator, wan):
    if ((lat.tocell() == FCC(a=1).tocell()).all() or
            (lat.tocell() == ORCF(a=1, b=2, c=3).tocell()).all()):
        pytest.skip("lattices not supported, yet")
    atoms = molecule('H2', pbc=True)
    atoms.cell = lat.tocell()
    atoms.center(vacuum=3.)
    calc = std_calculator
    wanf = wan(nwannier=4, fixedstates=2, atoms=atoms, calc=calc,
               initialwannier='bloch', full_calc=True)
    assert not (wanf.get_radii() == 0).all()


def test_get_functional_value(wan):
    # Only testing if the functional scales with the number of functions
    wan1 = wan(nwannier=3)
    f1 = wan1.get_functional_value()
    wan2 = wan(nwannier=4)
    f2 = wan2.get_functional_value()
    assert f1 < f2


@calc('gpaw')
def test_get_centers(factory):
    # Rough test on the position of the Wannier functions' centers
    gpaw = pytest.importorskip('gpaw')
    calc = gpaw.GPAW(gpts=(32, 32, 32), nbands=4, txt=None)
    atoms = molecule('H2', calculator=calc)
    atoms.center(vacuum=3.)
    atoms.get_potential_energy()
    wanf = Wannier(nwannier=2, calc=calc, initialwannier='bloch')
    centers = wanf.get_centers()
    com = atoms.get_center_of_mass()
    assert np.abs(centers - [com, com]).max() < 1e-4


def test_write_cube(wan):
    atoms = molecule('H2')
    atoms.center(vacuum=3.)
    wanf = wan(atoms=atoms)
    index = 0
    # It returns some errors when using file objects, so we use simple filename
    cubefilename = 'wanf.cube'
    wanf.write_cube(index, cubefilename)
    with open(cubefilename, mode='r') as inputfile:
        content = read_cube(inputfile)
    assert pytest.approx(content['atoms'].cell.array) == atoms.cell.array
    assert pytest.approx(content['data']) == wanf.get_function(index)


def test_localize(wan):
    wanf = wan(initialwannier='random')
    fvalue = wanf.get_functional_value()
    wanf.localize()
    assert wanf.get_functional_value() > fvalue


def test_get_spectral_weight_bloch(wan):
    nwannier = 4
    wanf = wan(initialwannier='bloch', nwannier=nwannier)
    for i in range(nwannier):
        assert wanf.get_spectral_weight(i)[:, i].sum() == pytest.approx(1)


def test_get_spectral_weight_random(wan, rng):
    nwannier = 4
    wanf = wan(initialwannier='random', nwannier=nwannier, rng=rng)
    for i in range(nwannier):
        assert wanf.get_spectral_weight(i).sum() == pytest.approx(1)


def test_get_pdos(wan):
    nwannier = 4
    gpaw = pytest.importorskip('gpaw')
    calc = gpaw.GPAW(gpts=(16, 16, 16), nbands=nwannier, txt=None)
    atoms = molecule('H2')
    atoms.center(vacuum=3.)
    atoms.calc = calc
    atoms.get_potential_energy()
    wanf = wan(atoms=atoms, calc=calc,
               nwannier=nwannier, initialwannier='bloch')
    eig_n = calc.get_eigenvalues()
    for i in range(nwannier):
        pdos_n = wanf.get_pdos(w=i, energies=eig_n, width=0.001)
        assert pdos_n[i] != pytest.approx(0)


def test_translate(wan, std_calculator):
    nwannier = 2
    calc = std_calculator
    atoms = calc.get_atoms()
    wanf = wan(nwannier=nwannier, initialwannier='bloch')
    wanf.translate_all_to_cell(cell=[0, 0, 0])
    c0_w = wanf.get_centers()
    for i in range(nwannier):
        c2_w = np.delete(wanf.get_centers(), i, 0)
        wanf.translate(w=i, R=[1, 1, 1])
        c1_w = wanf.get_centers()
        assert np.linalg.norm(c1_w[i] - c0_w[i]) == \
            pytest.approx(np.linalg.norm(atoms.cell.array.diagonal()))
        c1_w = np.delete(c1_w, i, 0)
        assert c1_w == pytest.approx(c2_w)


def test_translate_to_cell(wan, std_calculator):
    nwannier = 2
    calc = std_calculator
    atoms = calc.get_atoms()
    wanf = wan(nwannier=nwannier, initialwannier='bloch')
    for i in range(nwannier):
        wanf.translate_to_cell(w=i, cell=[0, 0, 0])
        c0_w = wanf.get_centers()
        assert (c0_w[i] < atoms.cell.array.diagonal()).all()
        wanf.translate_to_cell(w=i, cell=[1, 1, 1])
        c1_w = wanf.get_centers()
        assert (c1_w[i] > atoms.cell.array.diagonal()).all()
        assert np.linalg.norm(c1_w[i] - c0_w[i]) == \
            pytest.approx(np.linalg.norm(atoms.cell.array.diagonal()))
        c0_w = np.delete(c0_w, i, 0)
        c1_w = np.delete(c1_w, i, 0)
        assert c0_w == pytest.approx(c1_w)


def test_translate_all_to_cell(wan, std_calculator):
    nwannier = 2
    calc = std_calculator
    atoms = calc.get_atoms()
    wanf = wan(nwannier=nwannier, initialwannier='bloch')
    wanf.translate_all_to_cell(cell=[0, 0, 0])
    c0_w = wanf.get_centers()
    assert (c0_w < atoms.cell.array.diagonal()).all()
    wanf.translate_all_to_cell(cell=[1, 1, 1])
    c1_w = wanf.get_centers()
    assert (c1_w > atoms.cell.array.diagonal()).all()
    for i in range(nwannier):
        assert np.linalg.norm(c1_w[i] - c0_w[i]) == \
            pytest.approx(np.linalg.norm(atoms.cell.array.diagonal()))


def test_distances(wan, std_calculator):
    nwannier = 2
    calc = std_calculator
    atoms = calc.get_atoms()
    wanf = wan(nwannier=nwannier, initialwannier='bloch')
    cent_w = wanf.get_centers()
    dist_ww = wanf.distances([0, 0, 0])
    dist1_ww = wanf.distances([1, 1, 1])
    for i in range(nwannier):
        assert dist_ww[i, i] == pytest.approx(0)
        assert dist1_ww[i, i] == pytest.approx(np.linalg.norm(atoms.cell.array))
        for j in range(i + 1, nwannier):
            assert dist_ww[i, j] == dist_ww[j, i]
            assert dist_ww[i, j] == \
                pytest.approx(np.linalg.norm(cent_w[i] - cent_w[j]))


def test_get_hopping_bloch(wan):
    nwannier = 4
    wanf = wan(nwannier=nwannier, initialwannier='bloch')
    hop0_ww = wanf.get_hopping([0, 0, 0])
    hop1_ww = wanf.get_hopping([1, 1, 1])
    for i in range(nwannier):
        assert hop0_ww[i, i] != 0
        assert hop1_ww[i, i] != 0
        assert np.abs(hop1_ww[i, i]) < np.abs(hop0_ww[i, i])
        for j in range(i + 1, nwannier):
            assert hop0_ww[i, j] == 0
            assert hop1_ww[i, j] == 0
            assert hop0_ww[i, j] == hop0_ww[j, i]
            assert hop1_ww[i, j] == hop1_ww[j, i]


def test_get_hopping_random(wan, rng):
    nwannier = 4
    wanf = wan(nwannier=nwannier, initialwannier='random')
    hop0_ww = wanf.get_hopping([0, 0, 0])
    hop1_ww = wanf.get_hopping([1, 1, 1])
    for i in range(nwannier):
        for j in range(i + 1, nwannier):
            assert np.abs(hop0_ww[i, j]) == pytest.approx(np.abs(hop0_ww[j, i]))
            assert np.abs(hop1_ww[i, j]) == pytest.approx(np.abs(hop1_ww[j, i]))


def test_get_hamiltonian_bloch(wan):
    nwannier = 4
    atoms = molecule('H2', pbc=True)
    atoms.center(vacuum=3.)
    kpts = (2, 2, 2)
    Nk = kpts[0] * kpts[1] * kpts[2]
    wanf = wan(atoms=atoms, kpts=kpts,
               nwannier=nwannier, initialwannier='bloch')
    for k in range(Nk):
        H_ww = wanf.get_hamiltonian(k=k)
        for i in range(nwannier):
            assert H_ww[i, i] != 0
            for j in range(i + 1, nwannier):
                assert H_ww[i, j] == 0
                assert H_ww[i, j] == pytest.approx(H_ww[j, i])


def test_get_hamiltonian_random(wan, rng):
    nwannier = 4
    atoms = molecule('H2', pbc=True)
    atoms.center(vacuum=3.)
    kpts = (2, 2, 2)
    Nk = kpts[0] * kpts[1] * kpts[2]
    wanf = wan(atoms=atoms, kpts=kpts, rng=rng,
               nwannier=nwannier, initialwannier='random')
    for k in range(Nk):
        H_ww = wanf.get_hamiltonian(k=k)
        for i in range(nwannier):
            for j in range(i + 1, nwannier):
                assert np.abs(H_ww[i, j]) == pytest.approx(np.abs(H_ww[j, i]))


def test_get_hamiltonian_kpoint(wan, rng, std_calculator):
    nwannier = 4
    calc = std_calculator
    atoms = calc.get_atoms()
    wanf = wan(nwannier=nwannier, initialwannier='random')
    kpts = atoms.cell.bandpath(density=50).cartesian_kpts()
    for kpt_c in kpts:
        H_ww = wanf.get_hamiltonian_kpoint(kpt_c=kpt_c)
        for i in range(nwannier):
            for j in range(i + 1, nwannier):
                assert np.abs(H_ww[i, j]) == pytest.approx(np.abs(H_ww[j, i]))


def test_get_function(wan):
    nwannier = 2
    atoms = molecule('H2', pbc=True)
    atoms.center(vacuum=3.)
    nk = 2
    gpts = np.array([8, 8, 8])
    wanf = wan(atoms=atoms, gpts=gpts, kpts=(nk, nk, nk), rng=rng,
               nwannier=nwannier, initialwannier='bloch')
    assert (wanf.get_function(index=[0, 0]) == 0).all()
    assert wanf.get_function(index=[0, 1]) + wanf.get_function(index=[1, 0]) \
        == pytest.approx(wanf.get_function(index=[1, 1]))
    for i in range(nwannier):
        assert (gpts * nk == wanf.get_function(index=i).shape).all()
        assert (gpts * [1, 2, 3] ==
                wanf.get_function(index=i, repeat=[1, 2, 3]).shape).all()


def test_get_gradients(wan, rng):
    wanf = wan(nwannier=4, fixedstates=2, kpts=(1, 1, 1),
               initialwannier='bloch', std_calc=False)
    # create an anti-hermitian array/matrix
    step = rng.rand(wanf.get_gradients().size) + \
        1.j * rng.rand(wanf.get_gradients().size)
    step *= 1e-8
    step -= dagger(step)
    f1 = wanf.get_functional_value()
    wanf.step(step)
    f2 = wanf.get_functional_value()
    assert (np.abs((f2 - f1) / step).ravel() -
            np.abs(wanf.get_gradients())).max() < 1e-4
