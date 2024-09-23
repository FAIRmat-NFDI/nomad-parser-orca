import pytest
import numpy as np
from ase.cell import Cell


testcellpar = (2, 3, 4, 50, 60, 70)


@pytest.fixture
def cell():
    return Cell.new(testcellpar)


def test_lengths_angles(cell):
    assert cell.cellpar() == pytest.approx(testcellpar)
    assert cell.lengths() == pytest.approx(testcellpar[:3])
    assert cell.angles() == pytest.approx(testcellpar[3:])


def test_new():
    assert np.array_equal(Cell.new(), np.zeros((3, 3)))
    assert np.array_equal(Cell.new([1, 2, 3]), np.diag([1, 2, 3]))
    assert Cell.new(testcellpar).cellpar() == pytest.approx(testcellpar)
    arr = np.arange(9).reshape(3, 3)
    assert np.array_equal(Cell.new(arr), arr)
    with pytest.raises(ValueError):
        Cell.new([1, 2, 3, 4])


def test_handedness(cell):
    assert cell.handedness == 1
    cell[0] *= -1
    assert cell.handedness == -1
    cell[0] = 0
    assert cell.handedness == 0
