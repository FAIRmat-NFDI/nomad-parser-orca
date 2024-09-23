import numpy as np
import pytest
import xml.etree.ElementTree as ET
from nomad.units import ureg
from ..molproparser.parser import MolproXMLOutParser


@pytest.mark.parametrize(
    "version_major, version_minor, version_sha",
    [
        ("2021", "1", "a9e37c2"),
        ("2022", "2", "b9f47b3"),
    ],
)
def test_program_property(version_major, version_minor, version_sha):
    xml_data = f"""
    <molpro>
        <version major="{version_major}" minor="{version_minor}" SHA="{version_sha}"/>
    </molpro>
    """
    root = ET.fromstring(xml_data)

    parser = MolproXMLOutParser()
    parser._root = root
    program = parser.program

    assert program.name == "Molpro"
    assert program.version == f"{version_major}.{version_minor}"
    assert program.version_internal == version_sha


@pytest.mark.parametrize(
    "atoms",
    [
        (
            {"elem": "C", "pos": [0.0, 0.0, 0.0]},
            {"elem": "H", "pos": [5.0, 4.0, 2.0]},
        ),
    ],
)
def test_atoms_property(atoms):
    xml_data = f"""
    <molpro>
        {" ".join(f'<atom id="a{i+1}" elementType="{atom["elem"]}" x3="{atom["pos"][0]}" y3="{atom["pos"][1]}" z3="{atom["pos"][2]}"/>' for i, atom in enumerate(atoms))}
    </molpro>
    """
    root = ET.fromstring(xml_data)

    parser = MolproXMLOutParser()
    parser._root = root
    parser_atoms = parser.atoms

    assert parser_atoms.labels == [atom["elem"] for atom in atoms]
    assert (
        parser_atoms.positions == [atom["pos"] for atom in atoms] * ureg.angstrom
    ).all()


# Additional parametrized tests for all_atoms_group_property, user_table_str_property, etc.
