#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from xml.etree import ElementTree as ET
from nomad.datamodel.datamodel import EntryArchive
from nomad.datamodel.metainfo.basesections import Analysis
from nomad.datamodel.metainfo.simulation.calculation import Calculation
from nomad.datamodel.metainfo.simulation.run import Program, Run
from nomad.datamodel.metainfo.simulation.system import System, Atoms, AtomsGroup
from .schema import MolproAnalysisResult
from nomad.units import ureg


class MolproXMLOutParser:
    # TODO: migrate to XML parsing tools
    def find_tags(self, tag_name: str, element=None, results=None) -> list:
        """
        Recursively searches for XML elements with a specific tag name.

        Args:
            tag_name (str): The name of the tag to search for.
            element (Element, optional): The XML element to start the search from. Defaults to the root element.
            results (list, optional): A list to store the matching elements. Defaults to an empty list.

        Returns:
            list: A list of XML elements matching the specified tag name.
        """
        if element is None:
            element = self._root
        if results is None:
            results = []
        # Check if the current element matches the tag_name
        if element.tag.endswith(tag_name):
            results.append(element)
        # Recursively search in child elements
        for child in element:
            self.find_tags(tag_name, child, results)
        return results

    def remove_namespace(self, tree):
        """
        Removes the namespace from the XML tree.

        Parameters:
            tree (ElementTree.Element): The XML tree to remove the namespace from.

        Returns:
            None
        """
        for elem in tree.iter():
            elem.tag = elem.tag.split("}")[-1]

    # TODO: consider storing deeply nested tags upon extraction
    def extracted_atoms(self):
        """
        Returns a list of extracted atoms from the parsed data.

        If the list has not been extracted yet, it will be extracted using the `find_tags` method.

        Returns:
            list: A list of extracted atoms.
        """
        if not hasattr(self, "_extracted_atoms"):
            self._extracted_atoms = self.find_tags("atom")
        return self._extracted_atoms

    @property
    def program(self) -> Program:
        """Parse the program (name, version and compilation information).

        Returns:
            Program: The parsed program object.
        """
        program: Program = Program()
        program.name = "Molpro"
        version_tag = self.find_tags("version")[0]
        try:
            program.version = (
                f'{version_tag.attrib["major"]}.{version_tag.attrib["minor"]}'
            )
            program.version_internal = version_tag.attrib["SHA"]
            # program.compilation_datetime?
        except KeyError:
            self.logger.warning("Could not parse Molpro version information.")
        return program

    @property
    def atoms(self) -> Atoms:
        """Parse the atoms (labels and positions).

        Returns:
            Atoms: An object representing the atoms with labels, positions, and periodicity.
        """
        labels: list[str] = []
        positions: list[list[float]] = []

        if len(self.extracted_atoms()):
            for atom in self.find_tags("atom"):
                labels.append(atom.attrib["elementType"])
                positions.append([float(atom.attrib[f"{x}3"]) for x in ["x", "y", "z"]])

        return Atoms(
            labels=labels, positions=positions * ureg.angstrom, periodic=[False] * 3
        )

    @property
    def all_atoms_group(
        self,
    ) -> AtomsGroup:  # TODO: abstract out to any kind of `System`
        """Parse the atom indices and bonds of the entire system.

        Returns:
            AtomsGroup: An object containing the parsed atom indices and bond list.

        """
        convert_id = lambda x: int(x[1:])  # id-format: "a1" -> 1
        atom_indices: list[int] = []
        bond_list: list[list[int]] = []

        for atom in self.find_tags("atom"):
            atom_indices.append(convert_id(atom.attrib["id"]) - 1)
        for bond in self.find_tags("bond"):
            bond_list.append([convert_id(x) for x in bond.attrib["atomRefs2"].split()])

        return AtomsGroup(atom_indices=atom_indices, bond_list=bond_list, label="all")

    @property
    def user_table_str(self) -> str:
        """Extract user-defined tables.

        Returns:
            str: The user table as an HTML string.
        """
        user_tables = self.find_tags("table")
        if len(user_tables) > 1:
            self.logger.warning("Found more than one user table. Using the first one.")

        user_table = self.find_tags("table")[0]
        self.remove_namespace(user_table)
        user_table.attrib["border"] = "1"
        return ET.tostring(user_table, encoding="unicode", method="html")

    def parse(self, filepath: str, archive: EntryArchive, logger) -> EntryArchive:
        """Build up the archive from pre-defined sections.

        Args:
            filepath (str): The path to the file to be parsed.
            archive (EntryArchive): The archive to be built up.
            logger: The logger object for logging messages.

        Returns:
            EntryArchive: The archive with the parsed data.
        """
        self._root = ET.parse(filepath).getroot()
        self.logger = logger

        archive.run.append(Run())
        sec_run = archive.run[0]

        sec_run.program = self.program
        sec_run.system.append(
            System(
                atoms=self.atoms,
                atoms_group=[self.all_atoms_group],
                is_representative=True,
            )
        )
        sec_run.calculation.append(Calculation())

        archive.data = Analysis(
            name="Molpro output",
            outputs=[
                MolproAnalysisResult(
                    name="User-requested post-analysis",
                    result=self.user_table_str,
                ),
            ],
        )

        return archive
