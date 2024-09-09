from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

from nomad.config import config
from nomad.parsing.file_parser import Quantity as ParsedQuantity
from nomad.parsing.file_parser import TextParser
from nomad.parsing.parser import MatchingParser
from nomad.units import ureg
from nomad_simulations.schema_packages.general import Program, Simulation
from nomad_simulations.schema_packages.model_system import AtomicCell, ModelSystem

from nomad_parser_pyscf.schema_packages.schema_package import ExtendedAtomsState

configuration = config.get_plugin_entry_point(
    'nomad_parser_pyscf.parsers:parser_entry_point'
)


class LogParser(TextParser):
    def init_quantities(self):
        self._quantities = [
            ParsedQuantity(
                'program_version', r'PySCF *version *([\d\.]+)', repeats=False
            ),
            ParsedQuantity(
                'atoms_information',
                r'\[INPUT\] *\d *([a-zA-Z])+ *([\d\.\-]+) *([\d\.\-]+) *'
                r'([\d\.\-]+) *([a-zA-Z]+)[\d\.\-\s]*[a-zA-Z]* *([\d\.\-]+)',
                repeats=True,
            ),
        ]


class PySCFParser(MatchingParser):
    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        log_parser = LogParser(mainfile=mainfile, logger=logger)

        simulation = Simulation()
        program = Program(name='PySCF', version=log_parser.get('program_version'))
        simulation.program = program

        # Add the `Simulation` activity to the `archive`
        archive.data = simulation

        # Match the atoms information
        atoms_information = log_parser.get('atoms_information', [])
        # Instantiate `ModelSystem` and append it to `simulation`
        model_system = ModelSystem()
        simulation.model_system.append(model_system)
        # Instantiate `AtomicCell` and append it to `ModelSystem`
        atomic_cell = AtomicCell()
        model_system.cell.append(atomic_cell)
        # Instantiate `AtomsState` for each atom, populate these sections with the
        # information, and append them to `AtomicCell`
        positions = []
        for atom in atoms_information:
            try:
                atom_state = ExtendedAtomsState(
                    chemical_symbol=atom[0],
                    magnetic_moment=atom[-1] * ureg('bohr_magneton'),
                )
                atomic_cell.atoms_state.append(atom_state)
                position_unit = {
                    'AA': 'angstrom',
                    'Bohr': 'bohr',
                }
                positions.append(atom[1:4])
            except Exception:
                logger.warning(
                    'Matching `atoms_information` is missing some information.'
                )
        atomic_cell.positions = positions * ureg(position_unit[atom[-2]])
