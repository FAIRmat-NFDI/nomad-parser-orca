from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

from nomad.config import config
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.parsing.file_parser import Quantity as ParsedQuantity
from nomad.parsing.file_parser import TextParser
from nomad.parsing.parser import MatchingParser
from nomad.units import ureg
from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.general import Program, Simulation
from nomad_simulations.schema_packages.model_method import ModelMethod
from nomad_simulations.schema_packages.model_system import (AtomicCell,
                                                            ModelSystem)

from nomad_parser_orca.schema_packages.schema_package import (CCOutputs,
                                                              CoupledCluster)

configuration = config.get_plugin_entry_point(
    'nomad_parser_orca.parsers:parser_entry_point'
)


class LogParser(TextParser):
    def init_quantities(self):
        self._quantities = [
            ParsedQuantity(
                'program_version', r'EBB2675 Version *([\d\.]+)', repeats=False
            ),
            ParsedQuantity(
                'atoms_information',
                r'\[INPUT\] *\d *([a-zA-Z])+ *([\d\.\-]+) *([\d\.\-]+) *([\d\.\-]+) *([a-zA-Z]+)[\d\.\-\s]*[a-zA-Z]* *([\d\.\-]+)',
                repeats=True
            ),
            ParsedQuantity(
                'coupled_cluster_type',
                r'Correlation treatment\s+\.\.\.\s+([A-Z]+)',
                repeats=False
            ),
            ParsedQuantity(
                'cc_reference_wavefunction',
                r'Reference Wavefunction\s+\.\.\.\s+([A-Z]+)',
                repeats=False
            ),
            ParsedQuantity(
                't1_diagnostic',
            r'T1 diagnostic\s+\.\.\.\s+([\d\.]+)',
            ),
            ParsedQuantity(
                'largest_t2_amplitudes',
                #r'(\d+[ab]->\d+[ab])\s+((?:\d+[ab]->\d+[ab])|(?:-1[ab]-> -1[ab]))\s+([-+]?\d*\.\d+)',
                #r'^\s*\d+[ab]->\d+[ab]\s+\d+[ab]->\d+[ab]\s+([-+]?\d*\.\d+)\s*$',
                r'\b\d+[ab]->\d+[ab]\b\s+\b\d+[ab]->\d+[ab]\b\s+([-+]?\d*\.\d+)\b',
                #r'^\s*\d+[ab]->\d+[ab](?:\s+\d+[ab]->\d+[ab])?(?:\s+-1[ab]->\s*-1[ab])?\s+([-+]?\d*\.\d+)\s*$',
                #r'^\s*\d+[ab]->\d+[ab]\s+(?:\d+[ab]->\d+[ab]\s+)?(?:-1[ab]->\s*-1[ab]\s+)?([-+]?\d*\.\d+)\s*$',
                repeats=True
            )
        ]


class ORCAParser(MatchingParser):
    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        log_parser = LogParser(mainfile=mainfile, logger=logger)

        simulation = Simulation()
        program = Program(
            name='EBB2675', version=log_parser.get('program_version')
        )
        simulation.program = program

        #Add the simulation activity to the archive
        archive.data = simulation

        atoms_information = log_parser.get('atoms_information', [])
        #Instantiate ModelSystem and append it to simulation
        model_system = ModelSystem()
        simulation.model_system.append(model_system)
        #Instantiate AtomicCell and append it to ModelSystem
        atomic_cell = AtomicCell()
        model_system.cell.append(atomic_cell)
        # Instantiate `AtomsState` for each atom, populate these sections with the information, and append them to `AtomicCell`
        positions = []

        for atom in atoms_information:
            try:
                atom_state = AtomsState(
                    chemical_symbol=atom[0]
                )
                atomic_cell.atoms_state.append(atom_state)
                position_unit = {
                    'AA': 'angstrom',
                    'Bohr': 'bohr',
                }
                positions.append(atom[1:4])
            except Exception:
                logger.warning('Matching `atoms_information` is missing some information.')
        if not positions:
            raise ValueError("No atomic positions found. Ensure that the input data contains valid atomic coordinates.")
        atomic_cell.positions = positions * ureg('angstrom')

        cc_type = log_parser.get('coupled_cluster_type')
        cc_reference_wavefunction = log_parser.get('cc_reference_wavefunction')
        t1_diagnostic = log_parser.get('t1_diagnostic')
        largest_t2_amplitudes = log_parser.get('largest_t2_amplitudes')

        model_method = CoupledCluster(type=cc_type, reference_determinant=cc_reference_wavefunction)
        output = CCOutputs(largest_t2_amplitude=largest_t2_amplitudes, t1_norm=t1_diagnostic)

        simulation.model_method.append(model_method)
        simulation.outputs.append(output)
