from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

import numpy as np
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

from nomad_simulations.schema_packages.outputs import SCFOutputs
from nomad_parser_orca.schema_packages.schema_package import CoupledCluster
from nomad_parser_orca.schema_packages.outputs import CCOutputs
from nomad_parser_orca.parsers.old_parser import OldOutParser


configuration = config.get_plugin_entry_point(
    'nomad_parser_orca.parsers:parser_entry_point'
)

re_n = r'\r*\n'

class OutParser(TextParser):

    def init_quantities(self):
        self._quantities = [
            ParsedQuantity(
                'program_version', r'EBB2675 Version *([\d\.]+)', repeats=False
            ),
            ParsedQuantity(
                'atoms_information',
                #r'\[INPUT\] *\d *([a-zA-Z])+ *([\d\.\-]+) *([\d\.\-]+) *([\d\.\-]+) *([a-zA-Z]+)[\d\.\-\s]*[a-zA-Z]* *([\d\.\-]+)',
                #r'^ {2}[A-Z][a-z]?\s{1,}\d+\.\d{6}\s{1,}\d+\.\d{6}\s{1,}\d+\.\d{6}$',
                rf'CARTESIAN COORDINATES \(ANGSTROEM\)\s*\-+\s*([\s\S]+?){re_n}{re_n}',
            ),
            #Coupled cluster related quantities start here:
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
                r'\b\d+[ab]->\d+[ab]\b\s+\b\d+[ab]->\d+[ab]\b\s+([-+]?\d*\.\d+)\b',
                repeats=True
            ),
            ParsedQuantity(
                'reference_energy',
                r'E\(0\)\s*\.\.\.\s*(-?[\d\.]+)',
                repeats=False
            ),
            ParsedQuantity(
                'corr_energy_strong',
                r'E\(CORR\)\(strong-pairs\)\s*\.\.\.\s*(-?[\d\.]+)',
                repeats=False
            ),
            ParsedQuantity(
                'corr_energy_weak',
                r'E\(CORR\)\(weak-pairs\)\s*\.\.\.\s*(-?[\d\.]+)',
                repeats=False
            )
        ]



def str_to_cartesian_coordinates(val_in):
    if isinstance(val_in, list):
        symbols = []
        coordinates = []
        for i in range(0, len(val_in), 4):
            symbols.append(val_in[i])
            coordinates.append(val_in[i+1:i+4])
        coordinates = np.array(coordinates, dtype=float)
        return symbols, coordinates
    else:
        raise ValueError("Expected a list input for cartesian coordinates.")

class ORCAParser(MatchingParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_parser = OutParser()
        self.old_out_parser = OldOutParser()

    def parse_atomic_structure(self, out_parser, logger):
        atoms_information = out_parser.get('atoms_information', [])
        if isinstance(atoms_information, list):
            symbols, coordinates = str_to_cartesian_coordinates(atoms_information)
            if len(symbols) == len(coordinates):
                model_system = ModelSystem()
                atomic_cell = AtomicCell()
                for symbol, coord in zip(symbols, coordinates):
                    try:
                        atom_state = AtomsState(chemical_symbol=symbol)
                        atomic_cell.atoms_state.append(atom_state)
                    except Exception as e:
                        logger.warning(f'Error creating AtomsState: {e}')
                atomic_cell.positions = coordinates
                model_system.cell.append(atomic_cell)
                return model_system
            else:
                logger.error('Mismatch between number of symbols and coordinates.')
        else:
            logger.warning("No atoms information found or incorrect format.")
        return None

    def parse_coupled_cluster(self, out_parser, logger):
        cc_type = out_parser.get('coupled_cluster_type')
        if cc_type:
            model_method = CoupledCluster(
                type=cc_type,
                reference_determinant=out_parser.get('cc_reference_wavefunction')
            )
            output = CCOutputs(
                largest_t2_amplitude=out_parser.get('largest_t2_amplitudes'),
                t1_norm=out_parser.get('t1_diagnostic'),
                reference_energy=out_parser.get('reference_energy'),
                corr_energy_strong=out_parser.get('corr_energy_strong'),
                corr_energy_weak=out_parser.get('corr_energy_weak')
            )
            return model_method, output
        logger.warning('No coupled cluster data found.')
        return None, None

    def parse_old_quantities(self, old_out_parser, logger):
        old_quantities = {
            'basis_set_atom_labels': old_out_parser.get('basis_set_atom_labels'),
            'basis_set': old_out_parser.get('basis_set'),
            'basis_set_contracted': old_out_parser.get('basis_set_contracted')
        }
        return old_quantities

    def parse(self, mainfile, archive: 'EntryArchive', logger: 'BoundLogger', child_archives=None):
        # Initialize parsers
        self.out_parser.mainfile = mainfile
        self.out_parser.logger = logger
        self.old_out_parser.mainfile = mainfile
        self.old_out_parser.logger = logger

        # Perform parsing
        self.out_parser.parse()
        self.old_out_parser.parse()

        simulation = Simulation()
        simulation.program = Program(name='EBB2675', version=self.out_parser.get('program_version'))
        archive.data = simulation

        # Parse coordinates
        model_system = self.parse_atomic_structure(self.out_parser, logger)
        if model_system:
            simulation.model_system.append(model_system)

        # Parse coupled cluster data
        model_method, output = self.parse_coupled_cluster(self.out_parser, logger)
        if model_method:
            simulation.model_method.append(model_method)
        if output:
            simulation.outputs.append(output)

        # Parse old output quantities
        old_quantities = self.parse_old_quantities(self.old_out_parser, logger)
        print(old_quantities)