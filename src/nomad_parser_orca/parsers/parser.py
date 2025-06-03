import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

import numpy as np
import re
from nomad.units import ureg
from nomad.config import config
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.parsing.file_parser import Quantity, Parser
from nomad.parsing.file_parser.mapping_parser import (MetainfoParser,
                                                      TextParser)
from nomad_simulations.schema_packages.general import Program, Simulation
import nomad_parser_orca.schema_packages.schema
from .info_reader import OutReader

configuration = config.get_plugin_entry_point(
    'nomad_parser_orca.parsers:parser_entry_point'
)


def str_to_cartesian_coordinates(val_in):
    val_in_cleaned = [val.replace('>', '') if isinstance(val, str) else val for val in val_in if val != '>']

    if isinstance(val_in_cleaned, list):
        symbols = []
        coordinates = []
        for i in range(0, len(val_in_cleaned), 4):
            symbol = val_in_cleaned[i]
            if isinstance(symbol, str):
                symbol = symbol.replace('>', '')
            symbols.append(symbol)
            coordinates.append(val_in_cleaned[i+1:i+4])
            #print(coordinates)
        coordinates = np.array(coordinates, dtype=float)
        return symbols, coordinates
    else:
        raise ValueError("Expected a list input for cartesian coordinates.")

class OutParser(TextParser):

    def get_program_data(self, source: dict[str, Any]) -> dict[str, Any]:
        return dict(
            program_name = 'ORCA',
            program_version = source.get('program_version'),
        )
    
    def get_atoms(self, source: dict[str, Any]) -> dict[str, Any]:
        """
        Extracts atomic positions and related data from the source and returns a dictionary.
        """
        cartesian_coordinates = source.get('single_point', {}).get('cartesian_coordinates', [])
        if not cartesian_coordinates:
            return {}

        symbols, positions = str_to_cartesian_coordinates(cartesian_coordinates)

        # Create a list of dictionaries for atoms
        atoms = [{'symbol': symbol} for symbol in symbols]

        return dict(
            positions=np.array(positions, dtype=float),
            atoms=atoms
        )

    def get_basis_sets(self, source: dict[str, Any]) -> dict[str, Any]:
        try:
            basis_set_roles = {
                'main_basis_set': {'role': 'orbital', 'key': 'main_basis_set'},
                'aux_c_basis_set': {'role': 'auxiliary_post_hf', 'key': 'auxc_basis_set'},
                'aux_j_basis_set': {'role': 'auxiliary_scf', 'key': 'auxj_basis_set'},
                'aux_jk_basis_set': {'role': 'auxiliary_scf', 'key': 'auxjk_basis_set'},
            }
            basis_set_names = source.get('basis_set_name', {})
            ecp_basis_sets = source.get('ecp_basis_set_name', {}).get('capped_ecp', [])
            basis_sets = []
            for key, info in basis_set_roles.items():
                bs_name = basis_set_names.get(info['key'])
                if bs_name:
                    basis_sets.append({
                        "m_def": "nomad_simulations.schema_packages.basis_set.AtomCenteredBasisSet",
                        'basis_set': bs_name,
                        'type': 'GTO',
                        'role': info['role'],
                    })
            for element, bs in ecp_basis_sets:
                basis_sets.append({
                    "m_def": "nomad_simulations.schema_packages.basis_set.AtomCenteredBasisSet",
                    'basis_set': bs,
                    'type': 'GTO',
                    'role': 'cECP',
                    'species_scope': [element],
                })
            print("get_basis_sets: Extracted basis sets:", basis_sets)
            # Return a dictionary with the key that matches the container field:
            return {'basis_set_components': basis_sets}
        except Exception as e:
            print("Error in get_basis_sets:", e)
            # Return an empty structure so that the mapping doesn't break
            return {'basis_set_components': []}

    def get_dft_data(self, source: dict[str, Any]) -> dict[str, Any]:
        """
        Extracts DFT-related data, including XC functionals and SCF settings.
        """
        dft_data = source.get('single_point', {}).get('self_consistent', {}).get('scf_settings', {})
        xc_functionals = []

        # Exchange functional
        if dft_data.get('exchange_functional'):
            xc_functionals.append({
                'libxc_name': dft_data.get('exchange_functional'),
                'name': 'exchange',
                'weight': dft_data.get('scaling_exchange')
            })

        # Correlation functional
        if dft_data.get('correlation_functional'):
            xc_functionals.append({
                'libxc_name': dft_data.get('correlation_functional'),
                'name': 'correlation',
                'weight': dft_data.get('scaling_correlation')
            })
        #print(xc_functionals)
        return {
            'jacobs_ladder': 'metaGGA', # fix here later
            'xc_functionals': xc_functionals,
            'exact_exchange_mixing_factor': dft_data.get('fraction_hf_exchange'),
        }
        
    def get_numerical_settings(self, source: dict[str, Any]) -> dict[str, Any]:
        scf_convergence = source.get('single_point', {}) \
                                .get('self_consistent', {}) \
                                .get('scf_settings', {})

        if not scf_convergence:
            return {}

        return {
                "n_max_iterations": scf_convergence.get("n_max_iterations", 0),
                "threshold_change": scf_convergence.get("energy_change_tolerance", 1e-8)
        }
    


class ORCAParser(Parser):
    def parse(self, mainfile: str, archive: 'EntryArchive', logger: 'BoundLogger') -> None:

        info_parser = OutParser(text_parser=OutReader())
        info_parser.filepath = mainfile
        

        data_parser = MetainfoParser(data_object=Simulation())
        data_parser.annotation_key = 'info'
        info_parser.convert(data_parser)

    
        archive.data = data_parser.data_object
        self.info_parser = info_parser