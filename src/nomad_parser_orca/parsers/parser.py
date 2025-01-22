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
    



class ORCAParser(Parser):
    def parse(self, mainfile: str, archive: 'EntryArchive', logger: 'BoundLogger') -> None:

        info_parser = OutParser(text_parser=OutReader())
        info_parser.filepath = mainfile
        
        # Initialize the data parser with the Simulation object
        data_parser = MetainfoParser(data_object=Simulation())
        data_parser.annotation_key = 'info'
        info_parser.convert(data_parser)

        # # Extract atomic data
        # atomic_data = info_parser.get_atoms(info_parser.data)
        # if atomic_data:
        #     try:
        #         # Ensure model_system exists
        #         if not archive.data.model_system:
        #             logger.error("No model_system section in the archive.")
        #             return

        #         # Access the first AtomicCell section
        #         model_system = archive.data.model_system[0]
        #         if not model_system.cell:
        #             logger.error("No AtomicCell section in model_system.")
        #             return

        #         atomic_cell = model_system.cell[0]

        #         # Populate the AtomicCell with parsed data
        #         atomic_cell.positions = atomic_data['positions']
        #         atomic_cell.atoms_state = [
        #             {'chemical_symbol': atom['symbol']} for atom in atomic_data['atoms']
        #         ]

        #     except Exception as e:
        #         logger.warning(f"Failed to populate AtomicCell: {e}")

     
        archive.data = data_parser.data_object
        self.info_parser = info_parser
