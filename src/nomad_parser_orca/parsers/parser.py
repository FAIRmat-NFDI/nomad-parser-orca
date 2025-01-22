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


class ORCAParser(Parser):
    def parse(self, mainfile: str, archive: 'EntryArchive', logger: 'BoundLogger') -> None:

        info_parser = OutParser(text_parser=OutReader())
        info_parser.filepath = mainfile
        
        data_parser = MetainfoParser(data_object=Simulation())
        data_parser.annotation_key = 'info'

        info_parser.convert(data_parser)

        archive.data = data_parser.data_object
        self.info_parser = info_parser
