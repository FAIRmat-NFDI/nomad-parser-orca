# src/nomad_parser_orca/parsers/parser.py

import numpy as np
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import EntryArchive
    from structlog.stdlib import BoundLogger

from nomad.config import config
from nomad.parsing.file_parser import Parser
from nomad.parsing.file_parser.mapping_parser import TextParser, MetainfoParser
from nomad_simulations.schema_packages.general import Simulation
from .info_reader import OutReader

configuration = config.get_plugin_entry_point(
    'nomad_parser_orca.parsers:parser_entry_point'
)


def str_to_cartesian_coordinates(val_in):
    """
    Convert ORCA’s flat list of tokens into (symbols[], positions[n×3]).
    """
    cleaned = [v.replace('>', '') if isinstance(v, str) else v
               for v in val_in if v != '>']
    if not isinstance(cleaned, list):
        raise ValueError("Expected a list input for cartesian coordinates.")
    symbols = []
    coords = []
    for i in range(0, len(cleaned), 4):
        sym = cleaned[i]
        if isinstance(sym, str):
            sym = sym.replace('>', '')
        symbols.append(sym)
        coords.append(cleaned[i+1 : i+4])
    return symbols, np.array(coords, dtype=float)


class OutParser(TextParser):

    def get_program_data(self, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": "ORCA",
            "version": source.get("program_version"),
        }

    def get_atoms(self, source: dict[str, Any]) -> dict[str, Any]:
        """
        Build a minimal dictionary for one ModelSystem:
          - "positions": np.ndarray of shape (N,3)
          - "particle_states": list of {"m_def": AtomsState, "chemical_symbol": …}
        """
        cart = source.get("single_point", {}).get("cartesian_coordinates", [])
        if not cart:
            return {}

        symbols, positions = str_to_cartesian_coordinates(cart)

        # For each symbol, include an "m_def" so NOMAD knows to build an AtomsState:
        atom_list = [
            {
                "m_def": "nomad_simulations.schema_packages.atoms_state.AtomsState",
                "chemical_symbol": s
            }
            for s in symbols
        ]

        return {
            "positions": positions,
            "particle_states": atom_list
        }

    def get_dft_data(self, source: dict[str, Any]) -> dict[str, Any]:
        scf = source.get("single_point", {}) \
                    .get("self_consistent", {}) \
                    .get("scf_settings", {})

        if not scf:
            return {}

        xc_list = []
        if scf.get("exchange_functional"):
            xc_list.append({
                "libxc_name": scf["exchange_functional"],
                "name": "exchange",
                "weight": scf.get("scaling_exchange", 1.0)
            })
        if scf.get("correlation_functional"):
            xc_list.append({
                "libxc_name": scf["correlation_functional"],
                "name": "correlation",
                "weight": scf.get("scaling_correlation", 1.0)
            })

        return {
            "jacobs_ladder": "metaGGA",
            "exact_exchange_mixing_factor": scf.get("fraction_hf_exchange"),
            "xc_functionals": xc_list
        }


class ORCAParser(Parser):
    def parse(self, mainfile: str, archive: "EntryArchive", logger: "BoundLogger") -> None:
        # 1) Use TextParser→OutReader to get a raw dict
        info_parser = OutParser(text_parser=OutReader())
        info_parser.filepath = mainfile

        # We only need MetainfoParser to populate info_parser.data (the raw dict)
        info_parser.convert(MetainfoParser(data_object=Simulation(), annotation_key="info"))
        raw = info_parser.data

        # 2) Build the  "program"  block exactly as needed by Simulation:
        program_dict = {
            "name": "ORCA",
            "version": raw.get("program_version")
        }

        # 3) Build one ModelSystem entry if Cartesian coords exist:
        cart = raw.get("single_point", {}).get("cartesian_coordinates", [])
        if cart:
            symbols, coords = str_to_cartesian_coordinates(cart)

            ms_entry = {
                # "positions" → will fill ModelSystem.positions
                "positions": coords,

                # "particle_states" → list of AtomsState‐dicts (each with "m_def")
                "particle_states": [
                    {
                        "m_def": "nomad_simulations.schema_packages.atoms_state.AtomsState",
                        "chemical_symbol": s
                    }
                    for s in symbols
                ]
            }
            ms_list = [ms_entry]
        else:
            ms_list = []

        # 4) Build one DFT/ModelMethod entry if scf exists:
        scf = raw.get("single_point", {}).get("self_consistent", {}).get("scf_settings", {})
        if scf:
            xc_list = []
            if scf.get("exchange_functional"):
                xc_list.append({
                    "libxc_name": scf["exchange_functional"],
                    "name": "exchange",
                    "weight": scf.get("scaling_exchange", 1.0)
                })
            if scf.get("correlation_functional"):
                xc_list.append({
                    "libxc_name": scf["correlation_functional"],
                    "name": "correlation",
                    "weight": scf.get("scaling_correlation", 1.0)
                })

            mm_entry = {
                "jacobs_ladder": "metaGGA",
                "exact_exchange_mixing_factor": scf.get("fraction_hf_exchange"),
                "xc_functionals": xc_list
            }
            mm_list = [mm_entry]
        else:
            mm_list = []

        # 5) Assemble final dict that matches Simulation’s schema exactly:
        full_dict = {
            "program":      program_dict,
            "model_system": ms_list,
            "model_method": mm_list
            # (you can add "outputs", "numerical_settings", etc. here later)
        }
      
        archive.data = Simulation().m_from_dict(full_dict)
        self.info_parser = info_parser
