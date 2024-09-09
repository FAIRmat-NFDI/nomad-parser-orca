import nomad_simulations
import numpy as np
from nomad.config import config
from nomad.metainfo import Quantity, SchemaPackage

configuration = config.get_plugin_entry_point(
    'nomad_parser_pyscf.schema_packages:schema_package_entry_point'
)

m_package = SchemaPackage()


class ExtendedAtomsState(nomad_simulations.schema_packages.atoms_state.AtomsState):
    magnetic_moment = Quantity(
        type=np.float64,
        default=0.0,
        unit='bohr_magneton',
        description="""
        Magnetic moment of the atom in Bohr magneton units. This quantity is relevant
        only for spin-polarized calculations.
        """,
    )


m_package.__init_metainfo__()
