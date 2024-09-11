from typing import TYPE_CHECKING

import nomad_simulations.schema_packages
from nomad_simulations.schema_packages.model_method import \
    ModelMethodElectronic
from nomad_simulations.schema_packages.numerical_settings import \
    NumericalSettings
from nomad_simulations.schema_packages.outputs import Outputs

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

from nomad.config import config
from nomad.datamodel.data import Schema
from nomad.datamodel.metainfo.annotations import ELNAnnotation, ELNComponentEnum

import nomad_simulations
import numpy as np
import re
from nomad.metainfo import (
    Quantity,
    SubSection,
    MEnum,
    Section,
    Context,
    SchemaPackage
)

class PairNaturalOrbitalAnsatz(NumericalSettings):
    """ Numerical settings that control pair natural orbitals (PNOs)."""

"""
    t_cut_pairs = Quantity(
        type=np.int32,
        shape=['*'],
        description="""
        the cut-off for pairs
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    t_cut_pno = Quantity(
        type=np.int32,
        shape=['*'],
        description="""
        the threshold which controls how many PNO's are retained.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    t_cut_mkn = Quantity(
        type=np.int32,
        shape=['*'],
        description="""
        controls how large the domains that PNOs expand over.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )
    pass
"""

class LocMet(NumericalSettings):
    """ Numerical settings that control orbital localization."""
    type = Quantity(
        type=MEnum('FB',
                   'PM',
                   'IBO',
                   'IAOIBO',
                   'IAOBOYS'
                   'NEWBOYS'
                   'AHFB'),
        description="""
        Name of the localization method
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    ) # Extend from molpro

    n_max_iterations = Quantity(
        type=np.int32,
        description="""
        Specifies the maximum number of iterations for the orbital localization.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    threshold_change = Quantity(
        type=np.float64,
        description="""
        Specifies the convergence tolerance.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    threshold_core = Quantity(
        type = np.float64,
        description="""
        The Energy window for the first OCC MO to be localized (in a.u.).
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    #NORMALIZE
    pass


