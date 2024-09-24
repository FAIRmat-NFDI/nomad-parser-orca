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

class CCOutputs(Outputs):
    """
    This section contains the relevant output information from a Coupled-Cluster run.
    """

    reference_energy = Quantity(
        type=np.float32,
        description="""
        Converged SCF energy of the reference determinant.
        """,
    )

    corr_energy_strong = Quantity(
        type=np.float32,
        description="""
        Correlation energy contribution for the strong pairs.
        This contribution doesnt involve perturbative corrections!
        """,
    )

    corr_energy_weak = Quantity(
        type=np.float32,
        description="""
        Correlation energy contribution for the weak pairs.
        This contribution doesnt involve perturbative corrections!
        """,
    )

    corr_energy_perturbative = Quantity(
        type=np.float32,
        description="""
        Correlation energy contribution from perturbative treatment.
        """,
    )

    t1_norm = Quantity(
        type=np.float32,
        description="""
        The norm of T1 amplitudes.
        """,
    )

    largest_t2_amplitude = Quantity(
        type=np.float32,
        shape=['*'],
        description="""
        The largest T2 amplitude.
        """,
    )

    def normalize(self, archive, logger) -> None:
        '''Normalize the coupled-cluster output quantities
        '''
        super().normalize(archive, logger)  
