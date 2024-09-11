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
    corr_energy_strong = Quantity(
        type=np.float32,
        description="""
        Correlation energy contribution for the strong pairs.
        This contribution doesnt involve perturbative corrections!
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    corr_energy_weak = Quantity(
        type=np.float32,
        description="""
        Correlation energy contribution for the weak pairs.
        This contribution doesnt involve perturbative corrections!
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    corr_energy_perturbative = Quantity(
        type=np.float32,
        description="""
        Correlation energy contribution from perturbative treatment.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    t1_norm = Quantity(
        type=np.float32,
        description="""
        The norm of T1 amplitudes.
        Sanity check number 1.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    largest_t2_amplitude = Quantity(
        type=np.float32,
        shape=['*'],
        description="""
        The largest T2 amplitude.
        Sanity check number 2.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    def t1_diagnostic(self, logger) -> None:
        '''Perform a sanity check based on t1 norm.

        Raise a logging error if its larger than 0.02.'''

        if self.t1_norm > 0.02:
            logger.info(
                f'T1 diagnostic warning: T1 norm ({self.t1_norm}) exceeds the 0.02 threshold.'
            )
        else:
            logger.info(
                f'T1 diagnostic passed: T1 norm ({self.t1_norm}) is within the acceptable range.'
            )

    def t2_diagnostic(self, logger) -> None:
        '''Perform a sanity check based on the largest t2 amplitude.
        Log a warning if it's larger than 0.02.
        '''
        if not self.largest_t2_amplitude:
            logger.warning('T2 diagnostic warning: The list of largest T2 amplitudes is empty.')
            return

        max_amplitude = max(self.largest_t2_amplitude)

        if max_amplitude > 0.05:
            logger.info(
                f'T2 diagnostic warning: Largest T2 amplitude ({max_amplitude})'
                f'exceeds the 0.05 threshold. This may indicate a multiconfigurational character!'
            )
        else:
            logger.info(
                f'T2 diagnostic passed: Largest T2 amplitude ({max_amplitude})'
                f'is within the acceptable range.'
            )

    def normalize(self, archive, logger) -> None:
        '''Normalize the coupled-cluster output quantities and run diagnostic checks.

        Log warnings if any diagnostic thresholds are exceeded.
        '''
        super().normalize(archive, logger)  # Call the parent's normalize method

        # Run diagnostic checks
        self.t1_diagnostic(logger)
        self.t2_diagnostic(logger)

