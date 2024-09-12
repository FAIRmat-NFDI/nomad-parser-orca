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

class CoupledClusterEnergy(Section):
    method = Quantity(
        type=str,
        description="The name of the energy calculation method (e.g., 'pno_lmp2', 'pno_lmp2_f12')."
    )
    
    domain_correction = Quantity(
        type=np.float32,
        description="Domain correction energy."
    )
    
    correlation = Quantity(
        type=np.float32,
        description="Correlation energy."
    )
    
    total = Quantity(
        type=np.float32,
        description="Total energy."
    )


class CCOutputs(Outputs):
    """
    This section contains the relevant output information from a Coupled-Cluster run.
    """

    # Store general quantities
    reference_energy = Quantity(
        type=np.float32,
        description="""
        The converged energy of the reference determinant. Can be DFT, HF, etc.
        """
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

    # Use a list of MethodEnergy SubSections to store different methods' energies
    energies = SubSection(
        section=CoupledClusterEnergy,
        repeats=True,
        description="A list of energies corresponding to different methods."
    )

    def add_energy(self, method, domain_correction, correlation, total):
        """
        Add energies corresponding to a specific method dynamically.
        """
        energy_entry = MethodEnergy(
            method=method,
            domain_correction=domain_correction,
            correlation=correlation,
            total=total
        )
        self.energies.append(energy_entry)

    def get_energy(self, method, energy_type='total'):
        """
        Retrieve energy for a specific method and type.
        Default is 'total' energy, but can also be 'correlation' or 'domain_correction'.
        """
        if self.energies and method in self.energies:
            return self.energies[method].get(energy_type, None)
        else:
            raise ValueError(f"No energy found for method '{method}' or energy type '{energy_type}'.")

    def energy_diagnostic(self, method, logger) -> None:
        """
        Perform diagnostic checks on the energy values for a given method.
        Log a warning if the energy is significantly high or low (example threshold 1e-5).
        """
        total_energy = self.get_energy(method, 'total')
        if total_energy is None:
            logger.warning(f"No total energy found for method '{method}'.")
            return

        # Perform a diagnostic check (you can change this threshold logic)
        if abs(total_energy) < 1e-5:
            logger.warning(f"Energy diagnostic warning: {method} total energy ({total_energy}) is very small.")
        else:
            logger.info(f"Energy diagnostic passed for method '{method}': Total energy is {total_energy}.")

    def t1_diagnostic(self, logger) -> None:
        '''Perform a sanity check based on t1 norm.'''
        if self.t1_norm > 0.02:
            logger.info(
                f'T1 diagnostic warning: T1 norm ({self.t1_norm}) exceeds the 0.02 threshold.'
            )
        else:
            logger.info(
                f'T1 diagnostic passed: T1 norm ({self.t1_norm}) is within the acceptable range.'
            )

    def t2_diagnostic(self, logger) -> None:
        '''Perform a sanity check based on the largest t2 amplitude.'''
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
        '''Normalize the coupled-cluster output quantities and run diagnostic checks.'''
        super().normalize(archive, logger)  # Call the parent's normalize method

        # Run diagnostic checks
        self.t1_diagnostic(logger)
        self.t2_diagnostic(logger)

        # Perform energy diagnostics for various methods
        if self.energies:
            for method in self.energies:
                self.energy_diagnostic(method, logger)

