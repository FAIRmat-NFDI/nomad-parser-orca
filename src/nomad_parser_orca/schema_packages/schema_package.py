from typing import TYPE_CHECKING

import nomad_simulations.schema_packages
from nomad_simulations.schema_packages.model_method import \
    ModelMethodElectronic
from nomad_simulations.schema_packages.numerical_settings import \
    NumericalSettings
from nomad_simulations.schema_packages.outputs import Outputs
from nomad_simulations.schema_packages.physical_property import PhysicalProperty

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

configuration = config.get_plugin_entry_point(
    'nomad_parser_orca.schema_packages:schema_package_entry_point'
)


m_package = SchemaPackage()

class PerturbationMethod(ModelMethodElectronic):
    type = Quantity(
        type=MEnum('MP', 'RS', 'BW'),
        description="""
        Perturbation approach. The abbreviations stand for:
        | Abbreviation | Description |
        | ------------ | ----------- |
        | `'MP'`       | Moller-Plesset |
        | `'RS'`       | Rayleigh-Schrödigner |
        | `'BW'`       | Brillouin-Wigner |
        """,
        a_eln=ELNAnnotation(component='EnumEditQuantity'),
    )  # TODO: check if the special symbols are supported

    order = Quantity(
        type=np.int32,
        description="""
        Order up to which the perturbation is expanded.
        """,
        a_eln=ELNAnnotation(component='NumberEditQuantity'),
    )

    density = Quantity(
        type = MEnum('relaxed', 'unrelaxed'
        ),
        description="""
        unrelaxed density: MP2 expectation value density
        relaxed density  : incorporates orbital relaxation
        """
    )   

class CoupledCluster(ModelMethodElectronic):
    """
    A base section used to define the parameters of a Coupled Cluster calculation.
    A standard schema is defined, though the most common cases can be summarized in the `type` quantity.
    """

    valid_base_methods = [
        'CC2', 'CC3', 'CC4', 'CCD', 'CCSD', 'CCSDT', 'CCSDTQ',
        'BCCD', 'QCCD', 'VQCCD', 'LCCD', 'LCCSD', 'DLPNO-CCSD',
        'MP2', 'MP3', 'MP4', 'MP5'
    ]

    perturbative_corrections = ['(T)', '(T0)', '(T1)', '[T]', '[T0]',
                                '(Q)', '(2)', '(fT)', '(dT)']
    
    correlation_methods = ['-F12', '-R12']

    solver_prefixes = ['QV', 'B', 'Q']

    type = Quantity(
        type=str,
        description="""
        Coupled Cluster flavor.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )  

    excitation_order = Quantity(
        type=np.int32,
        shape=['*'],
        description="""
        Orders at which the excitation are used.
        1 = single, 2 = double, 3 = triple, 4 = quadruple, etc.
        """
    )

    reference_determinant = Quantity(
        type=MEnum('UHF','RHF','ROHF',
                   'UKS', 'RKS', 'ROKS'),
        description="""
        the type of reference determinant.
        """,
    )
    
    perturbation_method = SubSection(sub_section=PerturbationMethod.m_def)

    perturbative_correction = Quantity(
        type=MEnum('(T)', '[T]', 
                   '(T0)', '[T0]',
                   '(Q)'),
        description="""
        The type of perturbative corrections.
        A perturbative correction is different than a perturbation method.
        """
    ) # TODO: add more perturbative correction

    explicit_correlation = Quantity(
        type=MEnum('F12', 'F12a', 'F12b', 'F12c',
                   'R12', ''),
        default='',
        description="""
        Explicit correlation treatment.
        These methods introduce the interelectronic distance coordinate
        directly into the wavefunction to treat dynamical electron correlation.
        It can be added linearly (R12) or exponentially (F12).
        """,
    )

    is_frozencore = Quantity(
        type=bool,
        description="""
        frozen core approximation
        """,
    )  

    local_approximation = Quantity(
        #type=MEnum('LPNO', 'DLPNO', ''),
        type=str,
        description="""
        Is there a local approximation with pair natural orbitals
        or domain-based pair natural orbitals etc.
        """,
    )

    def dissect_cc_method(self, method_name):
        """
        Parses the Coupled Cluster method string and updates the corresponding attributes of the class.
        This is relevant for some program packages.
        """

        # regex to account for explicit correlation variants and perturbative corrections
        pattern = re.compile(
            r'^(DLPNO|LPNO)?(QV|B)?(CCSD|CCD|CCSDT|LCCSD)(\([TQ2fTdT0]\))?(-F12[a-cx]?)?$'
        )
        match = pattern.match(method_name)

        if not match:
            raise ValueError(f"Invalid Coupled Cluster method: {method_name}")

        # Extract components from the regex match
        self.local_approximation = match.group(1)  # DLPNO or LPNO
        self.solver = match.group(2)  # QV or B solver
        self.type = match.group(3)  # Main CC method (e.g., CCSD, CCD, etc.)
        perturbative_correction = match.group(4)  # Perturbative correction (e.g., (T))
        self.explicit_correlation = match.group(5)  # Explicit correlation (e.g., -F12a, -F12b, etc.)

        # Parse excitation order based on method_type and perturbative_correction
        self.excitation_order = []
        if self.type == 'CCSD':
            self.excitation_order.append(2)  # Double excitation for CCSD
        if perturbative_correction == '(T)':
            self.perturbative_order = [3]  # Triple perturbative correction for (T)

        # Handle other perturbative corrections and excitation orders
        if perturbative_correction in ['(Q)', '(T0)', '(T1)', '(2)', '(fT)', '(dT)']:
            if perturbative_correction == '(Q)':
                self.perturbative_order = [4]  # Quadruple for (Q)
            # Additional cases can be handled similarly as per the specific correction

        # Ensure the lists are sorted for consistency
        if isinstance(self.excitation_order, list):
            self.excitation_order = np.sort(self.excitation_order)
        if isinstance(self.perturbative_order, list):
            self.perturbative_order = np.sort(self.perturbative_order)


    def validate_type(self, logger) -> bool:
        """
        Validate the Coupled Cluster type to ensure it's a valid combination.
        Logs a warning if the input doesn't match expected patterns.
        """
        # Regular expression to match the input
        pattern = re.compile(
            r'^(QV|B)?(CC2|CC3|CC4|CCD|CCSD|CCSDT|CCSDTQ|BCCD|QCCD|VQCCD|LCCD|LCCSD|DLPNO-CCSD)'
            r'((\([TQ2fTdT]\))|(-F12)|(-R12))?$'
        )

        if not pattern.match(self.type):
            logger.warning(f'Invalid Coupled Cluster type: {self.type}. Check method input.')
            return False
        logger.info(f'Valid Coupled Cluster type: {self.type}')
        return True


    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        

m_package.__init_metainfo__()
