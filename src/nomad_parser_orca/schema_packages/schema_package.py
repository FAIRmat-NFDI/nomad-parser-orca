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

configuration = config.get_plugin_entry_point(
    'nomad_parser_orca.schema_packages:schema_package_entry_point'
)


m_package = SchemaPackage()


class CoupledCluster(ModelMethodElectronic):
    order_map = {k: v for k, v in enumerate(('S', 'D', 'T', 'Q'))}
    solver_map = {'QV': 'quasi-variational', 'B': 'Brueckner'}
    map_solver = {v: k for k, v in solver_map.items()}

    valid_base_methods = [
        'CC2', 'CC3', 'CC4', 'CCD', 'CCSD', 'CCSDT', 'CCSDTQ',
        'BCCD', 'QCCD', 'VQCCD', 'LCCD', 'LCCSD', 'DLPNO-CCSD',
        'MP2', 'MP3', 'MP4', 'MP5'
    ]

    # Define valid perturbative corrections and correlation methods
    perturbative_corrections = ['(T)', '(T0)', '(T1)', '[T]', '[T0]',
                                '(Q)', '(2)', '(fT)', '(dT)']
    correlation_methods = ['-F12', '-R12']

    # Solver prefixes
    solver_prefixes = ['QV', 'B', 'Q']

    type = Quantity(
        type=str,
        description="""
        Coupled Cluster flavor.
        Input must be a valid coupled cluster method.
        """,
        a_eln=ELNAnnotation(component='StringEditQuantity'),
    )  # TODO: add important stuff

    excitation_order = Quantity(
        type=np.int32,
        shape=['*'],
        description="""
        Orders at which the excitation are used.
        1 = single, 2 = double, 3 = triple, 4 = quadruple, etc.

        Note that coupled cluster typically start from doubles.
        Singles excitations in a Koopman-compliant scheme only make sense as a response to a perturbation.
        """,
    )

    reference_determinant = Quantity(
        type=MEnum('UHF','RHF','ROHF'
                   'UKS', 'RKS', 'ROKS'),
        description="""
        the type of reference determinant.
        """,
    )

    perturbative_order = Quantity(
        type=np.int32,
        shape=['*'],
        description="""
        Excitation order at which the perturbative correction is used.
        1 = single, 2 = double, 3 = triple, 4 = quadruple, etc.
        """,
    )

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

    ri_approximation = Quantity(
        type=str,
        description="""
        Flavor of RI approximation.
        In MOLPRO, it is denoted as density fitting!
        """,
    ) 

    ri_approximation_f12 = Quantity(
        type=str,
        description="""
        Flavor of RI approximation in explicit correlation.
        """,
    ) 

    is_frozencore = Quantity(
        type=bool,
        description="""
        frozen core approximation
        """,
    )  

    local_approximation = Quantity(
        type=MEnum('LPNO', 'DLPNO', ''),
        description="""
        Is there a local approximation with pair natural orbitals
        or domain-based pair natural orbitals etc.
        """,
    )

    def dissect_cc_method(self, method_name):
        """
        Parses the Coupled Cluster method string and updates the corresponding attributes of the class.
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

        # Optionally log the parsed method components (this requires a logger passed into the function)
        # logger.info(f"Parsed method: {method_name}, local_approximation: {self.local_approximation}, "
        #             f"solver: {self.solver}, type: {self.type}, perturbative_correction: {perturbative_correction}, "
        #             f"explicit_correlation: {self.explicit_correlation}")

    def cc_to_type(self) -> None:
        """Produce an educated guess based on the other parameters."""
        name = 'CC'
        # cover the basic cases
        if 2 in self.excitation_order:
            if 1 in self.excitation_order:
                name += 'SD'
            name += 'D'
        # cover extended excitations
        for order, abbrev in {3: 'T', 4: 'Q'}.items():
            if order in self.excitation_order:
                name += abbrev
            elif order in self.perturbative_order:
                name += f'({abbrev})'
        # cover explicit correlation
        if self.explicit_correlation is not None:
            name += self.explicit_correlation
        # cover specific solver approaches
        if self.solver in self.map_solver:
            name = self.solver_map[self.solver] + name

    def validate_type(self, logger) -> bool:
        """
        Validate the Coupled Cluster type to ensure it's a valid combination.
        Logs a warning if the input doesn't match expected patterns.
        This is from ndaelman
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


    def type_to_cc(self) -> None:
        """Try to extract the excitation and perturbation orders from the type.
        This is also from ndaelman.
        """
        match = re.match(
            r'(QV|B)?CC(S)?(D)?(T|\(T\))?(Q|\(Q\))?(-F12|-R12)?', self.type
        )
        if match is None:
            return

        ptb_initialized, exc_initialized = False, False
        for i in range(2, 6):
            if abbrev := match.group(i):
                order = i - 1
                if abbrev[0] == '(':
                    if not ptb_initialized:
                        self.perturbative_order = []
                        ptb_initialized = True
                    self.perturbative_order = np.append(self.perturbative_order, order)
                else:
                    if not exc_initialized:
                        self.excitation_order = []
                        exc_initialized = True
                    self.excitation_order = np.append(self.excitation_order, order)
        if match.group(1) in self.solver_map:
            self.solver = self.solver_map[match.group(1)]
        if match.group(6):
            self.explicit_correlation = match.group(6)[1:]  # remove the dash

    def normalize(self, archive, logger) -> None:
        super().normalize(archive, logger)
        if self.type is None:
            if self.check_orders(logger):
                self.cc_to_type()
        else:
            self.type_to_cc()
        if isinstance(self.excitation_order, list):
            self.excitation_order = np.sort(self.excitation_order)
        if isinstance(self.perturbative_order, list):
            self.perturbative_order = np.sort(self.perturbative_order)


m_package.__init_metainfo__()
