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
    )  # TODO: add important stuff


    ri_approximation_f12 = Quantity(
        type=str,
        description="""
        Flavor of RI approximation in explicit correlation.
        """,
    )  # TODO: add important stuff

    is_frozencore = Quantity(
        type=str,
        description="""
        frozen core approximation
        """,
    )  # TODO: add important stuff


    local_approximation = Quantity(
        type=MEnum('LPNO', 'DLPNO', ''),
        description="""
        Is there a local approximation with pair natural orbitals
        or domain-based pair natural orbitals etc.
        """,
    )

    def check_orders(self, logger) -> bool:
        """Perform a sanity check on the excitation and perturbation order.
        Raise a logging error if any inconsistency is found.
        (From ndaelman)
        """
        if self.excitation_order is None:
            logger.warning('`CoupledCluster.excitation_order` is undefined.')
            return False
        if len(self.excitation_order) > 1:
            if 2 not in self.excitation_order:
                logger.error('Coupled Cluster typically starts from doubles.')
                return False
        for order in (3, 4):
            if order in self.excitation_order and order in self.perturbative_order:
                logger.error(
                    f'Order {order} is defined as both excitation and perturbative.'
                )
                return False
        return True

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
