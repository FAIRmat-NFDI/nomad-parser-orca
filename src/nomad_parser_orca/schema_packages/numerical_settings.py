from typing import TYPE_CHECKING

import nomad_simulations.schema_packages
from nomad_simulations.schema_packages.model_method import \
    ModelMethodElectronic
from nomad_simulations.schema_packages.numerical_settings import \
    NumericalSettings, SelfConsistency, Mesh
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



class PNOSettings(NumericalSettings):
    """ Numerical settings that control pair natural orbitals (PNOs).
        The nomenclature has been adapted from Molpro.
    """
    domain_connectivity = Quantity(
         type=int,
         description="""
        the connectivity parameter for domain extension.
        """
    )

    domain_radius = Quantity(
         type=int,
         description="""
        the radius parameter for domain extension.
        """
    )

    t_domain_osv_occ = Quantity(
         type=np.float32,
         description="""
        OSV domain occupation number threshold.
        """
    )

    t_occ_lmp2 = Quantity(
         type=np.float32,
         description="""
        LMP2 PNO domains (occ. number threshold).
        """
    )

    t_energy_lmp2 = Quantity(
         type=np.float32,
         description="""
        LMP2 PNO domains (energy threshold).
        """
    )

    t_occ_lccsd = Quantity(
         type=np.float32,
         description="""
        LCCSD PNO domains (occ. number threshold).
        """
    )

    t_energy_lccsd = Quantity(
         type=np.float32,
         description="""
        LCCSD PNO domains (energy threshold).
        """
    )

    t_close_pair = Quantity(
         type=str,
         description="""
        close pair energy threshold.
        """
    )

    t_weak_pair = Quantity(
         type=np.float32,
         description="""
        weak pair energy threshold.
        """
    )

    t_distant_pair = Quantity(
         type=np.float32,
         description="""
        distant pair energy threshold
        """
    )

    t_verydistant_pair = Quantity(
         type=np.float32,
         description="""
        very distant pair energy threshold
        """
    )

    t_triples_preselection = Quantity(
         type=np.float32,
         description="""
        preselection of triples list.
        """
    )

    t_triples_iteration = Quantity(
         type=np.float32,
         description="""
        selection of triples for iterations
        """
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)


class Localization(SelfConsistency):
    """ Numerical settings that control orbital localization."""
    type = Quantity(
        #type=MEnum('FB',
        #           'PM',
        #           'IBO',
        #           'IAOIBO',
        #           'IAOBOYS'
        #           'NEWBOYS'
        #           'AHFB'),
        type=str,
        description="""
        Name of the localization method
        """,
    )

    orbital_window = Quantity(
        shape=['*'],
        description="""
        the Molecular orbital number of the orbitals to be localized.
        """
    )

    # TODO : add more method-dependent quantities


    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

class IntegrationGrid(Mesh):
    """Settings for integration grids.
    The integration grids can be different for different atoms!
    """

    type = Quantity(
        type=str,
        description= """ type of the grid """,
    ) # TODO: an MEnum for COSX or XC grids?

    angular_scheme = Quantity(
        type=str,  
        description=""" 
        the angular quadrature scheme.
        Most popular is Lebedev: A. D. Becke, J. Chem. Phys. 88, 2547 (1988), which provides angular grids of octahedral symmetry.

        Gauss-Legendre: y C. W. Murray, N. C. Handy and G. J. Laming, Mol. Phys. 78, 997 (1993)
        """,
    )

    radial_scheme = Quantity(
        type=str,
        description=""" 
        the radial quadrature scheme.
        """,
    )

    atom_partitioning = Quantity(
        type=str,
        description=""" 
        Weight generation scheme, also known as Voronoi scheme, typically Becke.
        """,
    )

    pruning_method = Quantity(
        type=str,
        description=""" 
        Angular grid pruning method
        """,
    )

