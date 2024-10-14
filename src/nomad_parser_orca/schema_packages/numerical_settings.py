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


class Mesh(NumericalSettings):
    """
    A base section used to define the mesh or space partitioning over which a discrete numerical integration is performed.
    """

    dimensionality = Quantity(
        type=np.int32,
        default=3,
        description="""
        Dimensionality of the mesh: 1, 2, or 3. Defaults to 3.
        """,
    )

    kind = Quantity(
        type=MEnum('equidistant', 'logarithmic', 'tan'),
        shape=['dimensionality'],
        description="""
        Kind of mesh identifying the spacing in each of the dimensions specified by `dimensionality`. It can take the values:

        | Name      | Description                      |
        | --------- | -------------------------------- |
        | `'equidistant'`  | Equidistant grid (also known as 'Newton-Cotes') |
        | `'logarithmic'`  | log distance grid |
        | `'Tan'`  | Non-uniform tan mesh for grids. More dense at low abs values of the points, while less dense for higher values |
        """,
    )

    grid = Quantity(
        type=np.int32,
        shape=['dimensionality'],
        description="""
        Amount of mesh point sampling along each axis.
        """,
    ) 

    n_points = Quantity(
        type=np.int32,
        description="""
        Number of points in the mesh.
        """,
    )

    points = Quantity(
        type=np.complex128,
        shape=['n_points', 'dimensionality'],
        description="""
        List of all the points in the mesh.
        """,
    )

    multiplicities = Quantity(
        type=np.float64,
        shape=['n_points'],
        description="""
        The amount of times the same point reappears. A value larger than 1, typically indicates
        a symmetry operation that was applied to the `Mesh`.
        """,
    )

    pruning = Quantity(
        type=str,
        description="""
        Pruning method applied for reducing the amount of points in the Mesh. This is typically
        used for numerical integration near the core levels in atoms, and it takes the value
        `adaptive`.
        """
    )

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)

        if self.dimensionality not in [1, 2, 3]:
            logger.error('`dimensionality` meshes different than 1, 2, or 3 are not supported.')


class NumericalIntegration(NumericalSettings):
    """
    Numerical integration settings used to resolve the following type of integrals by discrete
    numerical integration:
    
    ```math
    \int_{\vec{r}_a}^{\vec{r}_b} d^3 \vec{r} F(\vec{r}) \approx \sum_{n=a}^{b} w(\vec{r}_n) F(\vec{r}_n)   
    ```

    Here, $F$ can be any type of function which would define the type of rules that can be applied
    to solve such integral (e.g., 1D Gaussian quadrature rule or multi-dimensional `angular` rules like the 
    Lebedev quadrature rule).
    
    These multidimensional integral has a `Mesh` defined over which the integration is performed, i.e., the 
    $\vec{r}_n$ points. 
    """

    coordinate = Quantity(
        type=MEnum('all', 'radial', 'angular'),
        description="""
        Coordinate over which the integration is performed. `all` means the integration is performed in 
        all the space. `radial` and `angular` describe cases where the integration is performed for
        functions which can be splitted into radial and angular distributions (e.g., orbital wavefunctions). 
        """,
    )

    integration_rule = Quantity(
        type=str,  # ? extend to MEnum?
        description="""
        Integration rule used. This can be any 1D Gaussian quadrature rule or multi-dimensional `angular` rules, 
        e.g., Lebedev quadrature rule (see e.g., Becke, Chem. Phys. 88, 2547 (1988)).
        """
    )

    weight_partitioning = Quantity(
        type=str,
        description="""
        Approximation applied to the weight when doing the numerical integration. See e.g., C. W. Murray, N. C. Handy 
        and G. J. Laming, Mol. Phys. 78, 997 (1993).
        """
    )

    mesh = SubSection(sub_section=Mesh.m_def)

    def normalize(self, archive: 'EntryArchive', logger: 'BoundLogger') -> None:
        super().normalize(archive, logger)