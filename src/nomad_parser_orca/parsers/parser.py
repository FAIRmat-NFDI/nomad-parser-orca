from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

import numpy as np
import re
from nomad.config import config
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.parsing.file_parser import Quantity,  TextParser
from nomad.parsing.parser import MatchingParser
from nomad.units import ureg
from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.general import Program, Simulation
from nomad_simulations.schema_packages.model_method import ModelMethod
from nomad_simulations.schema_packages.model_system import (AtomicCell,
                                                            ModelSystem)
from nomad_simulations.schema_packages.basis_set import AtomCenteredBasisSet, AtomCenteredFunction
from nomad_simulations.schema_packages.numerical_settings import SelfConsistency
from nomad_simulations.schema_packages.outputs import Outputs

from nomad_parser_orca.schema_packages.schema_package import CoupledCluster
from nomad_parser_orca.schema_packages.numerical_settings import PNOSettings, Localization
from nomad_parser_orca.schema_packages.outputs import CCOutputs


configuration = config.get_plugin_entry_point(
    'nomad_parser_orca.parsers:parser_entry_point'
)

class OutParser(TextParser):

    def init_quantities(self):
        re_float = r'[-+]?\d+\.?\d*(?:[Ee][-+]\d+)?'
        re_n = r'\r*\n'
        self._energy_mapping = {
            'Total Energy': 'energy_total',
            'Nuclear Repulsion': 'energy_nuclear_repulsion',
            'Electronic Energy': 'elec_energy',
            'One Electron Energy': 'one_elec_energy',
            'Two Electron Energy': 'two_elec_energy',
            'Potential Energy': 'potential_energy',
            'Kinetic Energy': 'energy_kinetic_electronic',
            r'E\(X\)': 'energy_exchange',
            r'E\(C\)': 'energy_correlation',
            r'E\(XC\)': 'energy_XC',
        }

        self._timing_mapping = {
            'Total time': 'final_time',
            'Sum of individual times': 'sum_individual_times',
            'Fock matrix formation': 'fock_matrix_formation',
            'Coulomb formation': 'coulomb_formation',
            r'Split\-RI-J': 'split_rj',
            'XC integration': 'xc_integration',
            r'Basis function eval\.': 'basis_fn_evaluation',
            r'Density eval\.': 'density_evaluation',
            r'XC\-Functional eval\.': 'xc_functional_evaluation',
            r'XC\-Potential eval\.': 'potential_evaluation',
            'Diagonalization': 'diagonalization',
            'Density matrix formation': 'density_matrix_formation',
            'Population analysis': 'population_analysis',
            'Initial guess': 'initial_guess',
            'Orbital Transformation': 'orbital_transformation',
            'Orbital Orthonormalization': 'orbital_orthonormalization',
            'DIIS solution': 'diis_solution',
            'Grid generation': 'grid_generation',
            'Total SCF gradient time': 'scf_gradient',
        }

        # Coupled cluster related quantities
        coupled_cluster_quantities = [
            Quantity(
                'coupled_cluster_type',
                r'Correlation treatment\s+\.\.\.\s+([A-Z]+)',
                repeats=False,
            ),
            Quantity(
                'cc_reference_wavefunction',
                r'Reference Wavefunction\s+\.\.\.\s+([A-Z]+)',
                repeats=False,
            ),
            Quantity(
                't1_diagnostic',
                r'T1 diagnostic\s+\.\.\.\s+([\d\.]+)',
                repeats=False,
            ),
            Quantity(
                'largest_t2_amplitudes',
                #r'\b\d+[ab]->\d+[ab]\b\s+\b\d+[ab]->\d+[ab]\b\s+([-+]?\d*\.\d+)\b',
                #r'\b\d+[ab]?->\d+[ab]?\s+\d+[ab]?->\d+[ab]?\s+([-+]?\d*\.\d+)', 
                #r'(\d+[ab]?->\d+[ab]?)\s+(\d+[ab]?->\d+[ab]?)\s+([-+]?\d*\.\d+)',
                #r'\b\d+[ab]?->\d+[ab]?\s+\d+[ab]?->\d+[ab]?\s+([-+]?\d*\.\d+)', 
                #r'\b(\d+[ab]?->\d+[ab]?)\s+(\d+[ab]?->\d+[ab]?)\s+([-+]?\d*\.\d+)\b',
                r'\b\d+[ab]?\s*->\s*\d+[ab]?\s+\d+[ab]?\s*->\s*\d+[ab]?\s+([-+]?\d*\.\d+)', 
                repeats=True
            ), # TODO: find regex that is suitable for all 4 types
            Quantity(
                'reference_energy',
                r'E\(0\)\s*\.\.\.\s*(-?[\d\.]+)',
                repeats=False,
            ),
            Quantity(
                'corr_energy_strong',
                r'E\(CORR\)\(strong-pairs\)\s*\.\.\.\s*(-?[\d\.]+)',
                repeats=False,
            ),
            Quantity(
                'corr_energy_weak',
                r'E\(CORR\)\(weak-pairs\)\s*\.\.\.\s*(-?[\d\.]+)',
                repeats=False,
            ),
            Quantity(
                'tCutPairs',
                r'TCutPairs\s*=\s*([-+]?\d*\.\d+([eE][-+]?\d+)?)',
                repeats=False,
            ),
            Quantity(
                'perturbative_triple_excitations_on_off',
                r'Perturbative triple excitations\s*\.+\s*(\S+)',
                convert=False,
            ),
            Quantity(
                'f12_correction_on_off',
                r'Calculation of F12 correction\s*\.+\s*(\S+)',
                convert=False,
            ),
            Quantity(
                'kc_formation',
                r'K\(C\) Formation\s*\.+\s*(\S+)',
                convert=False,
            )
        ]

        # Basis set related quantities
        basis_set_quantities = [
            Quantity('basis_set_atom_labels', r'Type\s*(\w+)', repeats=True),
            Quantity('basis_set', r':\s*(\w+)\s*contracted\s*to', repeats=True),
            Quantity('basis_set_contracted', r'(\w+)\s*pattern', repeats=True) ]
        
        basis_set_naming_quantities = [
            Quantity(
                'main_basis_set',
                #r'Your calculation utilizes the basis:\s*(.*)',
                r'Your calculation utilizes the basis:\s*([^\s].*?)[\s]*\n',
                repeats=False, convert=False, flatten=False
            ),
            Quantity(
                'auxj_basis_set',
                r'----- AuxJ basis set information -----\s*Your calculation utilizes the auxiliary basis:\s*([^\s].*?)(?:\s*\n|$)', 
                repeats=False, convert=False, flatten=False
            ),
            Quantity(
                'auxjk_basis_set',
                r'----- AuxJK basis set information -----\s*Your calculation utilizes the auxiliary basis:\s*([^\s].*?)(?:\s*\n|$)', 
                repeats=False, convert=False, flatten=False
            ),
            Quantity(
                'auxc_basis_set',
                r'----- AuxC basis set information -----\s*Your calculation utilizes the auxiliary basis:\s*([^\s].*?)(?:\s*\n|$)',
                repeats=False, convert=False, flatten=False
            ),
        ]

        # Basis set statistics quantities
        basis_set_statistics_quantities = [
            Quantity(
                'nb_of_primitive_gaussian_shells',
                r'# of primitive gaussian shells\s*\.+\s*(\d+)',
                repeats=True,
                dtype=int,
            ),
            Quantity(
                'nb_of_primitive_gaussian_functions',
                r'# of primitive gaussian functions\s*\.+\s*(\d+)',
                repeats=True,
                dtype=int,
            ),
            Quantity(
                'nb_of_contracted_shells',
                r'# of contracted shells\s*\.+\s*(\d+)',
                repeats=True,
                dtype=int,
            ),
            Quantity(
                'nb_of_contracted_basis_functions',
                r'# of contracted (?:aux-)?basis functions\s*\.+\s*(\d+)',
                repeats=True,
                dtype=int,
            ),
            Quantity(
                'highest_angular_moment',
                r'Highest angular momentum\s*\.+\s*(\d+)',
                repeats=True,
                dtype=int,
            ),
            Quantity(
                'maximum_contraction_depth',
                r'Maximum contraction depth\s*\.+\s*(\d+)',
                repeats=True,
                dtype=int,
            ),
        ]

        # Grid related quantities
        grid_quantities = [
            Quantity(
                'gral_integ_accuracy',
                rf'General Integration Accuracy\s*IntAcc\s*\.+\s*({re_float})',
                dtype=float,
            ),
            Quantity(
                'radial_grid_type',
                r'Radial Grid Type\s*RadialGrid\s*\.+\s*(\S+)',
                convert=False,
            ),
            Quantity(
                'angular_grid',
                r'Angular Grid \(max\. acc\.\)\s*AngularGrid\s*\.+\s*(\S+)',
                convert=False,
            ),
            Quantity(
                'grid_pruning_method',
                r'Angular grid pruning method\s*GridPruning\s*\.+\s*(.+)',
                flatten=False,
                convert=False,
            ),
            Quantity(
                'weight_gener_scheme',
                r'Weight generation scheme\s*WeightScheme\s*\.+\s*(\w+)',
                convert=False,
            ),
            Quantity(
                'basis_fn_cutoff',
                rf'Basis function cutoff\s*BFCut\s*\.+\s*({re_float})',
                dtype=float,
            ),
            Quantity(
                'integr_weight_cutoff',
                rf'Integration weight cutoff\s*WCut\s*\.+\s*({re_float})',
                dtype=float,
            ),
            Quantity(
                'nb_grid_pts_after_initial_pruning',
                r'# of grid points \(after initial pruning\)\s*\.+\s*(\d+)',
                dtype=int,
            ),
            Quantity(
                'nb_grid_pts_after_weights_screening',
                r'# of grid points \(after weights\+screening\)\s*\.+\s*(\d+)',
                dtype=int,
            ),
            Quantity(
                'total_nb_grid_pts',
                r'Total number of grid points\s*\.+\s*(\d+)',
                dtype=int,
            ),
            Quantity(
                'total_nb_batches', r'Total number of batches\s*\.+\s*(\d+)', dtype=int
            ),
            Quantity(
                'avg_nb_points_per_batch',
                r'Average number of points per batch\s*\.+\s*(\d+)',
                dtype=int,
            ),
            Quantity(
                'avg_nb_grid_pts_per_atom',
                r'Average number of grid points per atom\s*\.+\s*(\d+)',
                dtype=int,
            ),
        ]

        # SCF convergence quantities
        scf_convergence_quantities = [
            Quantity(
                name.lower().replace(' ', '_').replace('-', '_'),
                rf'%s\s*\.+\s*({re_float})\s* Tolerance :\s*({re_float})' % name,
                dtype=float,
            )
            for name in [
                'Last Energy change',
                'Last MAX-Density change',
                'Last RMS-Density change',
            ]
        ]

        # Population analysis quantities
        population_quantities = [
            Quantity(
                'atomic_charges',
                r'[A-Z]+ ATOMIC CHARGES.*\n\-+([\s\S]+?)\-{10}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity('species', r'\n *\d+\s*(\w+)', repeats=True),
                        Quantity(
                            'charge', rf':\s*({re_float})', repeats=True, dtype=float
                        ),
                        Quantity(
                            'total_charge',
                            rf'Sum of atomic charges\s*:\s*({re_float})',
                            dtype=float,
                        ),
                    ]
                ),
            ),
            Quantity(
                'orbital_charges',
                rf'[A-Z]+ REDUCED ORBITAL CHARGES.*\s*\-+([\s\S]+?{re_n}{re_n})',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'atom',
                            r'([A-Z][a-z]?\s*[spdf][\s\S]+?)\n *(?:\d|\Z)',
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity(
                                        'species', r'([A-Z][a-z]?)', convert=False
                                    ),
                                    Quantity(
                                        'charge',
                                        rf'([spdf]\S*)\s*:\s*({re_float})',
                                        repeats=True,
                                    ),
                                ]
                            ),
                        )
                    ]
                ),
            ),
        ]

        # scf quantities
        self_consistent_quantities = [
            Quantity(
                'scf_settings',
                r'SCF SETTINGS\s*\-+([\s\S]+?)\-{12}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'n_max_iterations',
                            rf'Maximum # iterations\s*MaxIter\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                        Quantity(
                            'XC_functional_type',
                            r'Ab initio Hamiltonian\s*Method\s*\.+\s*(\S+)',
                            convert=False,
                        ),
                        Quantity(
                            'XC_functional_type',
                            r'Density Functional\s*Method\s*\.+\s*(\S+)',
                            convert=False,
                        ),
                        Quantity(
                            'exchange_functional',
                            r'Exchange Functional\s*Exchange\s*\.+\s*(\S+)',
                            convert=False,
                        ),
                        Quantity(
                            'xalpha_param',
                            rf'X-Alpha parameter\s*XAlpha\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                        Quantity(
                            'beckes_beta_param',
                            rf'Becke\'s b parameter\s*XBeta\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                        Quantity(
                            'correl_functional',
                            r'Correlation Functional Correlation\s*\.+\s*(\S+)',
                            convert=False,
                        ),
                        Quantity(
                            'lda_part_of_gga_corr',
                            r'LDA part of GGA corr\.\s*LDAOpt\s*\.+\s*(\S+)',
                            convert=False,
                        ),
                        Quantity(
                            'scalar_relativistic_method',
                            r'Scalar relativistic method\s*\.+\s*(\w+)',
                            convert=False,
                        ),
                        Quantity(
                            'speed_of_light_used',
                            rf'Speed of light used\s*Velit\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                        Quantity(
                            'hf_type',
                            r'Hartree-Fock type\s*HFTyp\s*\.+\s*(\w+)',
                            convert=False,
                        ),
                        Quantity(
                            'total_charge',
                            rf'Total Charge\s*Charge\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                        Quantity(
                            'multiplicity',
                            rf'Multiplicity\s*Mult\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                        Quantity(
                            'nelectrons',
                            rf'Number of Electrons\s*NEL\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                        Quantity(
                            'nuclear_repulsion',
                            rf'Nuclear Repulsion\s*ENuc\s*\.+\s*({re_float})',
                            dtype=float,
                            unit=ureg.hartree,
                        ),
                        Quantity(
                            'convergence_check_mode',
                            r'Convergence Check Mode ConvCheckMode\s*\.+\s*(\S+)',
                            convert=False,
                        ),
                        Quantity(
                            'energy_change_tolerance',
                            rf'Energy Change\s*TolE\s*\.+\s*({re_float})',
                            dtype=float,
                            unit=ureg.hartree,
                        ),
                        Quantity(
                            '1_elect_energy_change',
                            rf'1\-El\. energy change\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                    ]
                ),
            ),
            Quantity(
                'dft_grid_generation',
                r'DFT GRID GENERATION\s*\-+([\s\S]+?\-{10})',
                sub_parser=TextParser(quantities=grid_quantities),
            ),
            Quantity(
                'scf_iterations',
                r'SCF ITERATIONS\s*\-+([\s\S]+?)\*{10}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'energy',
                            rf'\n *\d+\s*({re_float})\s*{re_float}',
                            repeats=True,
                            dtype=float,
                            unit=ureg.hartree,
                        )
                    ]
                ),
            ),
            Quantity(
                'final_grid',
                r'Setting up the final grid:([\s\S]+?)\-{10}',
                sub_parser=TextParser(quantities=grid_quantities),
            ),
            Quantity(
                'total_scf_energy',
                r'TOTAL SCF ENERGY\s*\-+([\s\S]+?)\-{10}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            name,
                            rf'%s\s*:\s*({re_float})' % key,
                            dtype=float,
                            unit=ureg.hartree,
                        )
                        for key, name in self._energy_mapping.items()
                    ]
                    + [
                        Quantity(
                            'virial_ratio',
                            rf'Virial Ratio\s*:\s*({re_float})',
                            dtype=float,
                        ),
                        Quantity(
                            'nb_elect_alpha_channel',
                            rf'N\(Alpha\)\s*:\s*({re_float})',
                            dtype=float,
                        ),
                        Quantity(
                            'nb_elect_beta_channel',
                            rf'N\(Beta\)\s*:\s*({re_float})',
                            dtype=float,
                        ),
                        Quantity(
                            'nb_elect_total',
                            rf'N\(Total\)\s*:\s*({re_float})',
                            dtype=float,
                        ),
                    ]
                ),
            ),
            Quantity(
                'scf_convergence',
                r'SCF CONVERGENCE\s*\-+([\s\S]+?)\-{10}',
                sub_parser=TextParser(quantities=scf_convergence_quantities),
            ),
            Quantity(
                'orbital_energies',
                rf'NO\s*OCC\s*E\(Eh\)\s*E\(eV\)\s*([\s\S]+?){re_n}{re_n}',
                str_operation=lambda x: np.array(
                    [v.split()[:4] for v in x.split('\n')], dtype=float
                ),
                repeats=True,
            ),
            Quantity(
                'mulliken',
                r'MULLIKEN POPULATION ANALYSIS \*\s*\*+([\s\S]+?)\*{10}',
                sub_parser=TextParser(quantities=population_quantities),
            ),
            Quantity(
                'timings',
                r'\n *TIMINGS\s*\-+\s*([\s\S]+?)\-{10}',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            name,
                            rf'%s\s*\.+\s*({re_float})' % key,
                            dtype=float,
                            unit=ureg.s,
                        )
                        for key, name in self._timing_mapping.items()
                    ]
                ),
            ),
            Quantity(
                'time_calculation',
                r'Total SCF time\: (\d+) days (\d+) hours (\d+) min (\d+) sec ',
                dtype=np.dtype(np.float64),
            ),
        ]

        # TODO parse more properties, add to metainfo
        tddft_quantities = [
            Quantity(
                'absorption_spectrum_electric',
                r'ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS\s*'
                r'\-+[\s\S]+?\-+\n([\s\S]+?)\-{10}',
                str_operation=lambda x: [v.split() for v in x.strip().split('\n')],
            )
        ]

        def str_to_iteration_energy(val_in):
            val = [v.split() for v in val_in.strip().split('\n')]
            keys = val[0]
            val = np.transpose(
                np.array([v for v in val[1:] if len(v) == len(keys)], dtype=float)
            )
            return {keys[i]: val[i] for i in range(len(keys))}

        ci_quantities = [
            Quantity(
                'electronic_structure_method',
                r'Correlation treatment\s*\.+\s*(\S+)',
                convert=False,
            ),
            Quantity(
                'single_excitations_on_off',
                r'Single excitations\s*\.+\s*(\S+)',
                convert=False,
            ),
            Quantity(
                'orbital_opt_on_off',
                r'Orbital optimization\s*\.+\s*(\S+)',
                convert=False,
            ),
            Quantity(
                'z_vector_calc_on_off',
                r'Calculation of Z vector\s*\.+\s*(\S+)',
                convert=False,
            ),
            Quantity(
                'Brueckner_orbitals_calc_on_off',
                r'Calculation of Brueckner orbitals\s*\.+\s*(\S+)',
                convert=False,
            ),
            Quantity(
                'perturbative_triple_excitations_on_off',
                r'Perturbative triple excitations\s*\.+\s*(\S+)',
                convert=False,
            ),
            Quantity(
                'f12_correction_on_off',
                r'Calculation of F12 correction\s*\.+\s*(\S+)',
                convert=False,
            ),
            Quantity(
                'frozen_core_treatment',
                r'Frozen core treatment\s*\.+\s*(.+)',
                flatten=False,
                convert=False,
            ),
            Quantity(
                'reference_wave_function',
                r'Reference Wavefunction\s*\.+\s*(.+)',
                flatten=False,
                convert=False,
            ),
            Quantity(
                'nb_of_atomic_orbitals', r'Number of AO\'s\s*\.+\s*(\d+)', dtype=int
            ),
            Quantity(
                'nb_of_electrons', r'Number of electrons\s*\.+\s*(\d+)', dtype=int
            ),
            Quantity(
                'nb_of_correlated_electrons',
                r'Number of correlated electrons\s*\.+\s*(\d+)',
                dtype=int,
            ),
            Quantity(
                'integral_transformation',
                r'Integral transformation\s*\.+\s*(.+)',
                flatten=False,
                convert=False,
            ),
            Quantity(
                'level_shift_amplitude_update',
                rf'Level shift for amplitude update\s*\.+\s*({re_float})',
                dtype=float,
            ),
            Quantity(
                'coulomb_transformation_type',
                r'Transformation type\s*\.+\s*(.+)',
                flatten=False,
                convert=False,
            ),
            Quantity(
                'coulomb_transformation_dimension_basis',
                r'Dimension of the basis\s*\.+\s*(\d+)',
                dtype=int,
            ),
            Quantity(
                'nb_internal_alpha_mol_orbitals',
                r'Number of internal alpha\-MOs\s*\.+\s*(\d+)',
                dtype=int,
            ),
            Quantity(
                'nb_internal_beta_mol_orbitals',
                r'Number of internal beta\-MOs\s*\.+\s*(\d+)',
                dtype=int,
            ),
            Quantity('pair_cutoff', rf'Pair cutoff\s*\.+\s*({re_float})', dtype=float),
            Quantity(
                'atomic_orbital_integral_source',
                r'AO\-integral source\s*\.+\s*(.+)',
                flatten=False,
                convert=False,
            ),
            Quantity(
                'integral_package_used',
                r'Integral package used\s*\.+\s*(.+)',
                flatten=False,
                convert=False,
            ),
            Quantity(
                'nb_alpha_pairs_included',
                r'Number of Alpha\-MO pairs included\s*\.+\s*(\d+)',
                dtype=int,
            ),
            Quantity(
                'nb_beta_pairs_included',
                r'Number of Beta\-MO pairs included\s*\.+\s*(\d+)',
                dtype=int,
            ),
            Quantity(
                'mp2_energy_spin_aa',
                rf'EMP2\(aa\)=\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'mp2_energy_spin_bb',
                rf'EMP2\(bb\)=\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'mp2_energy_spin_ab',
                rf'EMP2\(ab\)=\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'mp2_initial_guess',
                rf'E\(0\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'mp2_energy',
                rf'E\(MP2\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'mp2_total_energy',
                rf'Initial E\(tot\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'T_and_T_energy',
                rf'<T\|T>\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'total_nb_pairs_included',
                r'Number of pairs included\s*\.+\s*(\d+)',
                dtype=int,
            ),
            Quantity(
                'iteration_energy',
                r'(Iter\s*E\(tot\)[\s\S]+?)\-{3}',
                str_operation=str_to_iteration_energy,
                convert=False,
            ),
            Quantity(
                'ccsd_correlation_energy',
                rf'E\(CORR\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'ccsd_total_energy',
                rf'E\(TOT\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'single_norm_half_ss',
                rf'Singles Norm <S\|S>\*\*1/2\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                't1_diagnostic',
                rf'T1 diagnostic\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'ccsdt_total_triples_correction',
                rf'Triples Correction \(T\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'ccsdt_aaa_triples_contribution',
                rf'alpha\-alpha\-alpha\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'ccsdt_aab_triples_contribution',
                rf'alpha\-alpha\-beta\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            # typo in metainfo?
            Quantity(
                'ccsdt_aba_triples_contribution',
                rf'alpha\-beta\-beta\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'ccsdt_bbb_triples_contribution',
                rf'beta\-beta\-beta\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'ccsdt_final_corr_energy',
                rf'Final correlation energy\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'ccsd_final_energy',
                rf'E\(CCSD\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'energy_total',
                rf'E\(CCSD\(T\)\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
        ]

        mp2_quantities = [
            Quantity(
                'mp2_basis_dimension',
                r'Dimension of the basis\s*\.+\s*(\d+)',
                dtype=int,
            ),
            Quantity(
                'scaling_mp2_energy',
                rf'Overall scaling of the MP2 energy\s*\.+\s*({re_float})',
                dtype=float,
            ),
            Quantity(
                'mp2_aux_basis_dimension',
                r'Dimension of the aux\-basis\s*\.+\s*(\d+)',
                dtype=int,
            ),
            Quantity(
                'energy_method_current',
                rf'RI\-MP2 CORRELATION ENERGY:\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            Quantity(
                'energy_total',
                rf'MP2 TOTAL ENERGY:\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
        ]

        localization_quantities = [
            Quantity('type',
                r'Localization creterion\s*\.+\s*(\S+)',
                convert=False,
            ),
            Quantity('n_max_iterations',
                rf'Max. number of iterations\s*\.+\s*({re_float})',
                dtype=float,
            ),
            Quantity(
                'energy_change_tolerance',
                rf'Convergence tolerance\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
        ]


        calculation_quantities = [
            Quantity(
                'cartesian_coordinates',
                #rf'CARTESIAN COORDINATES \(ANGSTROEM\)\s*\-+\s*([\s\S]+?)(?=\-+\s*{re_n}CARTESIAN COORDINATES \(A\.U\.\))',
                rf'CARTESIAN COORDINATES \(ANGSTROEM\)\s*\-+\s*([\s\S]+?){re_n}{re_n}',
                #str_operation=str_to_cartesian_coordinates,
                repeats=False,
            ),
            Quantity(
                'basis_set',
                r'\n *BASIS SET INFORMATION\s*\-+([\s\S]+?)\-{10}',
                sub_parser=TextParser(quantities=basis_set_quantities),
            ),
            Quantity(
                'auxiliary_basis_set',
                r'\n *AUXILIARY BASIS SET INFORMATION\s*\-+([\s\S]+?)\-{10}',
                sub_parser=TextParser(quantities=basis_set_quantities),
            ),
            Quantity(
                'basis_set_statistics',
                r'BASIS SET STATISTICS AND STARTUP INFO([\s\S]+?)\-{10}',
                sub_parser=TextParser(quantities=basis_set_statistics_quantities),
            ),
            Quantity(
                'self_consistent',
                r'((?:ORCA SCF|DFT GRID GENERATION)\s*\-+[\s\S]+?(?:\-{70}|\Z))',
                sub_parser=TextParser(quantities=self_consistent_quantities),
            ),
            Quantity(
                'tddft',
                r'ORCA TD\-DFT(?:/TDA)* CALCULATION\s*\-+\s*([\s\S]+?E\(tot\).*)',
                sub_parser=TextParser(quantities=tddft_quantities),
            ),
            Quantity(
                'mp2',
                r'ORCA MP2 CALCULATION([\s\S]+?MP2 TOTAL ENERGY:.+)',
                sub_parser=TextParser(quantities=mp2_quantities),
            ),
            Quantity(
                'ci',
                r'ORCA\-MATRIX DRIVEN CI([\s\S]+?E\(CCSD\(T\)\).*)',
                sub_parser=TextParser(quantities=ci_quantities),
            ),
            Quantity(
                'loc',
                r'\n *ORCA ORBITAL LOCALIZATION\s*\-+([\s\S]+?)\-{10}',
                sub_parser=TextParser(quantities=localization_quantities),
            ),
            # FIX HERE
            Quantity(
                'cc',
                r'ORCA\-MATRIX DRIVEN CI([\s\S]+?E\(CCSD\(T\)\).*)',
                # optional regex: ECCSDT part is not a must to match
                #r'ORCA\-MATRIX DRIVEN CI([\s\S]+?)(E\(CCSD\(T\)\).*)?',
                sub_parser=TextParser(quantities=coupled_cluster_quantities),
            )
        ]

        geometry_optimization_quantities = [
            Quantity(
                '%s_tol' % key.lower().replace(' ', '_').replace('.', ''),
                rf'%s\s*(\w+)\s*\.+\s*({re_float})' % key,
                dtype=float,
            )
            for key in [
                'Energy Change',
                'Max. Gradient',
                'RMS Gradient',
                'Max. Displacement',
                'RMS Displacement',
            ]
        ]

        geometry_optimization_quantities += [
            Quantity('update_method', r'Update method\s*(\w+)\s*\.+\s*(.+)'),
            Quantity('coords_choice', r'Choice of coordinates\s*(\w+)\s*\.+\s*(.+)'),
            Quantity('initial_hessian', r'Initial Hessian\s*(\w+)\s*\.+\s*(.+)'),
        ]

        geometry_optimization_quantities += [
            Quantity(
                'cycle',
                r'OPTIMIZATION CYCLE\s*\d+\s*\*\s*\*+([\s\S]+?)(?:\*\s*GEOMETRY|OPTIMIZATION RUN DONE|\Z)',
                repeats=True,
                sub_parser=TextParser(quantities=calculation_quantities),
            ),
            Quantity(
                'final_energy_evaluation',
                r'FINAL ENERGY EVALUATION AT THE STATIONARY POINT([\s\S]+?FINAL SINGLE POINT ENERGY.*)',
                sub_parser=TextParser(quantities=calculation_quantities),
            ),
        ]


        self._quantities = [
            Quantity(
                'program_version',
                #r'Program Version\s*([\w_.].*)',
                r'Program Version\s*([\d\.]+)',
                convert=False,
                flatten=False,
            ),
            Quantity(
                'program_svn', r'\(SVN:\s*\$([^$]+)\$\)\s', convert=False, flatten=False
            ),
            Quantity(
                'program_compilation_date',
                r'\(\$Date\:\s*(\w.+?)\s*\$\)',
                convert=False,
                flatten=False,
            ),
            Quantity(
                'input_file',
                r'INPUT FILE\s*\=+([\s\S]+?)END OF INPUT',
                #r'INPUT FILE\s*\=+\s*([\s\S]*?)(?:(?:\|\s*\d+\s*>)[\s\S]*)*END OF INPUT',
                #sub_parser=TextParser(
                #    quantities=[
                #        Quantity('xc_functional', r'\d+>\s*!\s*(\S+)'),
                #        Quantity('tier', r'(\w+SCF)'),
                #    ]
                #),
            ),
            Quantity(
                'basis_set_name',
                #r'----- Orbital basis set information -----([\s\S]+?)\={10}',
                r'----- Orbital basis set information -----([\s\S]+?)\={10}\s*',
                sub_parser=TextParser(quantities=basis_set_naming_quantities),
            ),
            Quantity(
                'single_point',
                r'\* Single Point Calculation \*\s*\*+([\s\S]+?(?:FINAL SINGLE POINT ENERGY.*|\Z))',
                sub_parser=TextParser(quantities=calculation_quantities),
            ),
            Quantity(
                'geometry_optimization',
                r'\* Geometry Optimization Run \*\s*\*+([\s\S]+?(?:OPTIMIZATION RUN DONE|\Z))',
                sub_parser=TextParser(quantities=geometry_optimization_quantities),
            ),
        ]

def str_to_cartesian_coordinates(val_in):
    if isinstance(val_in, list):
        symbols = []
        coordinates = []
        for i in range(0, len(val_in), 4):
            symbols.append(val_in[i])
            coordinates.append(val_in[i+1:i+4])
        coordinates = np.array(coordinates, dtype=float)
        return symbols, coordinates
    else:
        raise ValueError("Expected a list input for cartesian coordinates.")


class ORCAParser(MatchingParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_parser = OutParser()

    def parse_coordinates(self, out_parser, logger):
        cartesian_coordinates = out_parser.get('single_point', {}).get('cartesian_coordinates', [])
        if isinstance(cartesian_coordinates, list):
            symbols, coordinates = str_to_cartesian_coordinates(cartesian_coordinates)
            if len(symbols) == len(coordinates):
                model_system = ModelSystem()
                atomic_cell = AtomicCell()
                atomic_cell.n_atoms = len(symbols)  # Set the number of atoms

                for symbol, coord in zip(symbols, coordinates):
                    try:
                        atom_state = AtomsState(chemical_symbol=symbol)
                        atomic_cell.atoms_state.append(atom_state)
                    except Exception as e:
                        logger.warning(f'Error creating AtomsState: {e}')
                atomic_cell.positions = coordinates
                model_system.cell.append(atomic_cell)
                return model_system
            else:
                logger.error('Mismatch between number of symbols and coordinates.')
        else:
            logger.warning("No atoms information found or incorrect format.")
        return None


    def parse_basis_set(self, out_parser, model_system, logger):
        # Extract basis set information
        main_basis_set = out_parser.get('basis_set_name', {}).get('main_basis_set')
        auxc_basis_set = out_parser.get('basis_set_name', {}).get('auxc_basis_set')
        auxj_basis_set = out_parser.get('basis_set_name', {}).get('auxj_basis_set')
        auxjk_basis_set = out_parser.get('basis_set_name', {}).get('auxjk_basis_set')

        # Check if main basis set exists and append to the numerical settings?
        # Im sure its not the right place.
        if main_basis_set:
            atoms_ref = model_system.cell[0].atoms_state
            bs_settings = AtomCenteredBasisSet(
                main_basis_set=main_basis_set, 
                aux_c_basis_set=auxc_basis_set,
                aux_j_basis_set=auxj_basis_set,
                aux_jk_basis_set=auxjk_basis_set,
                atoms_ref = atoms_ref
            )
            return bs_settings
        else:
            logger.warning("No basis set information found.")
        return None

    def parse_scf(self, out_parser, logger):
        # Extract SCF convergence information
        scf_convergence = out_parser.get('single_point', {}).get('self_consistent', {}).get('scf_settings', {})

        if scf_convergence:
            scf = SelfConsistency(
                n_max_iterations=scf_convergence.get('n_max_iterations'),
                threshold_change=scf_convergence.get('energy_change_tolerance')
            )
            return scf
        else:
            logger.warning("No SCF convergence information found.")
        return None


    def parse_coupled_cluster(self, out_parser, logger):
        cc_data = out_parser.get('single_point', {}).get('cc', {})

        if cc_data:
            model_method = CoupledCluster(
                type= cc_data.get('coupled_cluster_type'),
                reference_determinant=cc_data.get('cc_reference_wavefunction'))
            #model_method.type = cc_data.get('coupled_cluster_type')
            #model_method.reference_determinant = cc_data.get('cc_reference_wavefunction')

            # Perturbative triples
            perturbative_triple_status = cc_data.get('perturbative_triple_excitations_on_off')
            if perturbative_triple_status == 'ON':
                model_method.perturbative_correction = '(T)'

            # Explicit correlation status
            explicit_correlation_status = cc_data.get('f12_correction_on_off')
            if explicit_correlation_status == 'ON':
                model_method.explicit_correlation = 'F12'

            local_approximation = cc_data.get('kc_formation')
            if local_approximation:
                model_method.local_approximation = local_approximation

            output = CCOutputs(
                largest_t2_amplitude=cc_data.get('largest_t2_amplitudes'),
                t1_norm=cc_data.get('t1_diagnostic'),
                reference_energy=cc_data.get('reference_energy'),
                corr_energy_strong=cc_data.get('corr_energy_strong'),
                corr_energy_weak=cc_data.get('corr_energy_weak')
            )
            return model_method, output
        else:
            return None, None


    def parse_localization(self, out_parser, logger):
        loc_data = out_parser.get('single_point', {}).get('loc', {})
        if loc_data:
            localization = Localization(
                type=loc_data.get('type'),
                n_max_iterations=loc_data.get('n_max_iterations'),
                threshold_change=loc_data.get('energy_change_tolerance')
            )
            return localization
    
    def parse(self, mainfile, archive: 'EntryArchive', logger: 'BoundLogger', child_archives=None) -> None:
        self.out_parser.mainfile = mainfile
        self.out_parser.logger = logger

        # Perform parsing
        self.out_parser.parse()
        simulation = Simulation()
        simulation.program = Program(name='ORCA', version=self.out_parser.get('program_version'))
        archive.data = simulation

        model_system = self.parse_coordinates(self.out_parser, logger)
        if model_system:
            simulation.model_system.append(model_system)

        # Initialize model_method 
        model_method = ModelMethod()

        # Parse basis set 
        basis_set = self.parse_basis_set(self.out_parser, model_system, logger)
        #print(basis_set, basis_set.atoms_ref, basis_set.atoms_ref[0])
        if basis_set:
            model_method.numerical_settings.append(basis_set)

        # Parse SCF 
        scf = self.parse_scf(self.out_parser, logger)
        if scf:
            model_method.numerical_settings.append(scf)

        # Parse Integration Grid 


        # Parse coupled cluster data and append if exists
        cc_method, output = self.parse_coupled_cluster(self.out_parser, logger)
        if cc_method:
            simulation.model_method.append(cc_method)
        if output:
            simulation.outputs.append(output)

        # Parse Integration Grid 


        simulation.model_method.append(model_method)



        


