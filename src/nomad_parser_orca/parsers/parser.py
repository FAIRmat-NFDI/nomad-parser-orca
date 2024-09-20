from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

import numpy as np
from nomad.config import config
from nomad.datamodel.metainfo.workflow import Workflow
from nomad.parsing.file_parser import Quantity as ParsedQuantity
from nomad.parsing.file_parser import TextParser
from nomad.parsing.parser import MatchingParser
from nomad.units import ureg
from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.general import Program, Simulation
from nomad_simulations.schema_packages.model_method import ModelMethod
from nomad_simulations.schema_packages.model_system import (AtomicCell,
                                                            ModelSystem)
from nomad_simulations.schema_packages.numerical_settings import SelfConsistency


from nomad_parser_orca.schema_packages.schema_package import CoupledCluster
from nomad_parser_orca.schema_packages.numerical_settings import PNOSettings, LocMet
from nomad_parser_orca.schema_packages.outputs import CCOutputs


configuration = config.get_plugin_entry_point(
    'nomad_parser_orca.parsers:parser_entry_point'
)

re_n = r'\r*\n'


class OutParser(TextParser):

    def init_quantities(self):
        re_float = r'[-+]?\d+\.?\d*(?:[Ee][-+]\d+)?'
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
            ParsedQuantity(
                'coupled_cluster_type',
                r'Correlation treatment\s+\.\.\.\s+([A-Z]+)',
                repeats=False
            ),
            ParsedQuantity(
                'cc_reference_wavefunction',
                r'Reference Wavefunction\s+\.\.\.\s+([A-Z]+)',
                repeats=False
            ),
            ParsedQuantity(
                't1_diagnostic',
                r'T1 diagnostic\s+\.\.\.\s+([\d\.]+)',
                repeats=False
            ),
            ParsedQuantity(
                'largest_t2_amplitudes',
                r'\b\d+[ab]->\d+[ab]\b\s+\b\d+[ab]->\d+[ab]\b\s+([-+]?\d*\.\d+)\b',
                repeats=True
            ),
            ParsedQuantity(
                'reference_energy',
                r'E\(0\)\s*\.\.\.\s*(-?[\d\.]+)',
                repeats=False
            ),
            ParsedQuantity(
                'corr_energy_strong',
                r'E\(CORR\)\(strong-pairs\)\s*\.\.\.\s*(-?[\d\.]+)',
                repeats=False
            ),
            ParsedQuantity(
                'corr_energy_weak',
                r'E\(CORR\)\(weak-pairs\)\s*\.\.\.\s*(-?[\d\.]+)',
                repeats=False
            ),
            ParsedQuantity(
                'tCutPairs',
                r'TCutPairs\s*=\s*([-+]?\d*\.\d+([eE][-+]?\d+)?)',
                repeats=False
            ),
        ]

        # Basis set related quantities
        basis_set_quantities = [
            ParsedQuantity('basis_set_atom_labels', r'Type\s*(\w+)', repeats=True),
            ParsedQuantity('basis_set', r':\s*(\w+)\s*contracted\s*to', repeats=True),
            ParsedQuantity('basis_set_contracted', r'(\w+)\s*pattern', repeats=True),
        ]

        # Basis set statistics quantities
        basis_set_statistics_quantities = [
            ParsedQuantity(
                'nb_of_primitive_gaussian_shells',
                r'# of primitive gaussian shells\s*\.+\s*(\d+)',
                repeats=True,
                dtype=int,
            ),
            ParsedQuantity(
                'nb_of_primitive_gaussian_functions',
                r'# of primitive gaussian functions\s*\.+\s*(\d+)',
                repeats=True,
                dtype=int,
            ),
            ParsedQuantity(
                'nb_of_contracted_shells',
                r'# of contracted shells\s*\.+\s*(\d+)',
                repeats=True,
                dtype=int,
            ),
            ParsedQuantity(
                'nb_of_contracted_basis_functions',
                r'# of contracted (?:aux-)?basis functions\s*\.+\s*(\d+)',
                repeats=True,
                dtype=int,
            ),
            ParsedQuantity(
                'highest_angular_moment',
                r'Highest angular momentum\s*\.+\s*(\d+)',
                repeats=True,
                dtype=int,
            ),
            ParsedQuantity(
                'maximum_contraction_depth',
                r'Maximum contraction depth\s*\.+\s*(\d+)',
                repeats=True,
                dtype=int,
            ),
        ]

        # Grid related quantities
        grid_quantities = [
            ParsedQuantity(
                'gral_integ_accuracy',
                rf'General Integration Accuracy\s*IntAcc\s*\.+\s*({re_float})',
                dtype=float,
            ),
            ParsedQuantity(
                'radial_grid_type',
                r'Radial Grid Type\s*RadialGrid\s*\.+\s*(\S+)',
                convert=False,
            ),
            ParsedQuantity(
                'angular_grid',
                r'Angular Grid \(max\. acc\.\)\s*AngularGrid\s*\.+\s*(\S+)',
                convert=False,
            ),
            ParsedQuantity(
                'grid_pruning_method',
                r'Angular grid pruning method\s*GridPruning\s*\.+\s*(.+)',
                flatten=False,
                convert=False,
            ),
            ParsedQuantity(
                'weight_gener_scheme',
                r'Weight generation scheme\s*WeightScheme\s*\.+\s*(\w+)',
                convert=False,
            ),
            ParsedQuantity(
                'basis_fn_cutoff',
                rf'Basis function cutoff\s*BFCut\s*\.+\s*({re_float})',
                dtype=float,
            ),
            ParsedQuantity(
                'integr_weight_cutoff',
                rf'Integration weight cutoff\s*WCut\s*\.+\s*({re_float})',
                dtype=float,
            ),
            ParsedQuantity(
                'nb_grid_pts_after_initial_pruning',
                r'# of grid points \(after initial pruning\)\s*\.+\s*(\d+)',
                dtype=int,
            ),
            ParsedQuantity(
                'nb_grid_pts_after_weights_screening',
                r'# of grid points \(after weights\+screening\)\s*\.+\s*(\d+)',
                dtype=int,
            ),
            ParsedQuantity(
                'total_nb_grid_pts',
                r'Total number of grid points\s*\.+\s*(\d+)',
                dtype=int,
            ),
            ParsedQuantity(
                'total_nb_batches', r'Total number of batches\s*\.+\s*(\d+)', dtype=int
            ),
            ParsedQuantity(
                'avg_nb_points_per_batch',
                r'Average number of points per batch\s*\.+\s*(\d+)',
                dtype=int,
            ),
            ParsedQuantity(
                'avg_nb_grid_pts_per_atom',
                r'Average number of grid points per atom\s*\.+\s*(\d+)',
                dtype=int,
            ),
        ]

        # SCF convergence quantities
        scf_convergence_quantities = [
            ParsedQuantity(
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
            ParsedQuantity(
                'atomic_charges',
                r'[A-Z]+ ATOMIC CHARGES.*\n\-+([\s\S]+?)\-{10}',
                sub_parser=TextParser(
                    quantities=[
                        ParsedQuantity('species', r'\n *\d+\s*(\w+)', repeats=True),
                        ParsedQuantity(
                            'charge', rf':\s*({re_float})', repeats=True, dtype=float
                        ),
                        ParsedQuantity(
                            'total_charge',
                            rf'Sum of atomic charges\s*:\s*({re_float})',
                            dtype=float,
                        ),
                    ]
                ),
            ),
            ParsedQuantity(
                'orbital_charges',
                rf'[A-Z]+ REDUCED ORBITAL CHARGES.*\s*\-+([\s\S]+?{re_n}{re_n})',
                sub_parser=TextParser(
                    quantities=[
                        ParsedQuantity(
                            'atom',
                            r'([A-Z][a-z]?\s*[spdf][\s\S]+?)\n *(?:\d|\Z)',
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[
                                    ParsedQuantity(
                                        'species', r'([A-Z][a-z]?)', convert=False
                                    ),
                                    ParsedQuantity(
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
            ParsedQuantity(
                'scf_settings',
                r'SCF SETTINGS\s*\-+([\s\S]+?)\-{12}',
                sub_parser=TextParser(
                    quantities=[
                        ParsedQuantity(
                            'XC_functional_type',
                            r'Ab initio Hamiltonian\s*Method\s*\.+\s*(\S+)',
                            convert=False,
                        ),
                        ParsedQuantity(
                            'XC_functional_type',
                            r'Density Functional\s*Method\s*\.+\s*(\S+)',
                            convert=False,
                        ),
                        ParsedQuantity(
                            'exchange_functional',
                            r'Exchange Functional\s*Exchange\s*\.+\s*(\S+)',
                            convert=False,
                        ),
                        ParsedQuantity(
                            'xalpha_param',
                            rf'X-Alpha parameter\s*XAlpha\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                        ParsedQuantity(
                            'beckes_beta_param',
                            rf'Becke\'s b parameter\s*XBeta\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                        ParsedQuantity(
                            'correl_functional',
                            r'Correlation Functional Correlation\s*\.+\s*(\S+)',
                            convert=False,
                        ),
                        ParsedQuantity(
                            'lda_part_of_gga_corr',
                            r'LDA part of GGA corr\.\s*LDAOpt\s*\.+\s*(\S+)',
                            convert=False,
                        ),
                        ParsedQuantity(
                            'scalar_relativistic_method',
                            r'Scalar relativistic method\s*\.+\s*(\w+)',
                            convert=False,
                        ),
                        ParsedQuantity(
                            'speed_of_light_used',
                            rf'Speed of light used\s*Velit\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                        ParsedQuantity(
                            'hf_type',
                            r'Hartree-Fock type\s*HFTyp\s*\.+\s*(\w+)',
                            convert=False,
                        ),
                        ParsedQuantity(
                            'total_charge',
                            rf'Total Charge\s*Charge\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                        ParsedQuantity(
                            'multiplicity',
                            rf'Multiplicity\s*Mult\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                        ParsedQuantity(
                            'nelectrons',
                            rf'Number of Electrons\s*NEL\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                        ParsedQuantity(
                            'nuclear_repulsion',
                            rf'Nuclear Repulsion\s*ENuc\s*\.+\s*({re_float})',
                            dtype=float,
                            unit=ureg.hartree,
                        ),
                        ParsedQuantity(
                            'n_max_iterations',
                            rf'Maximum # iterations\s*MaxIter\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                        ParsedQuantity(
                            'convergence_check_mode',
                            r'Convergence Check Mode ConvCheckMode\s*\.+\s*(\S+)',
                            convert=False,
                        ),
                        ParsedQuantity(
                            'energy_change_tolerance',
                            rf'Energy Change\s*TolE\s*\.+\s*({re_float})',
                            dtype=float,
                            unit=ureg.hartree,
                        ),
                        ParsedQuantity(
                            '1_elect_energy_change',
                            rf'1\-El\. energy change\s*\.+\s*({re_float})',
                            dtype=float,
                        ),
                    ]
                ),
            ),
            ParsedQuantity(
                'dft_grid_generation',
                r'DFT GRID GENERATION\s*\-+([\s\S]+?\-{10})',
                sub_parser=TextParser(quantities=grid_quantities),
            ),
            ParsedQuantity(
                'scf_iterations',
                r'SCF ITERATIONS\s*\-+([\s\S]+?)\*{10}',
                sub_parser=TextParser(
                    quantities=[
                        ParsedQuantity(
                            'energy',
                            rf'\n *\d+\s*({re_float})\s*{re_float}',
                            repeats=True,
                            dtype=float,
                            unit=ureg.hartree,
                        )
                    ]
                ),
            ),
            ParsedQuantity(
                'final_grid',
                r'Setting up the final grid:([\s\S]+?)\-{10}',
                sub_parser=TextParser(quantities=grid_quantities),
            ),
            ParsedQuantity(
                'total_scf_energy',
                r'TOTAL SCF ENERGY\s*\-+([\s\S]+?)\-{10}',
                sub_parser=TextParser(
                    quantities=[
                        ParsedQuantity(
                            name,
                            rf'%s\s*:\s*({re_float})' % key,
                            dtype=float,
                            unit=ureg.hartree,
                        )
                        for key, name in self._energy_mapping.items()
                    ]
                    + [
                        ParsedQuantity(
                            'virial_ratio',
                            rf'Virial Ratio\s*:\s*({re_float})',
                            dtype=float,
                        ),
                        ParsedQuantity(
                            'nb_elect_alpha_channel',
                            rf'N\(Alpha\)\s*:\s*({re_float})',
                            dtype=float,
                        ),
                        ParsedQuantity(
                            'nb_elect_beta_channel',
                            rf'N\(Beta\)\s*:\s*({re_float})',
                            dtype=float,
                        ),
                        ParsedQuantity(
                            'nb_elect_total',
                            rf'N\(Total\)\s*:\s*({re_float})',
                            dtype=float,
                        ),
                    ]
                ),
            ),
            ParsedQuantity(
                'scf_convergence',
                r'SCF CONVERGENCE\s*\-+([\s\S]+?)\-{10}',
                sub_parser=TextParser(quantities=scf_convergence_quantities),
            ),
            ParsedQuantity(
                'orbital_energies',
                rf'NO\s*OCC\s*E\(Eh\)\s*E\(eV\)\s*([\s\S]+?){re_n}{re_n}',
                str_operation=lambda x: np.array(
                    [v.split()[:4] for v in x.split('\n')], dtype=float
                ),
                repeats=True,
            ),
            ParsedQuantity(
                'mulliken',
                r'MULLIKEN POPULATION ANALYSIS \*\s*\*+([\s\S]+?)\*{10}',
                sub_parser=TextParser(quantities=population_quantities),
            ),
            ParsedQuantity(
                'timings',
                r'\n *TIMINGS\s*\-+\s*([\s\S]+?)\-{10}',
                sub_parser=TextParser(
                    quantities=[
                        ParsedQuantity(
                            name,
                            rf'%s\s*\.+\s*({re_float})' % key,
                            dtype=float,
                            unit=ureg.s,
                        )
                        for key, name in self._timing_mapping.items()
                    ]
                ),
            ),
            ParsedQuantity(
                'time_calculation',
                r'Total SCF time\: (\d+) days (\d+) hours (\d+) min (\d+) sec ',
                dtype=np.dtype(np.float64),
            ),
        ]

        # TODO parse more properties, add to metainfo
        tddft_quantities = [
            ParsedQuantity(
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
            ParsedQuantity(
                'electronic_structure_method',
                r'Correlation treatment\s*\.+\s*(\S+)',
                convert=False,
            ),
            ParsedQuantity(
                'single_excitations_on_off',
                r'Single excitations\s*\.+\s*(\S+)',
                convert=False,
            ),
            ParsedQuantity(
                'orbital_opt_on_off',
                r'Orbital optimization\s*\.+\s*(\S+)',
                convert=False,
            ),
            ParsedQuantity(
                'z_vector_calc_on_off',
                r'Calculation of Z vector\s*\.+\s*(\S+)',
                convert=False,
            ),
            ParsedQuantity(
                'Brueckner_orbitals_calc_on_off',
                r'Calculation of Brueckner orbitals\s*\.+\s*(\S+)',
                convert=False,
            ),
            ParsedQuantity(
                'perturbative_triple_excitations_on_off',
                r'Perturbative triple excitations\s*\.+\s*(\S+)',
                convert=False,
            ),
            ParsedQuantity(
                'f12_correction_on_off',
                r'Calculation of F12 correction\s*\.+\s*(\S+)',
                convert=False,
            ),
            ParsedQuantity(
                'frozen_core_treatment',
                r'Frozen core treatment\s*\.+\s*(.+)',
                flatten=False,
                convert=False,
            ),
            ParsedQuantity(
                'reference_wave_function',
                r'Reference Wavefunction\s*\.+\s*(.+)',
                flatten=False,
                convert=False,
            ),
            ParsedQuantity(
                'nb_of_atomic_orbitals', r'Number of AO\'s\s*\.+\s*(\d+)', dtype=int
            ),
            ParsedQuantity(
                'nb_of_electrons', r'Number of electrons\s*\.+\s*(\d+)', dtype=int
            ),
            ParsedQuantity(
                'nb_of_correlated_electrons',
                r'Number of correlated electrons\s*\.+\s*(\d+)',
                dtype=int,
            ),
            ParsedQuantity(
                'integral_transformation',
                r'Integral transformation\s*\.+\s*(.+)',
                flatten=False,
                convert=False,
            ),
            ParsedQuantity(
                'level_shift_amplitude_update',
                rf'Level shift for amplitude update\s*\.+\s*({re_float})',
                dtype=float,
            ),
            ParsedQuantity(
                'coulomb_transformation_type',
                r'Transformation type\s*\.+\s*(.+)',
                flatten=False,
                convert=False,
            ),
            ParsedQuantity(
                'coulomb_transformation_dimension_basis',
                r'Dimension of the basis\s*\.+\s*(\d+)',
                dtype=int,
            ),
            ParsedQuantity(
                'nb_internal_alpha_mol_orbitals',
                r'Number of internal alpha\-MOs\s*\.+\s*(\d+)',
                dtype=int,
            ),
            ParsedQuantity(
                'nb_internal_beta_mol_orbitals',
                r'Number of internal beta\-MOs\s*\.+\s*(\d+)',
                dtype=int,
            ),
            ParsedQuantity('pair_cutoff', rf'Pair cutoff\s*\.+\s*({re_float})', dtype=float),
            ParsedQuantity(
                'atomic_orbital_integral_source',
                r'AO\-integral source\s*\.+\s*(.+)',
                flatten=False,
                convert=False,
            ),
            ParsedQuantity(
                'integral_package_used',
                r'Integral package used\s*\.+\s*(.+)',
                flatten=False,
                convert=False,
            ),
            ParsedQuantity(
                'nb_alpha_pairs_included',
                r'Number of Alpha\-MO pairs included\s*\.+\s*(\d+)',
                dtype=int,
            ),
            ParsedQuantity(
                'nb_beta_pairs_included',
                r'Number of Beta\-MO pairs included\s*\.+\s*(\d+)',
                dtype=int,
            ),
            ParsedQuantity(
                'mp2_energy_spin_aa',
                rf'EMP2\(aa\)=\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'mp2_energy_spin_bb',
                rf'EMP2\(bb\)=\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'mp2_energy_spin_ab',
                rf'EMP2\(ab\)=\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'mp2_initial_guess',
                rf'E\(0\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'mp2_energy',
                rf'E\(MP2\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'mp2_total_energy',
                rf'Initial E\(tot\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'T_and_T_energy',
                rf'<T\|T>\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'total_nb_pairs_included',
                r'Number of pairs included\s*\.+\s*(\d+)',
                dtype=int,
            ),
            ParsedQuantity(
                'iteration_energy',
                r'(Iter\s*E\(tot\)[\s\S]+?)\-{3}',
                str_operation=str_to_iteration_energy,
                convert=False,
            ),
            ParsedQuantity(
                'ccsd_correlation_energy',
                rf'E\(CORR\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'ccsd_total_energy',
                rf'E\(TOT\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'single_norm_half_ss',
                rf'Singles Norm <S\|S>\*\*1/2\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                't1_diagnostic',
                rf'T1 diagnostic\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'ccsdt_total_triples_correction',
                rf'Triples Correction \(T\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'ccsdt_aaa_triples_contribution',
                rf'alpha\-alpha\-alpha\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'ccsdt_aab_triples_contribution',
                rf'alpha\-alpha\-beta\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            # typo in metainfo?
            ParsedQuantity(
                'ccsdt_aba_triples_contribution',
                rf'alpha\-beta\-beta\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'ccsdt_bbb_triples_contribution',
                rf'beta\-beta\-beta\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'ccsdt_final_corr_energy',
                rf'Final correlation energy\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'ccsd_final_energy',
                rf'E\(CCSD\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'energy_total',
                rf'E\(CCSD\(T\)\)\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
        ]

        mp2_quantities = [
            ParsedQuantity(
                'mp2_basis_dimension',
                r'Dimension of the basis\s*\.+\s*(\d+)',
                dtype=int,
            ),
            ParsedQuantity(
                'scaling_mp2_energy',
                rf'Overall scaling of the MP2 energy\s*\.+\s*({re_float})',
                dtype=float,
            ),
            ParsedQuantity(
                'mp2_aux_basis_dimension',
                r'Dimension of the aux\-basis\s*\.+\s*(\d+)',
                dtype=int,
            ),
            ParsedQuantity(
                'energy_method_current',
                rf'RI\-MP2 CORRELATION ENERGY:\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
            ParsedQuantity(
                'energy_total',
                rf'MP2 TOTAL ENERGY:\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
        ]

        localization_quantities = [
            ParsedQuantity('type',
                r'Localization creterion\s*\.+\s*(\S+)',
                convert=False,
            ),
            ParsedQuantity('n_max_iterations',
                rf'Max. number of iterations\s*\.+\s*({re_float})',
                dtype=float,
            ),
            ParsedQuantity(
                'energy_change_tolerance',
                rf'Convergence tolerance\s*\.+\s*({re_float})',
                dtype=float,
                unit=ureg.hartree,
            ),
        ]

        calculation_quantities = [
            ParsedQuantity(
                'cartesian_coordinates',
                rf'CARTESIAN COORDINATES \(ANGSTROEM\)\s*\-+\s*([\s\S]+?){re_n}{re_n}',
                #str_operation=str_to_cartesian_coordinates,
            ),
            ParsedQuantity(
                'basis_set',
                r'\n *BASIS SET INFORMATION\s*\-+([\s\S]+?)\-{10}',
                sub_parser=TextParser(quantities=basis_set_quantities),
            ),
            ParsedQuantity(
                'auxiliary_basis_set',
                r'\n *AUXILIARY BASIS SET INFORMATION\s*\-+([\s\S]+?)\-{10}',
                sub_parser=TextParser(quantities=basis_set_quantities),
            ),
            ParsedQuantity(
                'basis_set_statistics',
                r'BASIS SET STATISTICS AND STARTUP INFO([\s\S]+?)\-{10}',
                sub_parser=TextParser(quantities=basis_set_statistics_quantities),
            ),
            ParsedQuantity(
                'self_consistent',
                r'((?:ORCA SCF|DFT GRID GENERATION)\s*\-+[\s\S]+?(?:\-{70}|\Z))',
                sub_parser=TextParser(quantities=self_consistent_quantities),
            ),
            ParsedQuantity(
                'tddft',
                r'ORCA TD\-DFT(?:/TDA)* CALCULATION\s*\-+\s*([\s\S]+?E\(tot\).*)',
                sub_parser=TextParser(quantities=tddft_quantities),
            ),
            ParsedQuantity(
                'mp2',
                r'ORCA MP2 CALCULATION([\s\S]+?MP2 TOTAL ENERGY:.+)',
                sub_parser=TextParser(quantities=mp2_quantities),
            ),
            ParsedQuantity(
                'ci',
                r'ORCA\-MATRIX DRIVEN CI([\s\S]+?E\(CCSD\(T\)\).*)',
                sub_parser=TextParser(quantities=ci_quantities),
            ),
            ParsedQuantity(
                'loc',
                r'\n *ORCA ORBITAL LOCALIZATION\s*\-+([\s\S]+?)\-{10}',
                sub_parser=TextParser(quantities=localization_quantities),
            ),
            # FIX HERE
            ParsedQuantity(
                'cc',
                r'ORCA\-MATRIX DRIVEN CI([\s\S]+?E\(CCSD\(T\)\).*)',
                # optional regex: ECCSDT part is not a must to match
                #r'ORCA\-MATRIX DRIVEN CI([\s\S]+?)(E\(CCSD\(T\)\).*)?',
                sub_parser=TextParser(quantities=coupled_cluster_quantities),
            )
        ]

        geometry_optimization_quantities = [
            ParsedQuantity(
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
            ParsedQuantity('update_method', r'Update method\s*(\w+)\s*\.+\s*(.+)'),
            ParsedQuantity('coords_choice', r'Choice of coordinates\s*(\w+)\s*\.+\s*(.+)'),
            ParsedQuantity('initial_hessian', r'Initial Hessian\s*(\w+)\s*\.+\s*(.+)'),
        ]

        geometry_optimization_quantities += [
            ParsedQuantity(
                'cycle',
                r'OPTIMIZATION CYCLE\s*\d+\s*\*\s*\*+([\s\S]+?)(?:\*\s*GEOMETRY|OPTIMIZATION RUN DONE|\Z)',
                repeats=True,
                sub_parser=TextParser(quantities=calculation_quantities),
            ),
            ParsedQuantity(
                'final_energy_evaluation',
                r'FINAL ENERGY EVALUATION AT THE STATIONARY POINT([\s\S]+?FINAL SINGLE POINT ENERGY.*)',
                sub_parser=TextParser(quantities=calculation_quantities),
            ),
        ]


        self._quantities = [
            ParsedQuantity(
                'program_version',
                r'Program Version\s*([\w_.].*)',
                #r'EBB2675 Version *([\d\.]+)',
                #convert=False,
                #flatten=False,
            ),
            ParsedQuantity(
                'program_svn', r'\(SVN:\s*\$([^$]+)\$\)\s', convert=False, flatten=False
            ),
            ParsedQuantity(
                'program_compilation_date',
                r'\(\$Date\:\s*(\w.+?)\s*\$\)',
                convert=False,
                flatten=False,
            ),
            ParsedQuantity(
                'input_file',
                r'INPUT FILE\s*\=+([\s\S]+?)END OF INPUT',
                sub_parser=TextParser(
                    quantities=[
                        ParsedQuantity('xc_functional', r'\d+>\s*!\s*(\S+)'),
                        ParsedQuantity('tier', r'(\w+SCF)'),
                    ]
                ),
            ),
            ParsedQuantity(
                'single_point',
                r'\* Single Point Calculation \*\s*\*+([\s\S]+?(?:FINAL SINGLE POINT ENERGY.*|\Z))',
                sub_parser=TextParser(quantities=calculation_quantities),
            ),
            ParsedQuantity(
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
                print(atomic_cell)
                return model_system
            else:
                logger.error('Mismatch between number of symbols and coordinates.')
        else:
            logger.warning("No atoms information found or incorrect format.")
        return None

    def parse_coupled_cluster(self, out_parser, logger):
        cc_data = out_parser.get('single_point', {}).get('cc', {})
        if cc_data:
            model_method = CoupledCluster(
                type=cc_data.get('coupled_cluster_type'),
                reference_determinant=cc_data.get('cc_reference_wavefunction')
            )
            output = CCOutputs(
                largest_t2_amplitude=cc_data.get('largest_t2_amplitudes'),
                t1_norm=cc_data.get('t1_diagnostic'),
                reference_energy=cc_data.get('reference_energy'),
                corr_energy_strong=cc_data.get('corr_energy_strong'),
                corr_energy_weak=cc_data.get('corr_energy_weak')
            )
            return model_method, output
        logger.warning('No coupled cluster data found.')
        return None, None

    def parse_localization(self, out_parser, logger):
        loc_data = out_parser.get('single_point', {}).get('loc', {})
        if loc_data:
            localization = LocMet(
                type=loc_data.get('type'),
                n_max_iterations=loc_data.get('n_max_iterations'),
                threshold_change=loc_data.get('energy_change_tolerance')
            )
            return localization
        return None

    def parse(self, mainfile, archive: 'EntryArchive', logger: 'BoundLogger',
              child_archives=None) -> None:
        self.out_parser.mainfile = mainfile
        self.out_parser.logger = logger

        # Perform parsing
        self.out_parser.parse()
        simulation = Simulation()
        simulation.program = Program(name='EBB2675', version=self.out_parser.get('program_version'))
        archive.data = simulation

        model_system = self.parse_coordinates(self.out_parser, logger)
        if model_system:
            simulation.model_system.append(model_system)

        model_method, output = self.parse_coupled_cluster(self.out_parser, logger)
        if model_method:
            simulation.model_method.append(model_method)
        if output:
            simulation.outputs.append(output)

        scf_convergence = self.out_parser.get('single_point', {}).get('self_consistent', {}).get('scf_settings', {})
        if scf_convergence:
            scf = SelfConsistency(
                n_max_iterations=scf_convergence.get('n_max_iterations'),
                threshold_change=scf_convergence.get('energy_change_tolerance')
            )
            model_method.numerical_settings.append(scf)


