class ORCAParser(MatchingParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_parser = OutParser()

    def parse_atomic_structure(self, out_parser, logger):
        atoms_information = out_parser.get('atoms_information', [])
        if isinstance(atoms_information, list):
            symbols, coordinates = str_to_cartesian_coordinates(atoms_information)
            if len(symbols) == len(coordinates):
                model_system = ModelSystem()
                atomic_cell = AtomicCell()
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

    def parse_coupled_cluster(self, out_parser, logger):
        cc_type = out_parser.get('coupled_cluster_type')
        if cc_type:
            model_method = CoupledCluster(
                type=cc_type,
                reference_determinant=out_parser.get('cc_reference_wavefunction')
            )
            #numerical_settings = PNOSettings(t_close_pair=out_parser.get('tCutPairs'))
            output = CCOutputs(
                largest_t2_amplitude=out_parser.get('largest_t2_amplitudes'),
                t1_norm=out_parser.get('t1_diagnostic'),
                reference_energy=out_parser.get('reference_energy'),
                corr_energy_strong=out_parser.get('corr_energy_strong'),
                corr_energy_weak=out_parser.get('corr_energy_weak')
            )
            return model_method, output
        logger.warning('No coupled cluster data found.')
        return None, None

    def parse(self, mainfile, archive: 'EntryArchive', logger: 'BoundLogger', child_archives=None):
        self.out_parser.mainfile = mainfile
        self.out_parser.logger = logger

        # Perform parsing
        self.out_parser.parse()
        simulation = Simulation()
        simulation.program = Program(name='EBB2675', version=self.out_parser.get('program_version'))
        archive.data = simulation

        # Parse coordinates
        model_system = self.parse_atomic_structure(self.out_parser, logger)
        if model_system:
            simulation.model_system.append(model_system)
    
        # Parse coupled-cluster data
        cc_type = self.out_parser.get('coupled_cluster_type')
        if cc_type:
            model_method = CoupledCluster(
                type=cc_type,
                reference_determinant=self.out_parser.get('cc_reference_wavefunction')
            )
        
            output = CCOutputs(
                largest_t2_amplitude=self.out_parser.get('largest_t2_amplitudes'),
                t1_norm=self.out_parser.get('t1_diagnostic'),
                reference_energy=self.out_parser.get('reference_energy'),
                corr_energy_strong=self.out_parser.get('corr_energy_strong'),
                corr_energy_weak=self.out_parser.get('corr_energy_weak')
            )

            # Add model method and outputs to simulation
            simulation.model_method.append(model_method)
            simulation.outputs.append(output)
            
        else:
            logger.warning('No coupled cluster data found.')


        # Parse SCF settings
        #scf_settings = self.out_parser.get('self_consistent_quantities', {}).get('scf_settings', {})

        #scf = SelfConsistency(n_max_iterations =  scf_settings['max_n_iterations'],
        #                      threshold_change =  scf_settings['energy_change_tolerance'])
        
        #model_method.numerical_settings.append(scf)

        # Parse orbital localization settings

