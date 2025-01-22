#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD.
# See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from nomad.datamodel.metainfo.annotations import Mapper
from nomad.metainfo import SchemaPackage
from nomad.parsing.file_parser.mapping_parser import MAPPING_ANNOTATION_KEY
from nomad_simulations.schema_packages import (
    atoms_state,
    general,
    model_method,
    model_system,
    numerical_settings,
    outputs,
    properties,
    variables,
)

m_package = SchemaPackage()

# simulation
general.Simulation.m_def.m_annotations[MAPPING_ANNOTATION_KEY] = dict(
    info=Mapper(mapper='@'),
)
## program
general.Simulation.program.m_annotations[MAPPING_ANNOTATION_KEY] = dict(
    info=Mapper(mapper='.@')
)
### program quantities
general.Program.version.m_annotations[MAPPING_ANNOTATION_KEY] = dict(
    info=Mapper(mapper='.program_version')
)

general.Simulation.model_system.m_annotations[MAPPING_ANNOTATION_KEY] = dict(
    info=Mapper(mapper='@')
)

# AtomicCell annotations
model_system.AtomicCell.m_def.m_annotations[MAPPING_ANNOTATION_KEY] = dict(
    info=Mapper(mapper=('get_atoms', ['.@']))
)

# Map `positions` and `atoms_state` quantities to specific keys in the `get_atoms` output
model_system.AtomicCell.positions.m_annotations[MAPPING_ANNOTATION_KEY] = dict(
    info=Mapper(mapper='.positions')
)
model_system.AtomicCell.atoms_state.m_annotations[MAPPING_ANNOTATION_KEY] = dict(
    info=Mapper(mapper='.atoms')
)

# AtomsState quantities
atoms_state.AtomsState.chemical_symbol.m_annotations[MAPPING_ANNOTATION_KEY] = dict(
    info=Mapper(mapper='.symbol')
)


m_package.__init_metainfo__()