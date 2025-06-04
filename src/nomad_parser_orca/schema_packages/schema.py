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
)

m_package = SchemaPackage()


# Simulation
general.Simulation.m_def.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='@'))
)

# Program
general.Simulation.program.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.@'))
)

# Program quantities
general.Program.version.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper='.program_version'))
)

general.Simulation.model_system.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(dict(info=Mapper(mapper=('get_atoms', ['.@']))))


model_system.ModelSystem.positions.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(dict(info=Mapper(mapper='.positions')))

# 3) Also map the returned 'atoms' list → ModelSystem.particle_states (AtomsState sub‐section)
model_system.ModelSystem.particle_states.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(dict(info=Mapper(mapper='.particle_states')))

# AtomsState quantities
atoms_state.AtomsState.chemical_symbol.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(dict(info=Mapper(mapper='.chemical_symbol')))

# # ModelMethod annotations
# model_method.ModelMethod.m_def.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
#     dict(info=Mapper(mapper='@'))
# )

# # DFT annotations
model_method.DFT.m_def.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
    dict(info=Mapper(mapper=('get_dft_data', ['.@'])))
)

model_method.DFT.contributions.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(dict(info=Mapper(mapper='.@')))

model_method.DFT.jacobs_ladder.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(dict(info=Mapper(mapper='.jacobs_ladder')))

model_method.DFT.exact_exchange_mixing_factor.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(dict(info=Mapper(mapper='.exact_exchange_mixing_factor')))

model_method.DFT.xc_functionals.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(
    dict(
        info=Mapper(
            mapper='.xc_functionals', sub_section=model_method.XCFunctional.m_def
        )
    )
)

model_method.XCFunctional.libxc_name.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(dict(info=Mapper(mapper='.libxc_name')))

model_method.XCFunctional.name.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(dict(info=Mapper(mapper='.name')))

model_method.XCFunctional.weight.m_annotations.setdefault(
    MAPPING_ANNOTATION_KEY, {}
).update(dict(info=Mapper(mapper='.weight')))


# numerical_settings.SelfConsistency.m_def.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
#     dict(info=Mapper(mapper=('get_numerical_settings', ['.@'])))
# )


# numerical_settings.SelfConsistency.n_max_iterations.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {}).update(
#     dict(info=Mapper(mapper='.@'))
# )


m_package.__init_metainfo__()
