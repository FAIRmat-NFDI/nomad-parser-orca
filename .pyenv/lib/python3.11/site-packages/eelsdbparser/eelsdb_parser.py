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

import os.path
import os
import json
from datetime import datetime
import logging
import glob
import re

from nomad.datamodel import EntryArchive
from nomad.metainfo.metainfo import SubSection
from nomad.parsing.parser import MatchingParser
from nomad.units import ureg
from nomad.metainfo import Section, Package

from nomad.datamodel.metainfo.measurements import (
    Measurement, Sample, Instrument, EELSMeasurement, Spectrum)
from nomad.datamodel import Author
from nomad.datamodel.results import EELSInstrument


default_logger = logging.getLogger(__name__)

m_package = Package(name='eels')


# TODO replace with ParentSection once this is implement in the metainfo
class MyMeasurement(Measurement):
    m_def = Section(extends_base_section=True)
    eels = SubSection(section_def=EELSMeasurement)


class MyInstrument(Instrument):
    m_def = Section(extends_base_section=True)
    eels = SubSection(section_def=EELSInstrument)


m_package.__init_metainfo__()


class EELSDBParser(MatchingParser):
    def __init__(self):
        super().__init__(
            name='parsers/eels', code_name='eels', code_homepage='https://eelsdb.eu/',
            domain='ems',
            mainfile_mime_re=r'application/json',
            mainfile_contents_re=(r'https://eelsdb.eu/spectra')
        )

        self.logger = None

    def parse_msa_file(self, msa_path) -> Spectrum:
        ''' Parses the given msa file and returns a spectrum with data from that file. '''

        metadata_re = re.compile(r'^#\s*([A-Z0-9]+)\s*:(.*)\s*$')
        number_re = r'-?\d+(\.\d+)?(e[\+\-]?\d+)?'
        data_re = re.compile(f'({number_re}),\\s*({number_re})')

        metadata = {}
        energies = []
        counts = []
        with open(msa_path, 'rt') as f:
            for line in f.readlines():
                match = re.match(metadata_re, line)
                if match:
                    metadata[match.group(1)] = match.group(2)
                    continue
                match = re.match(data_re, line)
                if match:
                    energies.append(float(match.group(1)))
                    counts.append(float(match.group(4)))
                    continue

                self.logger.warning('Unexpected line format in .msa file')

        x_units = metadata.get('XUNITS')
        if not x_units or 'undefined' in x_units.lower():
            x_units = 'eV'
            self.logger.warning('Unknown energy units')

        if int(metadata.get('NPOINTS', 0)) != len(energies):
            self.logger.warning('Npoints metadata does not match value count')

        spectrum = Spectrum()
        spectrum.energy = energies * ureg(x_units)
        spectrum.count = counts
        return spectrum

    def parse(self, mainfile_path, archive: EntryArchive, logger=None):
        self.logger = logger
        if not self.logger:
            self.logger = default_logger

        with open(mainfile_path, 'rt') as f:
            raw_metadata = json.load(f)

        measurement = archive.m_create(Measurement)
        measurement.eels = EELSMeasurement()

        # Data
        msa_path = next(iter(
            glob.glob(os.path.join(os.path.dirname(mainfile_path), '*.msa'))), None)
        if msa_path:
            measurement.eels.spectrum = self.parse_msa_file(msa_path)
        else:
            logger.warning('No *.msa file found')

        # Sample
        sample = Sample()
        measurement.sample.append(sample)
        sample.chemical_formula = raw_metadata['formula']
        sample.name = raw_metadata['title']
        elements = raw_metadata.get('elements', [])
        if isinstance(elements, str):
            elements = json.loads(elements)
        sample.elements = elements

        # Measurement
        measurement.measurement_id = str(raw_metadata['id'])
        archive.metadata.external_id = str(raw_metadata['id'])
        measurement.method_name = 'electron energy loss spectroscopy'
        measurement.method_abbreviation = 'EELS'
        measurement.eels.publish_time = datetime.strptime(
            raw_metadata.get('published'), '%Y-%m-%d %H:%M:%S')
        edges = raw_metadata.get('edges', [])
        if isinstance(edges, str):
            edges = json.loads(edges)
        measurement.eels.edges = edges
        measurement.description = raw_metadata['description']

        # Instrument
        # TODO: Add units to variables here
        instrument = measurement.m_create(Instrument)
        instrument.eels = EELSInstrument()
        instrument.name = raw_metadata['microscope']
        max_energy_string = raw_metadata.get('max_energy')
        if max_energy_string is not None:
            value, unit = max_energy_string.split()
            instrument.eels.max_energy = float(value) * ureg(unit)
        min_energy_string = raw_metadata.get('min_energy')
        if min_energy_string is not None:
            value, unit = min_energy_string.split()
            instrument.eels.min_energy = float(value) * ureg(unit)
        resolution_string = raw_metadata.get('resolution')
        if resolution_string is not None:
            value, unit = resolution_string.split()
            instrument.eels.resolution = float(value) * ureg(unit)
        instrument.eels.guntype = raw_metadata['guntype']
        if raw_metadata.get('beamenergy') is not None:
            value, unit = raw_metadata.get('beamenergy').split()
            instrument.eels.beam_energy = float(value) * ureg(unit)
        instrument.eels.step_size = raw_metadata['stepSize']
        if raw_metadata.get('acquisition_mode') is not None:
            instrument.eels.acquisition_mode = raw_metadata['acquisition_mode']
        if raw_metadata.get('beamcurrent') is not None:
            instrument.eels.beam_current = raw_metadata['beamcurrent']
        instrument.eels.detector_type = raw_metadata['detector']
        if raw_metadata.get('darkcurrent') is not None:
            instrument.eels.dark_current = raw_metadata.get('darkcurrent') == 'Yes'

        # Origin
        archive.metadata.external_db = 'EELS Data Base'
        archive.metadata.references = [
            raw_metadata[reference]
            for reference in ['permalink', 'preview_url', 'entry_repository_url', 'api_permalink']
            if reference in raw_metadata]

        # # Author
        author = measurement.eels.m_create(Author)
        names = raw_metadata['author']['name'].rsplit(' ', 1)
        if len(names) == 2:
            author.first_name, author.last_name = names
        else:
            author.last_name = names[0]
        archive.metadata.entry_coauthors = [author]
        # author.author_profile_url = raw_metadata['author']['profile_url']
        # author.author_profile_api_url = raw_metadata['author']['profile_api_url']
