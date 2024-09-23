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

from nomad.datamodel.datamodel import EntryArchive
from nomad.metainfo import Package
from .xml_parser import MolproXMLOutParser

m_package = Package()


class MolproParser:
    def __init__(self):
        self.parser = MolproXMLOutParser()

    def parse(self, filepath: str, archive: EntryArchive, logger) -> EntryArchive:
        """
        Parse the given file and build up the archive from pre-defined sections.
        The actual parsing is delegated to the suited file-parser.

        Args:
            filepath (str): The path to the file to be parsed.
            archive (EntryArchive): The archive to store the parsed data.
            logger: The logger object for logging messages.

        Returns:
            EntryArchive: The updated archive with the parsed data.
        """
        return self.parser.parse(filepath, archive, logger)


m_package.__init_metainfo__()
