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

import requests
import json
import zipfile
import os.path


def download_eels(max: int = 1000):
    ''' Downloads all EELS data. '''
    with zipfile.ZipFile('eels-data.zip', mode='x') as zf:
        try:
            eelsdb = requests.get(f'https://api.eelsdb.eu/spectra?per_page={max}').json()
        except Exception as e:
            raise Exception('Could not download spectras from EELSDB') from e

        for entry in eelsdb:
            print(f'add {entry["permalink"]}')
            path = entry['permalink'].strip('https://').strip('/')
            with zf.open(os.path.join(path, 'metadata.json'), mode='w') as f:
                f.write(json.dumps(entry, indent=2).encode('utf-8'))

            try:
                data = requests.get(entry['download_link']).content
                with zf.open(os.path.join(path, 'data.msa'), mode='w') as f:
                    f.write(data)

            except Exception:
                print(f'Could not download spectra for {entry["permalink"]}')


if __name__ == '__main__':
    download_eels()
