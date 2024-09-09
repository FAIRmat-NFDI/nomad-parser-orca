from nomad.config.models.plugins import ParserEntryPoint


class PySCFEntryPoint(ParserEntryPoint):
    def load(self):
        from nomad_parser_pyscf.parsers.parser import PySCFParser

        return PySCFParser(**self.dict())


parser_entry_point = PySCFEntryPoint(
    name='PySCFParser',
    description='Parser for PySCF output written in a log text file.',
    mainfile_name_re='.*\.log.*',
    mainfile_contents_re=r'PySCF version [\d\.]*',
)
