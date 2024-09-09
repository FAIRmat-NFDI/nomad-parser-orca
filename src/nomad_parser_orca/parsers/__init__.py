from nomad.config.models.plugins import ParserEntryPoint

class ORCAEntryPoint(ParserEntryPoint):

    def load(self):
        from nomad_parser_orca.parsers.parser import ORCAParser

        return ORCAParser(**self.dict())


parser_entry_point = ORCAEntryPoint(
    name='ORCAParser',
    description='Parser for coupled cluster output written in a log text file.',
    mainfile_name_re='.*\.log.*',
    mainfile_contents_re=r'EBB2675 Version [\d\.]*',
)