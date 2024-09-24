from nomad.config.models.plugins import ParserEntryPoint

class ORCAEntryPoint(ParserEntryPoint):

    def load(self):
        from nomad_parser_orca.parsers.parser import ORCAParser

        return ORCAParser(**self.dict())


parser_entry_point = ORCAEntryPoint(
    name='ORCAParser',
    description='Parser for coupled cluster output written in an out text file.',
    mainfile_name_re='.*\.out.*',
    #mainfile_name_re='.*\.out$',
    #mainfile_contents_re=r'Program Version [\d\.]*',
    #mainfile_contents_re = r'\s*Program Version [\d\.]+ -\s+([A-Z]+)\s+-'
    #mainfile_contents_re = r'Program Version\s+([\d\.]+)'
    mainfile_contents_re = r'Program Version\s*([\w_.].*)',
    #mainfile_contents_re=r'EBB2675 Version [\d\.]*',
)