from nomad.config.models.plugins import ParserEntryPoint


class ORCAEntryPoint(ParserEntryPoint):
    def load(self):
        from nomad.parsing.parser import MatchingParserInterface

        return MatchingParserInterface(
            parser_class_name='nomad_parser_orca.parsers.parser.ORCAParser',
            **self.dict(),
        )


parser_entry_point = ORCAEntryPoint(
    name='ORCAParser',
    description='Parser for coupled cluster output written in an out text file.',
    mainfile_name_re=r'.*\.out.*',
    mainfile_contents_re=r'\s+\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\**\s*\s+\*\s+O\s+R\s+C\s+A\s+\*\s*\s+\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\**\s*\s*',
)
