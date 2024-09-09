import logging

from nomad.datamodel import EntryArchive
from nomad_parser_orca.parsers.parser import ORCAParser


def test_parse_file():
    parser = ORCAParser()
    archive = EntryArchive()
    parser.parse('tests/data/example.out', archive, logging.getLogger())

    assert archive.workflow2.name == 'test'
