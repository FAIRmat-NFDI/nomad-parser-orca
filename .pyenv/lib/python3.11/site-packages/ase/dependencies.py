import importlib
from typing import List, Tuple
from ase.utils import search_current_git_hash


def format_dependency(modname: str) -> Tuple[str, str]:
    """Return (name, path) for given module."""
    try:
        module = importlib.import_module(modname)
    except ImportError:
        return modname, 'not installed'

    version = getattr(module, '__version__', '?')
    name = f'{modname}-{version}'
    if modname == 'ase':
        githash = search_current_git_hash(module)
        if githash:
            name += '-{:.10}'.format(githash)

    # (only packages have __path__, but we are importing packages.)
    return name, str(module.__path__[0])  # type: ignore


def all_dependencies() -> List[Tuple[str, str]]:
    names = ['ase', 'numpy', 'scipy', 'matplotlib', 'spglib',
             'ase_ext', 'flask', 'psycopg2', 'pyamg']
    return [format_dependency(name) for name in names]
