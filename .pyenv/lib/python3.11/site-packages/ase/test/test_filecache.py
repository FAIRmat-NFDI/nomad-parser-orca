import pytest
from ase.utils.filecache import MultiFileJSONCache, CombinedJSONCache, Locked


pytestmark = pytest.mark.usefixtures('testdir')


@pytest.fixture
def cache():
    return MultiFileJSONCache('cache')


def sample_dict():
    return {'hello': [1, 2, 3], 'world': 'grumble'}


def test_basic(cache):
    assert len(cache) == 0

    cache['hello'] = 'grumble'
    assert len(cache) == 1
    assert 'hello' in cache

    grumble = cache.pop('hello')
    assert grumble == 'grumble'
    assert 'hello' not in cache
    assert len(cache) == 0


@pytest.mark.parametrize('dct', [
    {},
    sample_dict(),
])
def test_cache(dct, cache):
    cache.update(dct)
    assert dict(cache) == dct


def test_combine(cache):
    dct = sample_dict()
    cache.update(dct)
    combined = cache.combine()
    assert dict(combined) == dct


def test_split():
    dct = sample_dict()
    combined = CombinedJSONCache.dump_cache('cache', dct)
    assert dict(combined) == dct
    cache = combined.split()
    assert dict(cache) == dct
    assert len(combined) == 0


def test_lock(cache):
    with cache.lock('hello'):
        # When element is locked but nothing is written, the
        # cache is defined to "contain" None
        assert 'hello' in cache
        assert cache['hello'] is None

        # Other keys should function as normal:
        cache['xx'] = 1
        assert cache['xx'] == 1


def test_already_locked(cache):
    with cache.lock('hello') as handle:
        assert handle is not None
        with cache.lock('hello') as otherhandle:
            assert otherhandle is None

        with pytest.raises(Locked):
            cache['hello'] = 'world'


def test_no_overwrite_combine(cache):
    cache.combine()
    with pytest.raises(RuntimeError, match='Already exists'):
        cache.combine()
