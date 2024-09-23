from pathlib import Path
import json
from collections.abc import MutableMapping, Mapping
from contextlib import contextmanager
from ase.io.jsonio import read_json, write_json, encode
from ase.utils import opencew


def missing(key):
    raise KeyError(key)


class Locked(Exception):
    pass


class CacheLock:
    def __init__(self, fd, key):
        self.fd = fd
        self.key = key

    def save(self, value):
        json_utf8 = encode(value).encode('utf-8')
        try:
            self.fd.write(json_utf8)
        except Exception as ex:
            raise RuntimeError(f'Failed to save {value} to cache') from ex
        finally:
            self.fd.close()


class MultiFileJSONCache(MutableMapping):
    writable = True

    def __init__(self, directory):
        self.directory = Path(directory)

    def _filename(self, key):
        return self.directory / f'cache.{key}.json'

    def _glob(self):
        return self.directory.glob('cache.*.json')

    def __iter__(self):
        for path in self._glob():
            cache, key = path.stem.split('.', 1)
            if cache != 'cache':
                continue
            yield key

    def __len__(self):
        # Very inefficient this, but not a big usecase.
        return len(list(self._glob()))

    @contextmanager
    def lock(self, key):
        self.directory.mkdir(exist_ok=True, parents=True)
        path = self._filename(key)
        fd = opencew(path)
        try:
            if fd is None:
                yield None
            else:
                yield CacheLock(fd, key)
        finally:
            if fd is not None:
                fd.close()

    def __setitem__(self, key, value):
        with self.lock(key) as handle:
            if handle is None:
                raise Locked(key)
            handle.save(value)

    def __getitem__(self, key):
        path = self._filename(key)
        try:
            return read_json(path, always_array=False)
        except FileNotFoundError:
            missing(key)
        except json.decoder.JSONDecodeError:
            # May be partially written, which typically means empty
            # because the file was locked with exclusive-write-open.
            #
            # Since we decide what keys we have based on which files exist,
            # we are obligated to return a value for this case too.
            # So we return None.
            return None

    def __delitem__(self, key):
        try:
            self._filename(key).unlink()
        except FileNotFoundError:
            missing(key)

    def combine(self):
        cache = CombinedJSONCache.dump_cache(self.directory, dict(self))
        assert set(cache) == set(self)
        self.clear()
        assert len(self) == 0
        return cache

    def split(self):
        return self

    def filecount(self):
        return len(self)

    def strip_empties(self):
        empties = [key for key, value in self.items() if value is None]
        for key in empties:
            del self[key]
        return len(empties)


class CombinedJSONCache(Mapping):
    writable = False

    def __init__(self, directory, dct):
        self.directory = Path(directory)
        self._dct = dict(dct)

    def filecount(self):
        return int(self._filename.is_file())

    @property
    def _filename(self):
        return self.directory / 'combined.json'

    def _dump_json(self):
        target = self._filename
        if target.exists():
            raise RuntimeError(f'Already exists: {target}')
        self.directory.mkdir(exist_ok=True, parents=True)
        write_json(target, self._dct)

    def __len__(self):
        return len(self._dct)

    def __iter__(self):
        return iter(self._dct)

    def __getitem__(self, index):
        return self._dct[index]

    @classmethod
    def dump_cache(cls, path, dct):
        cache = cls(path, dct)
        cache._dump_json()
        return cache

    @classmethod
    def load(cls, path):
        # XXX Very hacky this one
        cache = cls(path, {})
        dct = read_json(cache._filename, always_array=False)
        cache._dct.update(dct)
        return cache

    def clear(self):
        self._filename.unlink()
        self._dct.clear()

    def combine(self):
        return self

    def split(self):
        cache = MultiFileJSONCache(self.directory)
        assert len(cache) == 0
        cache.update(self)
        assert set(cache) == set(self)
        self.clear()
        return cache


def get_json_cache(directory):
    try:
        return CombinedJSONCache.load(directory)
    except FileNotFoundError:
        return MultiFileJSONCache(directory)
