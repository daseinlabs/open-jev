"""Read single members of a large remote zip over HTTP range requests.

The Perception Test validation videos ship as one 75 GB zip. zipfile only needs the
central directory (at the end) plus each member's bytes, so a seekable file object that
turns reads into ``Range`` requests lets us pull a few hundred videos without the rest.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 1 << 20, timeout: float = 60.0):
        self.url, self.block, self.timeout, self.pos = url, block, timeout, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.headers.get("Accept-Ranges") != "bytes":
                raise OSError(f"{url} does not support range requests")
            self.size = int(r.headers["Content-Length"])
        self._cache: dict[int, bytes] = {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def _get(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read()

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        if n >= self.block:  # large member reads go straight through
            data = self._get(self.pos, self.pos + n)
        else:
            b0 = self.pos // self.block
            chunks = []
            for b in range(b0, (self.pos + n - 1) // self.block + 1):
                if b not in self._cache:
                    if len(self._cache) > 64:
                        self._cache.clear()
                    self._cache[b] = self._get(b * self.block, min(self.size, (b + 1) * self.block))
                chunks.append(self._cache[b])
            buf = b"".join(chunks)
            off = self.pos - b0 * self.block
            data = buf[off:off + n]
        self.pos += len(data)
        return data


def open_remote_zip(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(HTTPRangeFile(url))
