"""Dependency-free static review server with byte ranges for video seeking."""
from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import shutil


class StaticReviewHandler(SimpleHTTPRequestHandler):
    """Serve one inclusive HTTP byte range; unsupported ranges return 416."""

    def log_message(self, format, *args):
        # A detached launcher can lose its log pipe without losing the review UI.
        try:
            super().log_message(format, *args)
        except (OSError, ValueError):
            pass

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        self._range_bytes = None
        requested = self.headers.get("Range")
        path = Path(self.translate_path(self.path))
        if not requested or not path.is_file():
            return super().send_head()
        try:
            stream = path.open("rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        stat = path.stat()
        modified = self.date_time_string(stat.st_mtime)
        # We do not issue ETags. An unmatched If-Range validator requires the
        # full current representation instead of an arbitrary partial response.
        if self.headers.get("If-Range") not in (None, modified):
            stream.close()
            return super().send_head()
        size = stat.st_size
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
        start, end = 0, size-1
        try:
            if match is None or size == 0 or not any(match.groups()):
                raise ValueError("Unsupported range")
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size-1) if last else size-1
                if start >= size or end < start:
                    raise ValueError("Unsatisfiable range")
            else:
                suffix = int(last)
                if suffix <= 0:
                    raise ValueError("Invalid suffix")
                start = max(0, size-suffix)
        except ValueError:
            stream.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        stream.seek(start)
        self._range_bytes = end-start+1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self._range_bytes))
        self.send_header("Last-Modified", modified)
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        remaining = self._range_bytes
        if remaining is None:
            shutil.copyfileobj(source, outputfile)
            return
        while remaining:
            block = source.read(min(64 * 1024, remaining))
            if not block:
                break
            outputfile.write(block)
            remaining -= len(block)


class StaticReviewServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


def create_static_review_server(directory, port=8765):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("Review directory does not exist")
    return StaticReviewServer(("127.0.0.1", port), partial(StaticReviewHandler, directory=str(directory)))
