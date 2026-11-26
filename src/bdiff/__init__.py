"""bdiff: surface, name and track behavior changes between model versions under controlled input conditions."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("behavior-diff")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0"
