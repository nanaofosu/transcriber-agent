"""Summarisation helper package.

This package bundles together the logic used to generate concise summaries
from a transcript.  Currently it exposes a single function,
:func:`generate_summary`, but additional utilities (for example, sentiment
analysis) could live here in the future.
"""

from .summary_agent import generate_summary  # noqa: F401

__all__ = ["generate_summary"]
