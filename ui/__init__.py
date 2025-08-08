"""User interface helpers.

The ``ui`` package currently exposes only the CLI helper used by
``main.py``.  In the future this package could be expanded to include a
web or desktop GUI built with frameworks such as Streamlit or Tkinter.
"""

from .cli import parse_arguments  # noqa: F401

__all__ = ["parse_arguments"]
