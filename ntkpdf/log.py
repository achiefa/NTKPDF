"""
ntkpdf.log

Notebook-aware logging for the API / interactive path.

reportengine's CLI ``App`` installs ``reportengine.colors.ColorHandler`` on the root
logger (``app.py`` ``init_logging``), giving the ``[LEVEL]: message`` coloured output.
The programmatic API (``ntkpdf.api``) never goes through that ``App``, so in a notebook
nothing is configured: records below WARNING are dropped, and the blessings/ANSI colour
escapes would not render anyway.

This module provides a drop-in handler that *is* reportengine's ``ColorHandler`` outside
a notebook (so terminal/script output is unchanged) but renders each record as HTML via
``IPython.display`` inside Jupyter, plus :func:`setup_logger` to install it. ``ntkpdf.api``
calls :func:`setup_logger` on import, so API/notebook logs just appear.
"""

import html as _html
import logging

from reportengine.colors import ColorHandler


def is_in_notebook() -> bool:
    """True if running inside a Jupyter/IPython *kernel* (not a plain terminal
    IPython, which keeps the ANSI handler)."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    ip = get_ipython()
    return ip is not None and "IPKernelApp" in ip.config


class NotebookColorHandler(ColorHandler):
    """reportengine's :class:`~reportengine.colors.ColorHandler`, extended to render
    each record as HTML inside a Jupyter notebook (where the blessings/ANSI colour
    escapes do not display).

    Outside a notebook ``format``/``emit`` defer to the parent, so terminal and script
    output is byte-for-byte the reportengine behaviour.
    """

    # (CSS colour for the "[LEVEL]:" tag, bold the whole line) per level, mirroring
    # the parent's blessings palette.
    _html_levels = {
        logging.DEBUG: ("royalblue", False),
        logging.INFO: ("green", False),
        logging.WARNING: ("darkorange", False),
        logging.ERROR: ("red", True),
        logging.CRITICAL: ("white", True),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_notebook = is_in_notebook()

    def format(self, record):
        if not self.in_notebook:
            return super().format(record)
        color, bold = self._html_levels.get(record.levelno, (None, False))
        tag = _html.escape(f"[{record.levelname}]:")
        message = _html.escape(record.getMessage())
        tag_style = "font-weight:bold"
        if color:
            tag_style += f";color:{color}"
        if record.levelno >= logging.CRITICAL:
            tag_style += ";background-color:darkred"
        div_style = "font-weight:bold" if bold else ""
        return (
            f'<div style="{div_style}">'
            f'<span style="{tag_style}">{tag}</span> {message}'
            f"</div>"
        )

    def emit(self, record):
        if not self.in_notebook:
            super().emit(record)
            return
        try:
            from IPython.display import HTML, display

            display(HTML(self.format(record)))
        except Exception:
            self.handleError(record)


def setup_logger(level=logging.INFO):
    """Install :class:`NotebookColorHandler` on the root logger (idempotent).

    Mirrors reportengine's CLI ``init_logging`` for the API path -- sets the root level
    and quietens matplotlib -- but with the notebook-aware handler. A no-op if one of
    these handlers is already installed, so importing the API more than once (or a later
    ``logging.basicConfig``, which is itself a no-op once a handler exists) does not
    double up the output.
    """
    root = logging.getLogger()
    if any(isinstance(h, NotebookColorHandler) for h in root.handlers):
        return
    root.setLevel(level)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    root.addHandler(NotebookColorHandler())
