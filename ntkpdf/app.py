"""
ntkpdf.app.py

Module contains the main class for the ntkpdf app.
"""

import pathlib

from colibri.config import Environment
from validphys.app import App, providers

from ntkpdf.config import ntkConfig
from ntkpdf.model import install as _install_pdf_model_loader

# colibri's NTK providers load the PDF model themselves, by fit name, from inside
# worker threads -- so this is the only way to give them ntkpdf's model instead of
# colibri-n3fit's. Done here rather than in ``ntkpdf/__init__.py`` (which would drag
# jax/keras into every bare ``import ntkpdf``) and not in ``NTKApp.__init__`` (which
# the API path never calls). Both entry points import this module: the CLI via
# ``NTKApp``, the API because ``ntkpdf.api`` imports ``ntk_providers`` from here.
_install_pdf_model_loader()

validphys_providers = providers
colibri_providers = [
    "colibri.ntk.eigenvalues",
    "colibri.ntk.eigenvector",
    "colibri.ntk.plotntk",
    "colibri.ntk.ntk",
]

ntk_providers = validphys_providers + colibri_providers + \
  [
    "ntkpdf.data_theory",
    "ntkpdf.summary",
    "ntkpdf.pdfgrids",
    "ntkpdf.ntkdecomposition",
    "ntkpdf.hessian",
    "ntkpdf.plotting.pdfplots_providers",
    "ntkpdf.plotting.loss_providers",
    "ntkpdf.plotting.feature_plots",
  ]


class NTKApp(App):
    config_class = ntkConfig
    environment_class = Environment

    @property
    def default_style(self):
        # Override validphys's default (small.mplstyle) so the CLI applies the
        # NTKPDF style. reportengine's init_style still lets `--style` win, since
        # it takes precedence over default_style.
        from ntkpdf.plotting.style import NTKPDF_STYLE
        return NTKPDF_STYLE

    def __init__(self, name="ntkpdf"):
        super().__init__(name, ntk_providers)


    def get_commandline_arguments(self, cmdline=None):
        """Get commandline arguments"""
        args = super().get_commandline_arguments(cmdline)
        if args["output"] is None:
            args["output"] = pathlib.Path(args["config_yml"]).stem
        return args


def main():
    a = NTKApp(name="ntkpdf")
    a.main()

if __name__ == "__main__":
    main()
