# Existing docstring issues

Python sources and docstrings have not been changed. The authored `.rst` pages
describe the implementation where an existing docstring is stale.

## Sphinx diagnostics

These diagnostics remain visible; the build does not suppress them or rewrite
docstrings through an autodoc hook. Line numbers below refer to the Python
source, not Sphinx's relative `:docstring ...:N` locations.

| Source | Problem |
| --- | --- |
| `ntkpdf/ntkdecomposition.py:443-450` (`feature_grids_at_epoch`) | An inline literal beginning with double backticks before `(nreplicas, n, n)` spans a paragraph break. Docutils reports “Inline literal start-string without end-string.” The paragraph also describes the old `EvolutionOperator` dependency instead of `FeatureColumns`. |
| `ntkpdf/utils.py:62-67` (`flav_to_evol_matrix`) | Uneven indentation of prose, an equation and a following list produces “Unexpected indentation” (an ERROR) and “Block quote ends without a blank line; unexpected unindent.” The equation should eventually use an explicit literal or math block. |
| `ntkpdf/utils.py:92-94` (`get_rot_evol_to_flav`) | An indented equation immediately followed by unindented prose without a separating blank line produces “Definition list ends without a blank line; unexpected unindent.” |

## Import-time Python warnings

A fresh build also prints two Python 3.12 `SyntaxWarning`s, separately from
Sphinx's four diagnostics: `<unknown>:13` for the invalid escape `\G`, and
`<unknown>:57` for `\s`. Static parsing confirms the former in the upstream
`n3fit/layers/DIS.py` module docstring (`CPU\GPU`). The latter's originating
file has not been established; the build supplies only `<unknown>`. Neither
warning is suppressed. Static parsing of the local ntkpdf sources produced
neither warning.

## Other source-checked documentation inconsistencies

| Source | Problem |
| --- | --- |
| `ntkpdf/ntkdecomposition.py:85-102` (`NTKDecomposition`) | Describes a stacked, padded ensemble of `NTKStats`, but the class holds a single replica's DataFrames/arrays. Its attribute declarations also duplicate autodoc's dataclass field entries under Napoleon's default attribute-directive setting. The docs use `napoleon_use_ivar = True` to avoid duplicate index entries without changing the docstring. |
| `ntkpdf/ntkdecomposition.py:224-226` (`compute_ntk_decomposition_by_replica_at_epoch`) | Claims the result is unpadded, although the signature defaults to `pad=True` and the implementation pads. |
| `ntkpdf/ntkdecomposition.py:313-328` (`compute_ntk_decomposition_ensemble`) | Documents old parameter names (`eigenvalues_stat`, `eigenvectors_stat`, `M`), has an unfinished `training` description and names a nonexistent return class. The actual arguments include the train/validation pairs, `training` selects the subset, and the return object is `EvolutionOperator`. |
| `ntkpdf/ntkdecomposition.py:584-585` (`h_val_grid`) | Calls the cached `eigenvalue_grid` “epoch-independent”; the grid contains epoch-dependent eigenvalues. |
| `ntkpdf/plotting/pdfplots_providers.py:232-233` (`plot_pdfs_at_epoch`) | Says `normalise_to: 1` gives a ratio to the underlying law. `plot_grids` uses a 1-based grid index and the underlying law is second, so the correct index is 2. |
| `ntkpdf/plotting/pdfplots_providers.py:284-285` (`plot_pdfs_all_epochs`) | Repeats the same incorrect normalisation advice. Here the underlying law is appended after all selected snapshot grids, so its 1-based index depends on their number. |
| `ntkpdf/vpinterface.py:138` (`NTKPDF.load`) | Says it returns `N3LHAPDFSet`; the implementation returns the cached `NTKLHAPDF` wrapper. |
