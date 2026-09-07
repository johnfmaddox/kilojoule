# Notebook → PDF export

`export_notebook_to_pdf.py` turns a kilojoule Jupyter notebook into a
properly-typeset PDF, working around two problems in plain
`jupyter nbconvert --to pdf`:

> **Exporting a single notebook from inside itself?** Use
> `kilojoule.export.export_pdf()` instead — same fix, same calling
> convention as `export_html()` (`from kilojoule.export import export_pdf;
> export_pdf()`), no separate script/kernel needed. This script is for
> headless, from-scratch batch conversion (CI, a whole directory of
> notebooks, or a specific `--kernel`/`--python`).

1. **`\cancel{}` fails to compile.** kilojoule's energy/entropy-balance
   derivations use `\cancel{}` to strike through eliminated terms.
   nbconvert's default LaTeX template doesn't load the `cancel` package, so
   every cancellation throws `! Undefined control sequence.` (xelatex still
   emits a PDF in nonstopmode, but the strikethroughs are broken/missing).
2. **`Summary()` tables render as a flat list of numbers, not a table.**
   `Summary()` (and any `pandas.DataFrame.to_html()` output) is emitted under
   the `text/markdown` mimetype with a raw HTML `<table>` embedded in it.
   nbconvert converts `text/markdown` outputs via pandoc, and pandoc's
   markdown reader doesn't understand embedded raw HTML tables when the
   target is LaTeX — it drops the tags but keeps each cell's text as its own
   paragraph.

The script fixes both by executing the notebook, rewriting embedded HTML
tables as native LaTeX `tabular` blocks (pandoc passes unrecognized
`\begin{...}\end{...}` blocks straight through to LaTeX — the same mechanism
that already lets `\begin{align}` derivations come through untouched),
running `nbconvert --to latex`, patching in `\usepackage{cancel}`, and
compiling with a LaTeX engine (`xelatex` by default -- see `--engine` below).

## Installation

You need three things:

1. **A Python environment with kilojoule installed**, registered as a
   Jupyter kernel:

   ```bash
   pip install kilojoule ipykernel nbconvert
   python -m ipykernel install --user --name kilojoule --display-name "Python (kilojoule)"
   ```

   This registers a kernel named `kilojoule` (used as `--kernel kilojoule`
   below). If you already have a kernel with kilojoule installed under a
   different name, use that name instead — list kernels with
   `jupyter kernelspec list`.

2. **A TeX distribution** (MiKTeX or TeX Live) with `xelatex`, `lualatex`,
   or `pdflatex` on your `PATH`. The script auto-detects whichever of the
   three is available, in that order (`xelatex`/`lualatex` support
   Unicode/system fonts natively via `fontspec`, which matters for things
   like a bare "°" in a unit label; `pdflatex` is the most limited of the
   three but is also the most commonly preinstalled) — or pass `--engine`
   to name one explicitly. The `cancel`, `booktabs`, `longtable`, `array`,
   and `graphicx` packages are required — all are part of a standard
   MiKTeX/TeX Live install and will auto-install on first use if MiKTeX's
   "install packages on the fly" setting is on.

3. **This script has no extra Python dependencies of its own** beyond the
   standard library — it shells out to `nbconvert` and the LaTeX engine.

## Usage

```bash
python export_notebook_to_pdf.py "Some Notebook.ipynb"
```

This produces `Some Notebook.pdf` next to the source notebook.

Common options:

```bash
# Set the PDF's title explicitly (default: whatever nbconvert derives,
# usually the notebook's filename)
python export_notebook_to_pdf.py "Ex0.1 Filling a Tank.ipynb" \
    --title "Example 0.1: Filling a Tank"

# Use a specific kernel/interpreter and output location
python export_notebook_to_pdf.py notebook.ipynb \
    --kernel kilojoule \
    --python /path/to/python \
    --outdir build/ --outname Ch03_Example2

# Keep the intermediate .tex/.aux/.log files and the executed/table-fixed
# .ipynb copies for debugging a failed compile
python export_notebook_to_pdf.py notebook.ipynb --keep-intermediate

# Force a specific LaTeX engine (default: auto-detect xelatex, then
# lualatex, then pdflatex -- whichever is found first on PATH)
python export_notebook_to_pdf.py notebook.ipynb --engine lualatex
```

Run `python export_notebook_to_pdf.py --help` for the full option list
(`--passes` to control the number of LaTeX passes, `--no-strip-cocalc` to
disable stripping of CoCalc-only cells).

### What "stripping CoCalc-only cells" means

Notebooks authored on CoCalc sometimes end with a cell like:

```python
from kilojoule.export import export_html
export_html()
```

which reads a CoCalc-injected environment variable
(`COCALC_JUPYTER_FILENAME`) and raises `KeyError` anywhere else. By default,
the script drops any code cell referencing `COCALC_JUPYTER_FILENAME` or
`kilojoule.export` before executing. Pass `--no-strip-cocalc` if that's not
what you want, or edit the `COCALC_ONLY_PATTERNS` tuple near the top of the
script to match additional patterns.

## Troubleshooting

- **`jupyter_client.kernelspec.NoSuchKernel`** — the `--kernel` name doesn't
  match a registered kernel. Run `jupyter kernelspec list` to see what's
  available, or register one per the Installation section above.
- **`requires one of xelatex, lualatex, pdflatex on PATH`** — none of the
  three is installed/on `PATH`. Install a TeX distribution (see
  Installation above), or if one is installed but not detected, check that
  its `bin/` directory is actually on `PATH` in the environment running
  this script (not just in an interactive shell's `PATH`).
- **LaTeX errors on something other than `\cancel` or a table** — rerun
  with `--keep-intermediate` and inspect the generated `.tex`/`.log` files
  directly; the script leaves them in `--outdir` for exactly this reason.
- **A table still overflows the page** — `html_table_to_latex()` wraps every
  table in a conditional `\resizebox` that only shrinks it when wider than
  `\linewidth`, so this shouldn't happen; if it does, the table likely has
  columns this script's simple regex-based HTML parser mis-parsed (check
  for nested tags or multi-row headers) — inspect the `.tex` output near the
  offending table.
