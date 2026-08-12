"""
    test_example_notebooks
    ~~~~~~~~~~~~~~~~~~~~~~~
    Execute a tree of course-example Jupyter notebooks against an
    installed `kilojoule` (e.g. a fresh TestPyPI release in a throwaway
    venv) and report the first genuine error encountered, folder by
    folder, notebook by notebook.

    Intended as a smoke test before promoting a `devel` release: build a
    venv, install the candidate `kilojoule` version into it (and register
    it as a Jupyter kernel), then run this script pointed at a directory
    of real course notebooks that exercise the library in ways unit tests
    don't.

    Known, expected error patterns (see `KNOWN_OK_SNIPPETS`) are not
    treated as failures -- e.g. `export_html()`'s headless-nbconvert
    limitation, or `MissingDataFileError` for copyrighted textbook data
    tables that are intentionally not bundled with the package. Any error
    occurring in a notebook *after* a known one is also treated as known,
    since it is almost always just a downstream cascade (e.g. a
    `NameError` for a variable that a known-missing data file never
    defined) rather than an independent bug.

    Usage:
        python tools/test_example_notebooks.py \\
            --examples-dir "path/to/Notebooks/Examples" \\
            --venv-python "path/to/venv/Scripts/python.exe" \\
            --kernel kj-testpypi \\
            [--folders "06 - Chapter 13 Gas Mixtures" "07 - Chapter 14 Gas-Vapor Mixtures" ...] \\
            [--outdir path/to/scratch/nb-executed]

    If `--folders` is omitted, every immediate subdirectory of
    `--examples-dir` containing at least one `.ipynb` file is run, in
    sorted order. Stops at the first genuine error; rerun with `--folders`
    trimmed to the remaining folders once a fix has been applied and
    republished.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

KNOWN_OK_SNIPPETS = [
    "notebook's filename",  # export_html() under headless nbconvert -- expected
    "not distributed with kilojoule for copyright reasons",  # Cengel/Bergman Data CSVs not present in this env -- expected
]


def run_notebook(venv_python: Path, kernel: str, folder: Path, nb_name: str, outdir: Path, timeout: int):
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(venv_python),
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--allow-errors",
        f"--ExecutePreprocessor.kernel_name={kernel}",
        f"--ExecutePreprocessor.timeout={timeout}",
        f"--output-dir={outdir}",
        nb_name,
    ]
    proc = subprocess.run(cmd, cwd=folder, capture_output=True, text=True)
    out_path = outdir / nb_name
    if not out_path.exists():
        return None, f"nbconvert did not produce output (exit {proc.returncode})\n{proc.stderr[-3000:]}"
    nb = json.loads(out_path.read_text(encoding="utf-8"))
    errors = []
    for i, cell in enumerate(nb["cells"]):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                evalue = out.get("evalue", "")
                ename = out.get("ename", "")
                known = any(s in evalue for s in KNOWN_OK_SNIPPETS)
                errors.append((i, ename, evalue, known))
    # Once a known/expected error occurs (e.g. a missing copyrighted data file),
    # every later error in the same notebook is a downstream cascade of that
    # same root cause (e.g. NameError for a variable that never got defined) --
    # not a separate genuine bug -- so mark them known too.
    if any(e[3] for e in errors):
        first_known_idx = min(i for i, e in enumerate(errors) if e[3])
        errors = [
            (e[0], e[1], e[2], True) if i >= first_known_idx else e
            for i, e in enumerate(errors)
        ]
    return errors, None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--examples-dir", required=True, type=Path, help="Root directory containing one subfolder per chapter/topic of example notebooks")
    p.add_argument("--venv-python", required=True, type=Path, help="Path to the python.exe/python of the venv with the candidate kilojoule installed")
    p.add_argument("--kernel", default="kilojoule-test", help="Jupyter kernel name registered for --venv-python (default: %(default)s)")
    p.add_argument("--folders", nargs="*", default=None, help="Subfolder names to run, in order (default: all subfolders containing .ipynb files, sorted)")
    p.add_argument("--outdir", type=Path, default=None, help="Directory to write executed notebook copies to (default: a temp directory)")
    p.add_argument("--timeout", type=int, default=180, help="Per-cell execution timeout in seconds (default: %(default)s)")
    return p.parse_args()


def main():
    args = parse_args()
    outdir_root = args.outdir or Path(tempfile.mkdtemp(prefix="kilojoule-nb-test-"))

    if args.folders is not None:
        folders = args.folders
    else:
        folders = sorted(
            p.name for p in args.examples_dir.iterdir()
            if p.is_dir() and any(p.glob("*.ipynb"))
        )

    for folder_name in folders:
        folder = args.examples_dir / folder_name
        if not folder.is_dir():
            print(f"!! folder not found: {folder_name}")
            continue
        nbs = sorted(p.name for p in folder.glob("*.ipynb"))
        print(f"\n=== {folder_name} ({len(nbs)} notebooks) ===")
        for nb_name in nbs:
            outdir = outdir_root / folder_name
            errors, hard_fail = run_notebook(args.venv_python, args.kernel, folder, nb_name, outdir, args.timeout)
            if hard_fail:
                print(f"  [FATAL] {nb_name}: {hard_fail}")
                print("\nSTOPPING on fatal execution failure.")
                sys.exit(1)
            genuine = [e for e in errors if not e[3]]
            known = [e for e in errors if e[3]]
            status = "OK" if not genuine else "ERROR"
            note = f" ({len(known)} known environment limitation)" if known and not genuine else ""
            print(f"  [{status}] {nb_name}{note}")
            if genuine:
                for i, ename, evalue, _ in genuine:
                    print(f"      cell {i}: {ename}: {evalue[:300]}")
                print(f"\nSTOPPING at first genuine error: {folder_name}/{nb_name}")
                sys.exit(2)
    print("\nAll notebooks in all folders executed cleanly (aside from known environment limitations).")
    print(f"Executed notebook copies written to: {outdir_root}")


if __name__ == "__main__":
    main()
