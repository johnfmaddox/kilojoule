"""kiloJoule magics module

This module provides magics classes for parsing python code as text and
formatting for display using \LaTeX. The primary use case is coverting
Jupyter notebook cells into MathJax output by showing a progression of
caculations from symbolic to final numeric solution in a multiline
equation.
"""

from .display import Calculations

from IPython.core.magic import Magics, magics_class, line_cell_magic, needs_local_scope
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring


@magics_class
class ShowCalcMagics(Magics):
    @magic_arguments()
    @argument(
        "-c",
        "--comments",
        default=False,
        action="store_true",
        help="Show comments in the output",
    )
    @argument(
        "-C",
        "--no-comments",
        dest="comments",
        action="store_false",
        help="Don't show comments in the output",
    )
    @argument(
        "-p",
        "--progression",
        default=True,
        action="store_true",
        help="Show intermediate steps",
    )
    @argument(
        "-P",
        "--no-progression",
        dest="progression",
        action="store_false",
        help="Don't show intermediate steps",
    )
    @argument(
        "-v",
        "--verbose",
        default=False,
        action="store_true",
        help="Show output for debugging",
    )
    @argument(
        "-V",
        "--not-verbose",
        dest="verbose",
        action="store_false",
        help="Don't show extra output for debugging",
    )
    @argument(
        "-r",
        "--repeat-for",
        dest="repeat_for",
        default=False,
        help="Repeat the calculations in the cell for the specified generator expression, i.e. 'x in range(4)'",
    )
    @argument(
        "-n",
        "--repeat-n",
        dest="repeat_n",
        type=int,
        default=False,
        help="Repeat the calculations in the cell the specified number of times",
    )
    @needs_local_scope
    @line_cell_magic
    def showcalc(self, line=None, cell=None, local_ns=None):
        if cell is None:
            Calculations(execute=True, namespace=local_ns, input_string=line)
        else:
            args = parse_argstring(self.showcalc, line)
            Calculations(namespace=local_ns, input_string=cell, **vars(args))


ip = get_ipython()
ip.register_magics(ShowCalcMagics)
