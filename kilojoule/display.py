"""kiloJoule display module

This module provides classes for parsing python code as text and
formatting for display using \LaTeX. The primary use case is coverting
Jupyter notebook cells into MathJax output by showing a progression of
caculations from symbolic to final numeric solution in a multiline
equation. It makes use of sympy formula formatting and the \LaTeX code
can be stored as a string for writing to a file or copying to an
external document.
"""

from string import ascii_lowercase
from IPython.display import display, HTML, Math, Latex, Markdown

from sympy import sympify, latex
import regex as re
import inspect
from .organization import QuantityTable
from .common import get_caller_namespace
import ast

from rich import inspect

from .units import ureg, Quantity, Measurement

IN_COLAB = "google.colab" in str(get_ipython())


def enable_mathjax_colab():
    display(
        HTML(
            "<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/"
            "latest.js?config=default'></script>"
        )
    )


math_delim_begin = r""
math_delim_end = r""
math_latex_environment = r"align"
multiplication_symbol = " \cdot "

pre_sympy_latex_substitutions = {
    "Delta_(?!_)": "Delta*",
    "delta_(?!_)": "delta*",
    "Delta__": "Delta_",
    "delta__": "delta_",
    "math.log": "log",
    "np.pi": "pi",
    "math.pi": "pi",
    "Nu": "Nuplchldr",
    "_bar": "bar",
    "_ddot": "ddot",
    "_dot": "dot",
    "_ppprime|_tripleprime": "_tripprmplchldr",
    "_pprime|_doubleprime": "_doubprmplchldr",
    "_prime": "_prmplchldr",
}

post_sympy_latex_substitutions = {
    " to ": r"\\to{}",
    r"\\Delta ": r"\\Delta{}",
    r"\\delta ": r"\\delta{}",
    r"(?<!\(|\\cdot|,|\\to) (?!\\right|\\cdot|,|\\to)": r",",
    r"Nuplchldr": r"Nu",
    r"\\hbar": r"\\bar{h}",
    r"\\bar{": r"\\overline{",
    r"(infty|infinity)": r"\\infty",
    r"inf(,|})": r"\\infty\1",
    r"^inf$": r"\\infty",
    r"_\{tripprmplchldr\}|,tripprmplchldr": r"'''",
    r"_\{tripprmplchldr,": r"'''_\{",
    r"_\{doubprmplchldr\}|,doubprmplchldr": r"''",
    r"_\{doubprmplchldr,": r"''_{",
    r"_\{prmplchldr\}|,prmplchldr": r"'",
    r"_\{prmplchldr,": r"'_\{",
    r",to,": r"\\to{}",
    r",equals,": r"=",
    r",equal,": r"=",
    r"dimensionless": "",
    r"^([A-Za-z])o_molar": r"\\bar{\1}^{\\circ}",  # Changed value to being a raw string to fix an error in Python 3.9 - this line and the next may need additional edits
    r"^([A-Za-z])o_f_molar": r"\\bar{\1}^{\\circ}_f",
}

variable_name_latex_subs = {
    "np.log": r"\ln ",
    "math.log": r"\ln ",
    "log": r"\ln ",
}


def set_latex(sub_dict):
    for key, value in sub_dict.items():
        variable_name_latex_subs[key] = value


def _ast_to_string(ast_node, line_indent=""):
    next_line_indent = line_indent + "  "
    if isinstance(ast_node, ast.AST):
        return (
            ast_node.__class__.__name__
            + "("
            + ",".join(
                "\n"
                + next_line_indent
                + field_name
                + " = "
                + _ast_to_string(child_node, next_line_indent)
                for field_name, child_node in ast.iter_fields(ast_node)
            )
            + ")"
        )
    elif isinstance(ast_node, list):
        return (
            "["
            + ",".join(
                "\n" + next_line_indent + _ast_to_string(child_node, next_line_indent)
                for child_node in ast_node
            )
            + "]"
        )
    else:
        return repr(ast_node)


def to_numeric(code, namespace=None, verbose=False, line_indent=""):
    namespace = namespace or get_caller_namespace()
    if isinstance(code, str):
        if verbose:
            print(f"{line_indent}to_numeric: {code}")
        try:
            numeric_num = eval(code, namespace)
            numeric = numeric_to_string(numeric_num)
        except NameError:
            if verbose:
                print(f"Handling NameError")
            numeric = code
        except SyntaxError:
            if verbose:
                print(f"Handling SyntaxError")
            numeric = code
        except Exception as e:
            if verbose:
                print(f"{line_indent}Error in to_numeric: {e}")
                raise (e)
            numeric = "??"
    else:
        numeric = numeric_to_string(code)
    return numeric


def numeric_to_string(numeric):
    if isinstance(numeric, ureg.Quantity) or isinstance(numeric, ureg.Measurement):
        try:
            numeric = f"{numeric:.5~L}"
        except (ValueError, TypeError):
            numeric = f"{numeric:~L}"
        numeric = re.sub(r"\\\s*$", "", numeric)
    else:
        try:
            numeric = f" {numeric:.5} "
        except (ValueError, TypeError):
            numeric = f" {numeric} "
    return numeric


def to_latex(code, check_italics=False):
    if code in variable_name_latex_subs.keys():
        return variable_name_latex_subs[code]
    # print(code)
    code = str(code)
    if "[" in code:
        return index_to_latex(code, check_italics=check_italics)
    if "_over_" in code:
        return over_to_latex(code, check_italics=check_italics)
    else:
        for k, v in pre_sympy_latex_substitutions.items():
            code = re.sub(k, v, code)
        try:
            code = latex(sympify(code))
        except Exception as e:
            pass
        for key, value in post_sympy_latex_substitutions.items():
            code = re.sub(key, value, code)
        if check_italics:
            code = adjust_italics(code)
        return code


def over_to_latex(code, check_italics=False):
    """Format a variable name as a fraction if it has '_over_' in it's name"""
    num, denom = code.split("_over_")
    try:
        num_sym = to_latex(num, check_italics)
    except Exception as e:
        num_sym = num
    try:
        denom_sym = to_latex(denom, check_italics)
    except Exception as e:
        denom_sym = denom
    symbolic = f"{{ \\frac{{ {num_sym} }}{{ {denom_sym} }} }}"
    return symbolic


def index_to_latex(code, check_italics=False):
    """Format a variable name with the index in the subscript"""
    var, slc = code.split("[", 1)
    var_sym = to_latex(var)
    slc = slc[:-1]
    try:
        slc_sym = to_latex(slc)
    except Exception as e:
        slc_sym = slc
    symbolic = f"{{ {var_sym} }}_{{ {slc_sym} }}"
    return symbolic


def adjust_italics(code):
    # temporarily disable this feature
    return code
    split_code = code.split("_", 1)
    var = split_code[0]
    var_sympify = latex(sympify(var))
    if len(var) > 1 and "\\" not in var and "\\" not in var_sympify:
        var = f"\\mathrm{{{var}}}"
    else:
        var = var_sympify
    if len(split_code) > 1:
        sub = split_code[1]
        if sub[0] in "{([<":
            sub_delims = [sub[0], sub[-1]]
            subs = sub[1:-1].split(",")
        else:
            sub_delims = ["", ""]
            subs = sub.split(",")
        for i, s in enumerate(subs):
            s_sympify = latex(sympify(s))
            if len(s.strip()) > 1 and "\\" not in s and "\\" not in s_sympify:
                subs[i] = f"\\mathrm{{{s}}}"
            else:
                subs[i] = s_sympify
        sub = sub_delims[0] + ",".join(subs) + sub_delims[1]
        return f"{var}_{sub}"
    else:
        return f"{var}"


def get_node_source(node: ast.AST, input_lines: list) -> str:
    """Extract original input string between strarting and ending locations

    Returns the original source code that was parsed to produce the node
    and any text that follows the first # sybmol on the same line"""
    if isinstance(node, ast.Index):
        if hasattr(node.value, "value"):
            source = node.value.value
        elif hasattr(node.value, "id"):
            source = node.value.id
        trailing_comment = None
    else:
        # NOTE: lineno starts from 1 not 0, so you need to subtract 1 when indexing input lines
        if node.lineno == node.end_lineno:  # single line
            source = input_lines[node.lineno - 1][node.col_offset : node.end_col_offset]
        else:  # multi-line
            source = [input_lines[node.lineno - 1][node.col_offset :]]
            for line in range(node.lineno, node.end_lineno - 1):
                source.append(input_lines[line][node.col_offset :])
            source.append(
                input_lines[node.end_lineno - 1][node.col_offset : node.end_col_offset]
            )
            source = "\n".join(source)
        final_line = input_lines[node.end_lineno - 1]
        if len(final_line) > node.end_col_offset:
            trailing_source = final_line[node.end_col_offset :]
            trailing_comment = "#".join(trailing_source.split("#")[1:]).strip()
        else:
            trailing_comment = None
    return source, trailing_comment


def source_between_nodes(node1: ast.AST, node2: ast.AST, input_lines: list) -> str:
    """Extract original input string (including comments) between two nodes"""
    line_start = node1.end_lineno
    line_end = node2.lineno
    col_start = node1.end_col_offset
    col_end = node2.col_offset
    if line_start == line_end:  # single line
        line = input_lines[line_start - 1]
        line_len = len(line)
        if col_end > line_len:
            result = input_lines[line_start - 1][col_start:col_end]
    else:  # multi-line
        result = [input_lines[line_start - 1][col_start:]]
        for i in range(line_start, line_end - 1):
            result.append(input_lines[i])
        result.append(input_lines[line_end - 1][:col_end])
        result = "\n".join(result)
    return result


def source_before_node(node: ast.AST, input_lines: list) -> str:
    """Extract all text from original source code before the start of a node"""
    if node.lineno == 1:  # single line
        print(f"in source_before_node() -> single-line")
        result = input_lines[0][: node.col_offset]
    else:  # multi-line
        result = [input_lines[0]]
        for i in range(1, node.lineno - 1):
            result.append(input_lines[i])
        result.append(input_lines[node.lineno - 1][: node.col_offset])
        result = "\n".join(result)
    return result


def source_after_node(node: ast.AST, input_lines: list) -> str:
    """Extract all text from original source code after the end of a node"""
    if node.end_lineno == len(input_lines):  # single line
        result = input_lines[-1][node.end_col_offset]
    else:  # multi-line
        result = [input_lines[node.end_lineno][node.end_col_offset :]]
        for i in range(node.end_lineno, len(input_lines)):
            result.append(input_lines[i])
        result = "\n".join(result)
    return result


def strip_leading_hash(code: str) -> str:
    stripped_lines = []
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith("#"):
            line = line[1:]
        stripped_lines.append(line)
    return "\n".join(stripped_lines)


class FormatCalculation:
    """Format an assignment statement as an equation progression"""

    def __init__(
        self,
        input_node=None,
        namespace=None,
        progression=None,
        verbose=False,
        execute=False,
        source_code=None,
        input_lines=None,
        **kwargs,
    ):
        self.namespace = namespace or get_caller_namespace()
        self.input_node = input_node
        self.progression = progression
        self.verbose = verbose
        self.iscomplex = False
        self.source = source_code
        self.input_lines = input_lines
        self.kwargs = kwargs
        self._process_assignment_node()

    def display(self):
        if IN_COLAB:
            enable_mathjax_colab()
        display(Latex(self.output_string))

    def _execute_code(self, code, namespace=None):
        namespace = namespace or self.namespace
        try:
            exec(code, namespace)
            return None
        except Exception as e:
            return e

    def _process_assignment_node(self):
        node = self.input_node
        RHS_Symbolic = ""
        if self.verbose:
            print("\n*** Processing RHS ***")
        RHS = self._process_node(node.value, self.namespace, self.verbose)
        RHS_Symbolic = RHS["symbolic"]
        RHS_Numeric = RHS["numeric"]
        if self.verbose:
            print("\n*** Processing LHS ***")
        LHS_execution_error = self._execute_code(self.source)
        LHS = self._process_node(node.targets[0], self.namespace, self.verbose)
        LHS_Symbolic = LHS["symbolic"]
        if LHS_execution_error:
            LHS["numeric"] = "??"
        LHS_Numeric = LHS["numeric"]
        MID_Symbolic = ""
        if len(node.targets) > 1:
            for target in node.targets[1:]:
                targ = self._process_node(target)
                MID_Symbolic += targ["symbolic"] + " = "
        if self.verbose:
            print(
                f"LHS_Symbolic: {LHS_Symbolic}\nRHS_Symbolic: {RHS_Symbolic}\nRHS_Numeric: {RHS_Numeric}\nLHS_Numeric: {LHS_Numeric}"
            )
        result = f"{math_delim_begin}\\begin{{{math_latex_environment}}}\n  {LHS_Symbolic} &= {MID_Symbolic} {RHS_Symbolic} "
        RSymComp = RHS_Symbolic.replace(" ", "")
        RNumComp = RHS_Numeric.replace(" ", "")
        LNumComp = LHS_Numeric.replace(" ", "")
        if self.progression:
            if RSymComp != RNumComp != LNumComp:
                if self.iscomplex:
                    result += f"\\\\\n    &= {RHS_Numeric}\\\\\n    &= {LHS_Numeric}"
                else:
                    result += f" = {RHS_Numeric} = {LHS_Numeric}"
            elif RSymComp != RNumComp:
                result += f" = {RHS_Numeric} "
            elif RNumComp != LNumComp:
                result += f" = {LHS_Numeric} "
        else:
            result += f" = {LHS_Numeric}"
        result += f"\n\\end{{{math_latex_environment}}}{math_delim_end}\n"
        self.output_string = result
        if LHS_execution_error:
            if IN_COLAB:
                enable_mathjax_colab()
            display(Markdown(self.output_string))
            print(f"{LHS_execution_error}")
            for i, line in enumerate(self.source.split("\n")):
                print(f"{node.lineno+i+1}\t{line}")
            raise LHS_execution_error

    def _process_node(
        self,
        node,
        namespace=None,
        symbolic=True,
        numeric=True,
        verbose=None,
        level=1,
        **kwargs,
    ):
        namespace = namespace or self.namespace
        verbose = verbose or self.verbose
        if symbolic:
            symbolic = " "
        if numeric:
            numeric = " "
        code, trailing_comment = get_node_source(node, input_lines=self.input_lines)
        lst = []
        dct = {}
        line_indent = "  " * level
        next_level = level + 1
        if self.verbose:
            print(_ast_to_string(node, line_indent=line_indent))
            print(f"{line_indent}{numeric=}")

        # Number or String
        if isinstance(node, ast.Constant):
            symbolic = f"{node.value}"
            if numeric:
                numeric = symbolic

        # Name (Simple variable)
        elif isinstance(node, ast.Name):
            symbolic = to_latex(code)
            # symbolic = adjust_italics(code)
            if numeric:
                numeric = to_numeric(
                    code, namespace, verbose=self.verbose, line_indent=line_indent
                )

        # Subscript
        elif isinstance(node, ast.Subscript):
            val = self._process_node(node.value, level=next_level)
            slc = self._process_node(node.slice, level=next_level)
            subscript = slc["symbolic"]
            if self.verbose:
                print(f"{line_indent}  {subscript}")
            split_subscript = subscript.split()
            subscript_list = []
            for i in split_subscript:
                if "\\" not in i and len(i) > 1:
                    # subscript_list.append(f"\\mathrm{{{i}}}")
                    subscript_list.append(f"{{{i}}}")
                else:
                    subscript_list.append(i)
            subscript = "\\,".join(subscript_list)
            # symbolic = f"{{{val['symbolic']}}}_{{ {slc['numeric']} }}"
            symbolic = f"{{{val['symbolic']}}}_{{ {subscript} }}"
            if numeric:
                numeric = to_numeric(
                    code, namespace, verbose=self.verbose, line_indent=line_indent
                )

        # Index
        elif isinstance(node, ast.Index):
            if self.verbose:
                print(f"{line_indent}processing index: {code}")
            source = node
            result = self._process_node(node.value, level=next_level)
            # symbolic = f'\\mathrm{{{result["symbolic"]}}}'
            if isinstance(node.value, ast.Name):
                symbolic = f"{{{to_latex(result['numeric'])}}}"
            else:
                # symbolic = f"\\mathrm{{{to_latex(code)}}}"
                symbolic = f"{{{to_latex(code)}}}"
            if self.verbose:
                print(f"{line_indent}symbolic: {symbolic}")
            if numeric:
                numeric = to_numeric(
                    code, namespace, verbose=self.verbose, line_indent=line_indent
                )

        # Simple Math Operation
        elif isinstance(node, ast.BinOp):
            self.iscomplex = True
            left = self._process_node(node.left, level=next_level)
            right = self._process_node(node.right, level=next_level)

            # Addition
            if isinstance(node.op, ast.Add):
                code = f"{left['code']} + {right['code']}"
                symbolic = f"{left['symbolic']} + {right['symbolic']}"
                if numeric:
                    numeric = f"{left['numeric']} + {right['numeric']}"

            # Subtraction
            elif isinstance(node.op, ast.Sub):
                code = f"{left['code']} - ({right['code']})"
                if isinstance(node.right, ast.BinOp):
                    if isinstance(node.right.op, ast.Add) or isinstance(
                        node.right.op, ast.Sub
                    ):
                        right["symbolic"] = f" \\left( {right['symbolic']} \\right)"
                        right["numeric"] = f"\\left( {right['numeric']} \\right)"
                if right["numeric"].startswith("-"):
                    right["numeric"] = f"\\left( {right['numeric']} \\right)"
                symbolic = f" {left['symbolic']} - {right['symbolic']} "
                if numeric:
                    numeric = f" {left['numeric']} - {right['numeric']} "

            # Multiplication
            elif isinstance(node.op, ast.Mult):
                code = f"({left['code']})*({right['code']})"
                if isinstance(node.left, ast.BinOp):
                    if isinstance(node.left.op, ast.Add) or isinstance(
                        node.left.op, ast.Sub
                    ):
                        left["symbolic"] = f"\\left( {left['symbolic']} \\right)"
                        left["numeric"] = f"\\left( {left['numeric']} \\right)"
                if isinstance(node.right, ast.BinOp):
                    if isinstance(node.right.op, ast.Add) or isinstance(
                        node.right.op, ast.Sub
                    ):
                        right["symbolic"] = f"\\left( {right['symbolic']} \\right)"
                        right["numeric"] = f"\\left( {right['numeric']} \\right)"
                symbolic = (
                    f" {left['symbolic']} {multiplication_symbol} {right['symbolic']} "
                )
                if numeric:
                    numeric = f" {left['numeric']} {multiplication_symbol} {right['numeric']} "

            # Division
            elif isinstance(node.op, ast.Div):
                # code = f"({left['code']})/({right['code']})"
                symbolic = f"\\frac{{ {left['symbolic']} }}{{ {right['symbolic']} }}"
                if numeric:
                    numeric = f"\\frac{{ {left['numeric']} }}{{ {right['numeric']} }}"

            # Exponent
            elif isinstance(node.op, ast.Pow):
                # code = f"({left['code']})**({right['code']})"
                if isinstance(node.left, ast.BinOp):
                    left["symbolic"] = f"\\left({left['symbolic']}\\right)"
                    left["numeric"] = f"\\left({left['numeric']}\\right)"
                elif "\ " in left["numeric"] or "^" in left["numeric"]:
                    left["numeric"] = f"\\left({left['numeric']} \\right)"
                if isinstance(node.right, ast.BinOp):
                    if not isinstance(node.right.op, ast.Div):
                        right["symbolic"] = f"\\left({right['symbolic']}\\right)"
                        right["numeric"] = f"\\left({right['numeric']}\\right)"

                symbolic = f"{{{left['symbolic']}}}^{{{right['symbolic']}}}"
                if numeric:
                    numeric = f"{{{left['numeric']}}}^{{{right['numeric']}}}"

            else:
                print(f"BinOp not implemented for {node.op.__class__.__name__}")
                print(_ast_to_string(node))

        # Unary Operation
        elif isinstance(node, ast.UnaryOp):
            if self.verbose:
                print(f"{line_indent}Processing UnaryOp: {node.operand}")
            if isinstance(node.op, ast.USub):
                operand = self._process_node(node.operand, level=next_level)
                symbolic = f"-{operand['symbolic']}"
                # code = symbolic
                if numeric:
                    numeric = f"-\\left( {operand['numeric']} \\right)"
            else:
                print(
                    f"{line_indent}UnaryOp not implemented for {node.op.__class__.__name__}"
                )
                _ast_to_string(node, line_indent=line_indent)

        # Function call
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                attr = self._process_node(node.func, in_fn_call=True, level=next_level)
                fn_name_sym = attr["symbolic"]
                fn_name_code = attr["code"]
            else:
                fn_name_sym = fn_name_code = node.func.id
            fn_base_name = fn_name_code.split(".")[-1]
            # absolute value
            if fn_base_name == "abs":
                symbolic = " \\left| "
                if numeric:
                    numeric = symbolic
                symbolic_close = numeric_close = " \\right|"
            # square root
            elif fn_base_name == "sqrt":
                symbolic = numeric = "\\sqrt{"
                symbolic_close = numeric_close = "}"
            else:
                fn_name_sym = re.sub("_", r"\_", fn_name_sym)
                symbolic = numeric = f"\\mathrm{{ {fn_name_sym} }}\\left( "
                symbolic_close = numeric_close = " \\right)"
            code = f"{fn_name_code}("
            arg_idx = 0
            for idx, arg in enumerate(node.args):
                if idx > 0:
                    code += ", "
                    symbolic += ", "
                    if numeric:
                        numeric += ", "
                if verbose:
                    print(f"{line_indent}Processing Arg: {arg}")
                parg = self._process_node(arg, level=next_level)
                if verbose:
                    inspect(parg)
                code += parg["code"]
                symbolic += parg["symbolic"]
                if numeric is not None:
                    numeric += parg["numeric"]
                arg_idx += 1
            for kw in node.keywords:
                val = self._process_node(kw.value, level=next_level)
                if arg_idx > 0:
                    code += ", "
                    symbolic += ", "
                    if numeric is not None:
                        numeric += ", "
                code += f"{kw.arg} = {val['code']}"
                kw_sym = re.sub("_", r"\_", kw.arg)
                symbolic += f"\\mathrm{{ {kw_sym} }} = {val['symbolic']}"
                if numeric is not None:
                    numeric += f"\\mathrm{{ {kw_sym} }} = {val['numeric']}"
                arg_idx += 1
            code += ")"
            symbolic += symbolic_close
            if numeric is not None:
                numeric += symbolic_close

            # Quantity
            if fn_base_name == "Quantity":
                if verbose:
                    print(
                        f"{line_indent}Processing Quantity: {code}\n{line_indent}  node.args={node.args}"
                    )
                symbolic = to_numeric(
                    code,
                    namespace=self.namespace,
                    verbose=self.verbose,
                    line_indent=line_indent,
                )
                numeric = symbolic
            # .plus_minus()
            elif fn_base_name == "plus_minus":
                #                 try:
                uncertainty = node.args[0].value
                relative = False
                for kw in node.keywords:
                    if kw.arg == "relative":
                        relative = kw.value.value
                if relative:
                    unc_str = f"{100*uncertainty}\\%"
                else:
                    unc_str = f"{uncertainty}"
                if verbose:
                    print(
                        f"{line_indent}Processing plus_minus: {code}\n{line_indent}  node.args={node.args}\n{line_indent}  node.keywords={node.keywords}"
                    )
                    for kw in node.keywords:
                        print(
                            f"{line_indent}kw.arg={kw.arg}; {self._process_node(kw.value, level=next_level)}"
                        )
                        print(f"{line_indent}{kw.value.value}")
                val = self._process_node(node.func.value, level=next_level)
                symbolic = val["symbolic"]
                code = val["code"]
                if numeric is not None:
                    numeric = val["numeric"]
                quantity = eval(code)
                symbolic = f"\\left( {quantity.magnitude} \\pm {unc_str} \\right)\\ {quantity.units:~L}"
                if numeric is not None:
                    numeric = symbolic
            # .to()
            elif fn_base_name == "to":
                val = self._process_node(node.func.value, level=next_level)
                symbolic = val["symbolic"]
                code = val["code"]
                if numeric is not None:
                    numeric = val["numeric"]
            # sum()
            if fn_base_name == "sum":
                symbolic = numeric = ""
                if isinstance(node.args[0], ast.ListComp):
                    listcomp = self._process_node(
                        node.args[0],
                        join_symb="+",
                        list_delim=["", ""],
                        level=next_level,
                    )
                    elt = self._process_node(node.args[0].elt, level=next_level)
                    for comprehension in node.args[0].generators:
                        symbolic += r"\sum"
                        # numeric += r"\sum"
                        target = self._process_node(
                            comprehension.target,
                            level=next_level,
                        )
                        comp_iter = self._process_node(
                            comprehension.iter, level=next_level
                        )
                        symbolic += f"_{{{target['symbolic']}={comp_iter['symbolic']}}}"
                        # numeric += f"_{{{target['numeric']}}}"
                    symbolic += f"{{ {elt['symbolic']} }}"
                    if numeric is not None:
                        numeric += f"{{ {listcomp['numeric']} }}"

        # Attribute
        elif isinstance(node, ast.Attribute):
            val = self._process_node(node.value, nested_attr=True, level=next_level)
            code = f"{val['code']}.{node.attr}"
            symbolic = code
            if numeric is not None:
                numeric = symbolic
            if "nested_attr" not in kwargs:
                *paren, attr = code.split(".")
                symbolic = f"\\underset{{ {'.'.join(paren)} }}{{ {attr} }}"
                if "in_fn_call" in kwargs:
                    if numeric is not None:
                        numeric = symbolic
                else:
                    if self.verbose:
                        print(f"code: {code}")
                    if numeric is not None:
                        numeric = to_numeric(
                            code,
                            namespace=self.namespace,
                            verbose=self.verbose,
                            line_indent=line_indent,
                        )

        # List
        elif isinstance(node, ast.List):
            lst = []
            for i in node.elts:
                if self.verbose:
                    print(f"{line_indent}{i}")
                lst.append(self._process_node(i, level=next_level))
                if self.verbose:
                    print(f"{line_indent}{lst[-1]}")
            if self.verbose:
                print(f"{line_indent}{lst}")
            code = "[" + ",".join([i["code"] for i in lst]) + "]"
            if len(lst) <= 3:
                symbolic = "[" + ",".join([i["symbolic"] for i in lst]) + "]"
                if numeric is not None:
                    numeric = "[" + ",".join([i["numeric"] for i in lst]) + "]"
            else:
                symbolic = f"[{lst[0]['symbolic']}, \ldots, {lst[-1]['symbolic']}]"
                if numeric is not None:
                    numeric = f"[{lst[0]['numeric']}, \ldots, {lst[-1]['numeric']}]"

        # List Comprehension
        elif isinstance(node, ast.ListComp):
            if "join_symb" in kwargs:
                join_symb = kwargs["join_symb"]
            else:
                join_symb = ", "
            if "list_delim" in kwargs:
                list_delim = kwargs["list_delim"]
            else:
                list_delim = ["\\left[", "\\right]"]
            lst = eval(get_node_source(node, self.input_lines)[0], self.namespace)
            elt = self._process_node(node.elt, level=next_level)
            symbolic = f"{{\\left[ {elt['symbolic']} \\right]}}"
            for comprehension in node.generators:
                target = self._process_node(comprehension.target, level=next_level)
                comp_iter = self._process_node(comprehension.iter, level=next_level)
                symbolic += f"_{{{target['symbolic']}={comp_iter['symbolic']}}}"
            if numeric is not None:
                if len(lst) <= 3:
                    numeric = (
                        list_delim[0]
                        + join_symb.join(
                            [
                                to_numeric(
                                    i,
                                    self.namespace,
                                    self.verbose,
                                    line_indent=line_indent,
                                )
                                for i in lst
                            ]
                        )
                        + list_delim[1]
                    )
                else:
                    numeric = f"[{to_numeric(lst[0],self.namespace, line_indent=line_indent)}{join_symb}\ldots{join_symb}{to_numeric(lst[-1],self.namespace, line_indent=line_indent)}]"

        # Not Implemented
        else:
            if self.verbose:
                print(f"{line_indent}not implemented for {node.__class__.__name__}")
                _ast_to_string(node, line_indent=line_indent)
            code = get_node_source(node, self.input_lines)[0]
            symbolic = code
            if numeric is not None:
                numeric = f"{eval(code, self.namespace)}"

        output = dict(symbolic=symbolic, numeric=numeric, code=code, list=lst, dict=dct)
        if self.verbose:
            print(f"{line_indent}{type(node)} output: {output}")
        return output


class Calculations:
    """Display the calculations in the current cell"""

    def __init__(
        self,
        namespace=None,
        input_string=None,
        comments=True,
        progression=True,
        return_latex=False,
        verbose=False,
        execute=False,
        symbolic=True,
        numeric=True,
        repeat_for=False,
        repeat_n=False,
        **kwargs,
    ):
        self.namespace = namespace or get_caller_namespace()
        self.cell_string = input_string or self.namespace["_ih"][-1]
        self.input_lines = self.cell_string.split("\n")
        self.output = ""
        self.progression = progression
        self.comments = comments
        self.verbose = verbose
        self.kwargs = kwargs
        self.execute = execute
        self.symbolic = symbolic
        self.numeric = numeric
        self.cell_output = ""
        globals()["__inside_kj_display_Calculations__"] = True

        if repeat_for:
            gen_split = repeat_for.split(" in ")
            gen_var = gen_split[0][1:].strip()
            gen_range = eval(gen_split[1][:-1], self.namespace)
            if verbose:
                print(f"{gen_range}")
            for gen_val in gen_range:
                if verbose:
                    print(f"{gen_var=}; {gen_val=}")
                self.namespace[gen_var] = gen_val
                self.process_body()
        elif repeat_n:
            for i in range(repeat_n):
                self.process_body()
        else:
            self.process_body()
        globals()["__inside_kj_display_Calculations__"] = False

    def process_body(self):
        self.cell_output = ""
        self.tree = ast.parse(self.cell_string)
        for i, node in enumerate(self.tree.body):
            # print comments if enabled
            if self.comments:
                if i == 0:
                    leading_source = source_before_node(node, self.input_lines)
                else:
                    leading_source = source_between_nodes(
                        self.tree.body[i - 1], node, self.input_lines
                    )
                self.cell_output += strip_leading_hash(leading_source)
            self.process_node(node)
        if self.comments:
            trailing_source_code = source_after_node(
                self.tree.body[-1], self.input_lines
            )
            self.cell_output += strip_leading_hash(trailing_source_code)
        if IN_COLAB:
            enable_mathjax_colab()
        display(Markdown(self.cell_output))

    def process_node(self, node):
        source_code, trailing_comment = get_node_source(node, self.input_lines)
        if isinstance(node, ast.Assign):
            formatted_calc = FormatCalculation(
                node,
                namespace=self.namespace,
                progression=self.progression,
                verbose=self.verbose,
                symbolic=self.symbolic,
                numeric=self.numeric,
                source_code=source_code,
                trailing_comment=trailing_comment,
                input_lines=self.input_lines,
                **self.kwargs,
            )
            self.cell_output += formatted_calc.output_string + "\n"
        elif isinstance(node, ast.Expr):
            result = eval(source_code, self.namespace)
            self.cell_output += str(result) + "\n"
        else:
            try:
                exec(source_code, self.namespace)
            except Exception as e:
                if IN_COLAB:
                    enable_mathjax_colab()
                display(Markdown(self.cell_output))
                print(f"{e}")
                split_lines = source_code.split("\n")
                for i, line in enumerate(split_lines):
                    print(f"{node.lineno+i+1}\t{line}")
                raise (e)


class QuantityTables:
    """Display all QuantityTables in namespace"""

    def __init__(self, namespace=None, show=False, **kwargs):
        self.namespace = namespace or get_caller_namespace()

        self.output_string = ""
        for k, v in sorted(self.namespace.items()):
            if not k.startswith("_"):
                if isinstance(v, QuantityTable):
                    self.output_string += v.display(show=show)

    def __str__(self):
        return self.output_string


class Quantities:
    """Display Quantities in namespace

    If a list of variables is provided, display the specified
    variables.  Otherwise display all variables with units.
    """

    def __init__(
        self, variables=None, n_col=3, style=None, namespace=None, show=False, **kwargs
    ):
        self.namespace = namespace or get_caller_namespace()
        self.style = style
        self.n = 1
        self.n_col = n_col
        self.latex_string = f"{math_delim_begin}\\begin{{{math_latex_environment}}}{{ "
        if variables is not None:
            for variable in variables:
                self.add_variable(variable, **kwargs)
        else:
            for k, v in sorted(self.namespace.items()):
                if not k.startswith("_"):
                    try:
                        if isinstance(v, Quantity) or isinstance(v, Measurement):
                            self.add_variable(k, **kwargs)
                    except TypeError:
                        pass
        self.latex_string += f" }}\\end{{{math_latex_environment}}}{math_delim_end}"
        # use regex to remove empty line from end of align environment if it exists
        self.latex_string = re.sub(
            r"\\\\\s*{\s*}\s*\\end{" + math_latex_environment + r"}",
            r"\\end{" + math_latex_environment + r"}",
            self.latex_string,
        )
        self.latex = Latex(self.latex_string)
        if show:
            if IN_COLAB:
                enable_mathjax_colab()
            display(self.latex)

    def add_variable(self, variable, **kwargs):
        """Add a variable to the display list

        Args:
          variable:
          **kwargs:

        Returns:

        """
        symbol = to_latex(variable, check_italics=True)
        value = to_numeric(variable, self.namespace)
        boxed_styles = ["box", "boxed", "sol", "solution"]
        if self.style in boxed_styles:
            self.latex_string += r"\Aboxed{ "
        self.latex_string += symbol + r" }&={ " + value
        if self.style in boxed_styles:
            self.latex_string += r" }"
        if self.n < self.n_col:
            self.latex_string += r" }&{ "
            self.n += 1
        else:
            self.latex_string += r" }\\{ "
            self.n = 1

    def __str__(self):
        return self.latex_string


class Summary:
    """Display all quantities and QuantityTables in namespace

    If a list of variables is provided, display only those variables,
    otherwise display all quantities defined in the namespace.
    """

    def __init__(
        self,
        variables=None,
        n_col=None,
        namespace=None,
        style=None,
        show=None,
        **kwargs,
    ):
        if show is None:
            if "__inside_kj_display_Calculations__" in globals():
                if globals()["__inside_kj_display_Calculations__"]:
                    show = False
                else:
                    show = True
            else:
                show = True
        self.namespace = namespace or get_caller_namespace()
        if variables is not None:
            if n_col is None:
                n_col = 1
            self.quantities = Quantities(
                variables, n_col=n_col, namespace=self.namespace, style=style, show=show
            )
            self.state_tables = None
        else:
            if n_col is None:
                n_col = 3
            self.quantities = Quantities(
                namespace=self.namespace, n_col=n_col, show=show, **kwargs
            )
            self.state_tables = QuantityTables(
                namespace=self.namespace, show=show, **kwargs
            )

    def __str__(self):
        output_string = ""
        if self.quantities:
            output_string += "\n" + str(self.quantities)
        if self.state_tables:
            output_string += "\n" + str(self.state_tables)
        return output_string
