"""kiloJoule solve module

This module provides classes for parsing python code as text and
formatting for non-linear solving.  
"""

from string import ascii_lowercase
from IPython.display import display, HTML, Math, Latex, Markdown
from sympy import sympify, latex

# import re
import regex as re
import functools
import inspect
import logging
from .organization import QuantityTable
from .common import get_caller_namespace
import ast
from .units import units, Quantity


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


def to_numeric(code, namespace=None):
    namespeace = namespace or get_caller_namespace()
    try:
        numeric = eval(code, namespace)
        numeric = numeric_to_string(numeric)
    except Exception as e:
        numeric = "??"
    return numeric


def numeric_to_string(numeric):
    if isinstance(numeric, units.Quantity):
        try:
            numeric = f"{numeric:.5~L}"
        except:
            numeric = f"{numeric:~L}"
    else:
        try:
            numeric = f" {numeric:.5} "
        except:
            numeric = f" {numeric} "
    return numeric


def to_latex(code):
    if "[" in code:
        return index_to_latex(code)
    if code in __variable_latex_subs__.keys():
        return __variable_latex_subs__[code]
    else:
        for k, v in pre_sympy_latex_substitutions.items():
            code = re.sub(k, v, code)
        code = latex(sympify(code))
        for key, value in post_sympy_latex_substitutions.items():
            code = re.sub(key, value, code)
        return code


def index_to_latex(code):
    var, slc = code.split("[", 1)
    var_sym = to_latex(var)
    slc = slc[:-1]
    try:
        slc_sym = to_latex(slc)
    except Execption as e:
        slc_sym = slc
    symbolic = f"{{ {var_sym} }}_{{ {slc_sym} }}"
    return symbolic


class FormatCalculation:
    """Format an assignment statement as a equation progression"""

    def __init__(
        self,
        input_node=None,
        namespace=None,
        progression=None,
        verbose=False,
        execute=False,
        **kwargs,
    ):
        self.namespace = namespace or get_caller_namespace()
        self.input_node = input_node
        self.progression = progression
        self.verbose = verbose
        self.iscomplex = False
        self.kwargs = kwargs
        if execute:
            exec(self.input_string, self.namespace)
        self._process_line()

    def display(self):
        display(Latex(self.output_string))

    def _process_line(self):
        line = self.input_node
        LHS = self._process_node(line.targets[0], self.namespace, self.verbose)
        LHS_Symbolic = LHS["symbolic"]
        LHS_Numeric = LHS["numeric"]
        MID_Symbolic = ""
        if len(line.targets) > 1:
            for target in line.targets[1:]:
                targ = self._process_node(target)
                MID_Symbolic += targ["symbolic"] + " = "
        RHS_Symbolic = ""
        RHS = self._process_node(line.value, self.namespace, self.verbose)
        RHS_Symbolic = RHS["symbolic"]
        RHS_Numeric = RHS["numeric"]
        if self.verbose:
            print(
                f"LHS_Symbolic: {LHS_Symbolic}\nRHS_Symbolic: {RHS_Symbolic}\nRHS_Numeric: {RHS_Numeric}\nLHS_Numeric: {LHS_Numeric}"
            )
        result = (
            f"\\begin{{aligned}}\n  {LHS_Symbolic} &= {MID_Symbolic} {RHS_Symbolic} "
        )
        if self.progression:
            if RHS_Symbolic.strip() != RHS_Numeric.strip() != LHS_Numeric.strip():
                if self.iscomplex:
                    result += f"\\\\\n    &= {RHS_Numeric}\\\\\n    &= {LHS_Numeric}"
                else:
                    result += f" = {RHS_Numeric} = {LHS_Numeric}"
            elif RHS_Symbolic.strip() != RHS_Numeric.strip():
                result += f" = {RHS_Numeric} "
            elif RHS_Numeric.strip() != LHS_Numeric.strip():
                result += f" = {LHS_Numeric} "
        else:
            result += f" = {LHS_Numeric}"
        result += "\n\end{aligned}\n"
        self.output_string = result

    def _process_node(self, node, namespace=None, verbose=False, **kwargs):
        namespace = namespace or get_caller_namespace()
        symbolic = ""
        numeric = ""
        code = ""

        if verbose:
            print(_ast_to_string(node))

        # Number or String
        if isinstance(node, ast.Constant):
            symbolic = f"{node.value}"
            numeric = symbolic
            if isinstance(node.value, str):
                code = f'"{node.value}"'
            else:
                code = symbolic

        # Simple variable
        elif isinstance(node, ast.Name):
            code = node.id
            symbolic = to_latex(code)
            numeric = to_numeric(code, namespace)

        # Subscript
        elif isinstance(node, ast.Subscript):
            val = self._process_node(node.value, namespace)
            slc = self._process_node(node.slice, namespace)
            code = f"{val['code']}[{slc['code']}]"
            symbolic = f"{{{val['symbolic']}}}_{{ {slc['symbolic']} }}"
            numeric = to_numeric(code, namespace)

        # Index
        elif isinstance(node, ast.Index):
            result = self._process_node(node.value, namespace)
            code = result["code"]
            symbolic = result["symbolic"]
            numeric = to_numeric(code, namespace)

        # Simple Math Operation
        elif isinstance(node, ast.BinOp):
            self.iscomplex = True
            left = self._process_node(node.left, namespace)
            right = self._process_node(node.right, namespace)

            # Addition
            if isinstance(node.op, ast.Add):
                code = f"{left['code']} + {right['code']}"
                symbolic = f"{left['symbolic']} + {right['symbolic']}"
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
                symbolic = f" {left['symbolic']} - {right['symbolic']} "
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
                numeric = (
                    f" {left['numeric']} {multiplication_symbol} {right['numeric']} "
                )

            # Division
            elif isinstance(node.op, ast.Div):
                code = f"({left['code']})/({right['code']})"
                symbolic = f"\\frac{{ {left['symbolic']} }}{{ {right['symbolic']} }}"
                numeric = f"\\frac{{ {left['numeric']} }}{{ {right['numeric']} }}"

            # Exponent
            elif isinstance(node.op, ast.Pow):
                code = f"({left['code']})**({right['code']})"
                if isinstance(node.left, ast.BinOp):
                    left["symbolic"] = f"\\left({left['symbolic']}\\right)"
                    left["numeric"] = f"\\left({left['numeric']}\\right)"
                elif "\ " in left["numeric"]:
                    left["numeric"] = f"\\left({left['numeric']} \\right)"
                if isinstance(node.right, ast.BinOp):
                    if not isinstance(node.right.op, ast.Div):
                        right["symbolic"] = f"\\left({right['symbolic']}\\right)"
                        right["numeric"] = f"\\left({right['numeric']}\\right)"
                symbolic = f"{left['symbolic']}^{right['symbolic']}"
                numeric = f"{left['numeric']}^{right['numeric']}"

            else:
                print(f"BinOp not implemented for {node.op.__class__.__name__}")
                _ast_to_string(node)

        # Unary Operation
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                operand = self._process_node(node.operand, namespace)
                symbolic = f"-{operand['symbolic']}"
                numeric = f"-\\left( {operand['numeric']} \\right)"
            else:
                print(f"UnaryOp not implemented for {node.op.__class__.__name__}")
                _ast_to_string(node)

        # Function call
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                attr = self._process_node(node.func, namespace, in_fn_call=True)
                fn_name_sym = attr["symbolic"]
                fn_name_code = attr["code"]
            else:
                fn_name_sym = fn_name_code = node.func.id
            fn_base_name = fn_name_code.split(".")[-1]
            # absolute value
            if fn_base_name == "abs":
                symbolic = numeric = " \\left| "
                symbolic_close = numeric_close = " \\right|"
            # square root
            elif fn_base_name == "sqrt":
                symbolic = numeric = "\\sqrt{"
                symbolic_close = numeric_close = "}"
            else:
                symbolic = numeric = f"\\mathrm{{ {fn_name_sym} }}\\left( "
                symbolic_close = numeric_close = " \\right)"
            code = f"{fn_name_code}("
            arg_idx = 0
            for arg in node.args:
                if arg_idx > 0:
                    code += ", "
                    symbolic += ", "
                    numeric += ", "
                parg = self._process_node(arg, namespace)
                code += parg["code"]
                symbolic += parg["symbolic"]
                numeric += parg["numeric"]
                arg_idx += 1
            for kw in node.keywords:
                val = self._process_node(kw.value, namespace)
                if arg_idx > 0:
                    code += ", "
                    symbolic += ", "
                    numeric += ", "
                code += f"{kw.arg} = {val['code']}"
                symbolic += f"\\mathrm{{ {kw.arg} }} = {val['symbolic']}"
                numeric += f"\\mathrm{{ {kw.arg} }} = {val['numeric']}"
                arg_idx += 1
            code += ")"
            symbolic += symbolic_close
            numeric += symbolic_close

            # Quantity
            if fn_base_name == "Quantity":
                symbolic = to_numeric(code, namespace)
                numeric = symbolic
            # .to()
            elif fn_base_name == "to":
                val = self._process_node(node.func.value, namespace)
                symbolic = val["symbolic"]
                code = f'{val["code"]}.to("{node.args[0].value}")'
                numeric = to_numeric(code, namespace)

        # Attribute
        elif isinstance(node, ast.Attribute):
            val = self._process_node(node.value, namespace, nested_attr=True)
            code = f"{val['code']}.{node.attr}"
            symbolic = code
            numeric = symbolic
            if "nested_attr" not in kwargs:
                *paren, attr = code.split(".")
                symbolic = f"\\underset{{ {'.'.join(paren)} }}{{ {attr} }}"
                if "in_fn_call" in kwargs:
                    numeric = symbolic
                else:
                    numeric = to_numeric(code, namespace)

        else:
            print(f"not implemented for {node.__class__.__name__}")
            _ast_to_string(node)

        output = dict(symbolic=symbolic, numeric=numeric, code=code)
        return output


class SolveCell:
    """Solve the equation in the current cell"""

    def __init__(
        self,
        namespace=None,
        input_string=None,
        comments=True,
        progression=True,
        return_latex=False,
        verbose=False,
        execute=False,
        **kwargs,
    ):
        self.namespace = namespace or get_caller_namespace()
        self.cell_string = input_string or self.namespace["_ih"][-1]
        self.output = ""
        self.progression = progression
        self.comments = comments
        self.verbose = verbose
        self.kwargs = kwargs
        if execute:
            exec(self.cell_string, self.namespace)
        self.input = self.filter_string(self.cell_string)
        self.process_input_string(self.input)

    def process_code(self, string):
        output = ""
        self.parsed_tree = ast.parse(string)
        for line in self.parsed_tree.body:
            if isinstance(line, ast.Assign):
                formatted_calc = FormatCalculation(
                    line,
                    namespace=self.namespace,
                    progression=self.progression,
                    verbose=self.verbose,
                    **self.kwargs,
                )
                formatted_calc.display()
                output += formatted_calc.output_string

    def process_input_string(self, string):
        if self.comments:
            lines = string.split("\n")
            code_block = ""
            for line in lines:
                if line.startswith("#"):
                    if code_block != "":
                        self.process_code(code_block)
                        code_block = ""
                    processed_string = re.sub("^#", "", line)
                    self.output += re.sub("#", "", line) + r"<br/>"  # + '\n'
                    display(Markdown(processed_string))
                else:
                    code_block += line + "\n"
            if code_block != "":
                self.process_code(code_block)
                code_block = ""
        else:
            self.process_code(string)

    def filter_string(self, string):
        result = ""
        for line in string.split("\n"):
            if (not line.startswith("#")) and ("#" in line):
                code, comment = line.split("#", 1)
                if not any(i in comment for i in "hide noshow suppress".split()):
                    result += line + "\n"
            else:
                result += line + "\n"
        return result


class QuantityTables:
    """Display all StatesTables in namespace"""

    def __init__(self, namespace=None, **kwargs):
        self.namespace = namespace or get_caller_namespace()

        for k, v in sorted(self.namespace.items()):
            if not k.startswith("_"):
                if isinstance(v, QuantityTable):
                    v.display()


class Quantities:
    """Display Quantities in namespace

    If a list of variables is provided, display the specified
    variables.  Otherwise display all variables with units.
    """

    def __init__(self, variables=None, n_col=3, style=None, namespace=None, **kwargs):
        self.namespace = namespace or get_caller_namespace()
        self.style = style
        self.n = 1
        self.n_col = n_col
        self.latex_string = r"\begin{aligned}{ "
        if variables is not None:
            for variable in variables:
                self.add_variable(variable, **kwargs)
        else:
            for k, v in sorted(self.namespace.items()):
                if not k.startswith("_"):
                    if isinstance(v, units.Quantity):
                        self.add_variable(k, **kwargs)
        self.latex_string += r" }\end{aligned}"
        self.latex = self.latex_string
        display(Latex(self.latex_string))

    def add_variable(self, variable, **kwargs):
        """Add a variable to the display list

        Args:
          variable:
          **kwargs:

        Returns:

        """
        symbol = to_latex(variable)
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


class Summary:
    """Display all quantities and StatesTables in namespace

    If a list of variables if provided, display only those variables,
    otherwise display all quantities defined in the namespace.
    """

    def __init__(
        self, variables=None, n_col=None, namespace=None, style=None, **kwargs
    ):
        self.namespace = namespace or get_caller_namespace()
        if variables is not None:
            if n_col is None:
                n_col = 1
            Quantities(variables, n_col=n_col, namespace=self.namespace, style=style)
        else:
            if n_col is None:
                n_col = 3
            self.quantities = Quantities(
                namespace=self.namespace, n_col=n_col, **kwargs
            )
            self.state_tables = QuantityTables(namespace=self.namespace, **kwargs)
