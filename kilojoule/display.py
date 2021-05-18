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
import functools
import inspect
import logging
from .organization import PropertyTable
from .common import get_caller_namespace
import ast

from .units import units, Quantity

multiplication_symbol = ' \cdot '

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
    r"dimensionless": "",
}

__variable_latex_subs__ = {
    'np.log':'\ln ',
    'math.log':'\ln ',
    'log':'\ln ',
}
def set_latex(var_name_string, latex_string):
    __variable_latex_subs__[var_name_string]=latex_string

def ast_to_string(ast_node, line_indent=''):
    next_line_indent = line_indent + '  '
    if isinstance(ast_node, ast.AST):
        return (ast_node.__class__.__name__
                + '('
                + ','.join('\n' + next_line_indent + field_name + ' = ' + ast_to_string(child_node, next_line_indent)
                           for field_name, child_node in ast.iter_fields(ast_node))
                + ')')
    elif isinstance(ast_node, list):
        return ('['
                + ','.join('\n' + next_line_indent + ast_to_string(child_node, next_line_indent)
                           for child_node in ast_node)
                + ']')
    else:
        return repr(ast_node) 

    
def to_numeric(code, namespace=None):
    namespeace = namespace or get_caller_namespace()
    try:
        numeric = eval(code, namespace)
        numeric = numeric_to_string(numeric)
    except Exception as e:
        numeric = code
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
    if code in __variable_latex_subs__.keys():
        return __variable_latex_subs__[code]
    else:
        for k, v in pre_sympy_latex_substitutions.items():
            code = re.sub(k, v, code)
        code = latex(sympify(code))    
        for key, value in post_sympy_latex_substitutions.items():
            code = re.sub(key, value, code)
        return code


def process_node(node, namespace=None, verbose=False, **kwargs):
    namespace = namespace or get_caller_namespace()
    symbolic = ''
    numeric = ''
    code = ''

    if verbose:
        print(ast_to_string(node))
    
    # Number or String
    if isinstance(node, ast.Constant):
        symbolic = f'{node.value}'
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
        val = process_node(node.value, namespace)
        slc = process_node(node.slice, namespace)
        code = f"{val['code']}[{slc['code']}]"
        symbolic = f"{val['symbolic']}_"+"{"+f"{slc['symbolic']}"+"}"
        numeric = to_numeric(code, namespace)
        
    # Index
    elif isinstance(node, ast.Index):
        result = process_node(node.value, namespace)
        code = result['code']
        symbolic = result['symbolic']
        numeric = to_numeric(code, namespace)
        
    # Simple Math Operation
    elif isinstance(node, ast.BinOp):
        left = process_node(node.left, namespace)
        right = process_node(node.right, namespace)
        
        # Addition
        if isinstance(node.op, ast.Add):
            code = left['code'] + ' + ' + right['code']            
            symbolic = left['symbolic'] + ' + ' + right['symbolic']
            numeric = left['numeric'] + ' + ' + right['numeric']
            
        # Subtraction
        elif isinstance(node.op, ast.Sub):
            code = left['code']+'-'+'('+right['code']+')'
            if isinstance(node.right, ast.BinOp):
                right['symbolic'] = '''\\left(''' + right['symbolic'] + '''\\right)'''
                right['numeric'] = '''\\left(''' + right['numeric'] + '''\\right)'''                
            symbolic = left['symbolic'] + ' - ' + right['symbolic']
            numeric = left['numeric'] + ' - ' + right['numeric']
            
        # Multiplication
        elif isinstance(node.op, ast.Mult):
            code = f"({left['code']})*({right['code']})"
            if isinstance(node.left, ast.BinOp):
                if not isinstance(node.left.op, ast.Div):
                    left['symbolic'] = f"\\left({left['symbolic']}\\right)"
                    left['numeric'] = f"\\left({left['numeric']}\\right)"                    
            if isinstance(node.right, ast.BinOp):
                if not isinstance(node.right.op, ast.Div):
                    right['symbolic'] = f"\\left({right['symbolic']}\\right)"
                    right['numeric'] = f"\\left({right['numeric']}\\right)"
            symbolic = f"{left['symbolic']} {multiplication_symbol} {right['symbolic']}"
            numeric = f"{left['numeric']} {multiplication_symbol} {right['numeric']}"            
            
        # Division
        elif isinstance(node.op, ast.Div):
            code = f"({left['code']})/({right['code']})"
            symbolic = f"\\frac{{{left['symbolic']}}}{{{right['symbolic']}}}"
            numeric = f"\\frac{{{left['numeric']}}}{{{right['numeric']}}}"
        
        # Exponent
        elif isinstance(node.op, ast.Pow):
            code = f"({left['code']})**({right['code']})"
            if isinstance(node.left, ast.BinOp):
                left['symbolic'] = f"\\left({left['symbolic']}\\right)"
                left['numeric'] = f"\\left({left['numeric']}\\right)"                    
            if isinstance(node.right, ast.BinOp):
                if not isinstance(node.right.op, ast.Div):
                    right['symbolic'] = f"\\left({right['symbolic']}\\right)"
                    right['numeric'] = f"\\left({right['numeric']}\\right)"
            symbolic = f"{left['symbolic']}^{right['symbolic']}"
            numeric = f"{left['numeric']}^{right['numeric']}" 

        else:
            print(f'BinOp not implemented for {node.op.__class__.__name__}')
            ast_to_string(node)
                
    # Function call
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            attr = process_node(node.func, namespace)
            fn_name_sym = attr['symbolic']
            fn_name_code = attr['code']
        else:
            fn_name_sym = fn_name_code = node.func.id
        fn_base_name = fn_name_code.split('.')[-1]
        # absolute value
        if fn_base_name == 'abs':
            symbolic = numeric = '\\left|'
            symbolic_close = numeric_close = '\\right|'
        # square root
        elif fn_base_name == 'sqrt':
            symbolic = numeric = '\\sqrt{'
            symbolic_close = numeric_close = '}'
        else:
            symbolic = numeric = f"\\mathrm{{{fn_name_sym}}}\\left("
            symbolic_close = numeric_close = '\\right)'
        code = f"{fn_name_code}("
        arg_idx = 0
        for arg in node.args:
            if arg_idx>0:
                code += ', '
                symbolic += ', '
                numeric += ', '
            parg = process_node(arg, namespace)
            code += parg['code']
            symbolic += parg['symbolic']
            numeric += parg['numeric']
            arg_idx += 1
        for kw in node.keywords:
            val = process_node(kw.value, namespace)
            if arg_idx>0:
                code += ', '
                symbolic += ', '
                numeric += ', '
            code += f"{kw.arg} = {val['code']}"
            symbolic += f"\\mathrm{{{kw.arg}}} = {val['symbolic']}"
            numeric += f"\\mathrm{{{kw.arg}}} = {val['numeric']}"
            arg_idx += 1
        code += ')'
        symbolic += symbolic_close
        numeric += symbolic_close

        # Quantity
        if fn_base_name == 'Quantity':
            symbolic = numeric_to_string(eval(code, namespace))
            numeric = symbolic
        # .to()
        elif fn_base_name == 'to':
            val = process_node(node.func.value, namespace)
            symbolic = val['symbolic']
            code = f'{val["code"]}.to("{node.args[0].value}")'
            numeric = numeric_to_string(eval(code, namespace))

    # Attribute
    elif isinstance(node, ast.Attribute):
        val = process_node(node.value, namespace, nested_attr=True)
        code = f"{val['code']}.{node.attr}"
        symbolic = code
        numeric = symbolic
        if 'nested_attr' not in kwargs:
            *paren, attr = code.split('.')
            symbolic = '''\\underset{''' + '.'.join(paren) + '''}{''' + attr + '''}'''
            numeric = symbolic
    
    else:
        print(f'not implemented for {node.__class__.__name__}')
        ast_to_string(node)

    output = dict(symbolic=symbolic, numeric=numeric,code=code)
    return output


class Calculations:
    """Display the calculations in the current cell"""

    def __init__(
        self,
        namespace=None,
        comments=True,
        progression=True,
        return_latex=False,
        verbose=False,
        **kwargs,
    ):
        self.namespace = namespace or get_caller_namespace()
        self.output = ''
        self.show_progression = progression
        self.comments = comments
        self.verbose = verbose
        self.cell_string = self.namespace["_ih"][-1]
        self.input = self.filter_string(self.cell_string)
        self.process_input_string(self.input)
        

    def process_code(self, string):
        output = ''
        self.parsed_tree = ast.parse(string)
        for line in self.parsed_tree.body:
            if isinstance(line, ast.Assign):
                LHS = process_node(line.targets[0], self.namespace, self.verbose)
                LHS_Symbolic = LHS['symbolic']
                LHS_Numeric = LHS['numeric']
                MID_Symbolic = ''
                if len(line.targets)>1:
                    for target in line.targets[1:]:
                        targ = process_node(target)
                        MID_Symbolic += targ['symbolic'] + ' = '
                RHS_Symbolic = ''
                RHS = process_node(line.value, self.namespace, self.verbose)
                RHS_Symbolic = RHS['symbolic']
                RHS_Numeric = RHS['numeric']
                result = f"\\begin{{aligned}}\n  {LHS_Symbolic} &= {MID_Symbolic} {RHS_Symbolic} "
                if self.show_progression:
                    if RHS_Symbolic.strip() != RHS_Numeric.strip() != LHS_Numeric.strip():
                        result += f"\\\\\n    &= {RHS_Numeric}\\\\\n    &= {LHS_Numeric}"
                    elif RHS_Symbolic.strip() != RHS_Numeric.strip():
                        result += f" = {RHS_Numeric} "
                    elif RHS_Numeric.strip() != LHS_Numeric.strip():
                        result += f" = {LHS_Numeric} "
                else:
                    result += f" = {LHS_Numeric}"
                result += "\n\end{aligned}\n"
                self.output += result
                display(Latex(result))
            
    def process_input_string(self, string):
        if self.comments:
            lines = string.split("\n")
            code_block = ''
            for line in lines:
                if line.startswith("#"):
                    if code_block != '':
                        self.process_code(code_block)
                        code_block = ''
                    processed_string = re.sub("^#", "", line)
                    self.output += re.sub("#", "", line) + r"<br/>"  # + '\n'
                    display(Markdown(processed_string))
                else:
                    code_block += line + '\n'
            if code_block != '':
                self.process_code(code_block)
                code_block = ''
        else:
            self.process_code(string)

    def filter_string(self, string):
        result = ''
        for line in string.split('\n'):
            if (not line.startswith("#")) and ('#' in line):
                #code, comment = line.split("#",1)
                pass
            else:
                result += line + '\n'
        return result
                

class PropertyTables:
    """Display all StatesTables in namespace"""

    def __init__(self, namespace=None, **kwargs):
        self.namespace = namespace or get_caller_namespace()

        for k, v in sorted(self.namespace.items()):
            if not k.startswith("_"):
                if isinstance(v, PropertyTable):
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
        value =to_numeric(variable, self.namespace)
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

    def __init__(self, variables=None, n_col=None, namespace=None, style=None, **kwargs):
        self.namespace = namespace or get_caller_namespace()
        if variables is not None:
            if n_col is None:
                n_col = 1
            Quantities(variables, n_col=n_col, namespace=self.namespace, style=style)
        else:
            if n_col is None:
                n_col = 3
            self.quantities = Quantities(namespace=self.namespace, n_col=n_col, **kwargs)
            self.state_tables = PropertyTables(namespace=self.namespace, **kwargs)
