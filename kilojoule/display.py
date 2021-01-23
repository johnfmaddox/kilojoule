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
import re
import functools
import inspect
import logging
from .organization import PropertyTable

from .units import units, Quantity


pre_sympy_latex_substitutions = {
    "Delta_": "Delta*",
    "delta_": "delta*",
    #     'Delta*':'Delta_',
    #     'delta*':'delta_',
    "Delta__": "Delta",
    "delta__": "delta ",
    "math.log": "log",
}
post_sympy_latex_substitutions = {
    " to ": r"\to",
    r"\Delta ": r"\Delta{}",
    r"\delta ": r"\delta{}",
    " ": ",",
}

# create a list of the form: [aa, ab, ac, ... , ca, cb, cc, ..., zz]
plchldrpfixlst = [i+j+k for i in ascii_lowercase for j in ascii_lowercase for k in ascii_lowercase]


# From: https://stackoverflow.com/questions/3589311/get-defining-class-of-unbound-method-object-in-python-3/25959545#25959545
def get_class_that_defined_method(meth):
    if isinstance(meth, functools.partial):
        return get_class_that_defined_method(meth.func)
    if inspect.ismethod(meth) or (inspect.isbuiltin(meth) and getattr(meth, '__self__', None) is not None and getattr(meth.__self__, '__class__', None)):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, '__func__', meth)  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth),
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
                      None)
        if isinstance(cls, type):
            return cls
    return getattr(meth, '__objclass__', None)  # handle special descriptor objects


class EqTerm:
    """Parses a single term from an equation
    
    Instance recieves a string and treats it as a single term in an
    equation.  The string is parsed as a variable using sympy to
    create a \LaTeX representation and is evaluated in the namespace
    provided to get a numeric value for the variable.
    """

    def __init__(
        self,
        term_string,
        namespace=locals(),
        numeric_brackets="{}",
        plchldr_prefix=None,
        verbose=False,
        **kwargs,
    ):
        
        # if verbose:
        # print(f"EqTerm({term_string})")
        self.verbose = verbose
        self.namespace = namespace
        self.orig_string = term_string
        self.prefix = plchldr_prefix
        for k, v in pre_sympy_latex_substitutions.items():
            term_string = re.sub(k, v, term_string)
        self.term_string = term_string
        if ".to(" in self.term_string:
            self.term_string = self.term_string.split(".to(")[0]
        if "(" in self.term_string and ")" in self.term_string:
            if self.verbose: print('processing as function')
            self.process_function()
        elif "[" in self.term_string and "]" in self.term_string:
            # print(f"processing {self.term_string} as index")
            if self.verbose: print('processing as index')
            self.process_index(**kwargs)
        else:
            # print('processing as generic')
            try:
                self.to_sympy(**kwargs)
            except Exception as e:
                if self.verbose:
                    print(e)
                if self.verbose:
                    print(f"Failed: self.to_sympy() for {term_string}")
            try:
                self.to_numeric(**kwargs)
            except Exception as e:
                if self.verbose:
                    print(e)
                if self.verbose:
                    print(f"Failed: self.to_numeric() for {term_string}")
        try:
            self.sympified_placeholder = latex(sympify(self.placeholder))
        except Exception as e:
            if self.verbose:
                print(e)
            if verbose:
                print(f"Failed: self.sympified_placeholder for {term_string}")
            self.sympified_placeholder = self.placeholder

    def apply_local_latex_subs(self):
        """Modify the default \LaTeX string produced by sympy"""
        for key, value in post_sympy_latex_substitutions.items():
            self.latex = self.latex.replace(key, value)

    def to_sympy(self):
        """Parse string using sympify from sympy

        Parses the string stored in self.term_string. If the string is
        not an operator, i.e. +,-,*,/,etc., create a sympified latex
        representation and a sanitized-placeholder that will be
        treated as a generic term when the full equation is sympified
        later.
        """
        string = self.term_string
        if string not in "**/+-=^()":
            try:
                check = float(string)
                self.sympy_expr = string
                self.latex = string
                self.placeholder = string
                self.sanitize_placeholder()
            except Exception as e:
                if self.verbose:
                    print(e)
                try:
                    string = re.sub("\[", "_", string)
                    string = re.sub("]", "", string)
                    string = re.sub(",", "_", string)
                    self.sympy_expr = sympify(string)
                    self.latex = latex(self.sympy_expr)
                    self.placeholder = "PlcHldr" + string.replace("_", "SbScrpt")
                    self.sanitize_placeholder()
                except Exception as e:
                    if self.verbose:
                        print(e)
                    if verbose:
                        print(f"Could not sympify: {string}")
                    self.sympy_expr = string
                    self.latex = string
                    self.placeholder = string
                    self.sanitize_placeholder()
        elif string == "**":
            self.sympy_expr = "**"
            self.latex = "^"
            self.placeholder = "**"
        elif string == "*":
            self.sympy_expr = "*"
            self.latex = "\cdot"
            self.placeholder = "*"
        else:
            self.sympy_expr = string
            self.latex = string
            self.placeholder = string
        self.apply_local_latex_subs()

    def to_numeric(self, numeric_brackets="()", verbose=False, **kwargs):
        """Evaluate string in provided namespace

        Attempt to evaluate the provide in the provided namespace
        (which is usually locals() provide from the environment where
        this class was called

        Args:
          numeric_brackets:  (Default value = "()")
          verbose:  (Default value = False)
          **kwargs: 
        """
        if numeric_brackets == "{}":
            leftbrace = "\\left\\{"
            rightbrace = "\\right\\}"
        else:
            leftbrace = f"\\left{numeric_brackets[0]}"
            rightbrace = f"\\right{numeric_brackets[1]}"
        string = self.orig_string
        if string not in "**/+-=^()":
            try:
                self.numeric = eval(string, self.namespace)
                if isinstance(self.numeric, units.Quantity):
                    try:
                        self.numeric = f"{leftbrace} {self.numeric:.5~L} {rightbrace}"
                    except:
                        self.numeric = f"{leftbrace} {self.numeric:~L} {rightbrace}"
                else:
                    try:
                        self.numeric = f" {self.numeric:.5} "
                    except:
                        self.numeric = f" {self.numeric} "
                # # remove extra space after dimensionless quantities
                # self.numeric = re.sub(r"\ \right", r"\right", self.numeric) 
            except Exception as e:
                if self.verbose:
                    print(e)
                if verbose:
                    print(f"Could not get numeric value: {string}")
                self.numeric = "??"
        elif string == "**":
            self.numeric = "^"
        elif string == "*":
            self.numeric = "{\cdot}"
        else:
            self.numeric = string

    def process_function(self, numeric_brackets="()", underset_function_parents=True):
        """Parse a string representing a function call

        Set the font of the function name and evaluate the result for
        numeric display.  Include special processing for specific
        functions.

        Args: numeric_brackets (str): display brackets (Default value= "()")
        """
        if self.verbose:
            print(f"EqTerm.process_function({self.term_string})")
        if numeric_brackets == "{}":
            leftbrace = "\\left\\{"
            rightbrace = "\\right\\}"
        else:
            leftbrace = f"\\left{numeric_brackets[0]}"
            rightbrace = f"\\right{numeric_brackets[1]}"
        string = self.term_string
        function_name, arg = string.split("(")
        arg = arg[:-1]
        args = arg.split(",")
        if self.verbose:
            print(function_name, arg)
        # store parents if function is method
        *parent_name,func = function_name.split(".")
        function_obj = eval(function_name, self.namespace)
        if self.verbose: print(f'function __name__: {function_obj.__name__}')
        function_name = function_obj.__name__
        if function_name in ["Q_", "Quantity"]:
            if self.verbose:
                print("Attempting to process as a quantity")
            try:
                self.numeric = eval(self.orig_string, self.namespace)
                if isinstance(self.numeric, units.Quantity):
                    try:
                        self.numeric = f"{leftbrace} {self.numeric:.5~L} {rightbrace}"
                    except:
                        self.numeric = f"{leftbrace} {self.numeric:~L} {rightbrace}"
                else:
                    self.numeric = f" {self.numeric} "
            except Exception as e:
                if self.verbose:
                    print(e)
                if self.verbose:
                    print(f"Could not get numeric value: {string}")
                self.numeric = string
            self.latex = self.numeric
        #         elif function_name == 'abs':
        #             self.latex = r'\left|' + arg + r'\right|'
        #             self.numeric = eval(self.orig_string,self.namespace)
        #         elif isinstance(function_obj, functools.partial) and '.' in function_name:
        #             if self.verbose: print('in property loop')
        #             fluid,prop = function_name.split('.')
        #             self.latex = prop + r'_{' + fluid + r'}(' + arg + r')'
        #             self.numeric = eval(self.orig_string, self.namespace)
        else:
            if self.verbose:
                print(f"Attempting to format function: {function_name}")
            try:
                self.latex = r"\mathrm{"
                if underset_function_parents:
                    self.latex += r"\underset{"
                    self.latex += ",".join(parent_name)
                    self.latex += r"}{"
                self.latex += function_name
                if underset_function_parents:
                    self.latex += r"}"
                
                self.latex += r"}" + r"\left("
                if self.verbose: print(self.latex)
                for arg in args:
                    if self.verbose: print(f'processing arg: {arg}')
                    if "=" in arg:
                        if self.verbose: print('processing as key=value pairs')
                        if self.latex[-1] != "(":
                            self.latex += r" , "
                        key, value = arg.split("=")
                        self.latex += r"\mathrm{" + key + r"}="
                        self.latex += EqTerm(value).latex
                    else:
                        if self.verbose: print(arg)
                        
                        self.latex += EqTerm(arg).latex
                self.latex += r"\right)"
            except Exception as e:
                if self.verbose:
                    print(e)
                self.latex = string
            self.numeric = eval(self.orig_string, self.namespace)
            self.numeric = f"{self.numeric:.5}"
        self.placeholder = "FncPlcHolder" + function_name + arg
        self.sanitize_placeholder()

    def process_index(self):
        """Process a string for an index lookup
        
        If the string appears to represent the indexing of a variable,
        i.e. a dict key or list index, create a latex expression with
        the index value in the subscript and the numeric value as the
        indexed value.
        """
        string = self.term_string
        string = string.replace("[", "_")
        for i in r""""']""":
            string = string.replace(i, "")
        self.sympy_expr = sympify(string)
        self.latex = latex(self.sympy_expr)
        # print(string)
        # print("PlcHldr")
        # print(string.replace("_", "Index"))
        self.placeholder = "PlcHldr" + string.replace("_", "Indx")
        self.sanitize_placeholder()
        self.to_numeric()
        self.apply_local_latex_subs()

    def sanitize_placeholder(self):
        """Post process placehoder string

        Replace problematic characters/strings in the the placehoder used to typeset the parent equation with sympy.  This also add a three character alphabetic 
        """
        remove_symbols = "_=*+-/([])^.," + '"' + "'"
        for i in remove_symbols:
            self.placeholder = self.placeholder.replace(i, "")
        replace_num_dict = {
            "0": "Zero",
            "1": "One",
            "2": "Two",
            "3": "Three",
            "4": "Four",
            "5": "Five",
            "6": "Six",
            "7": "Seven",
            "8": "Eight",
            "9": "Nine",
        }
        for k, v in replace_num_dict.items():
            self.placeholder = self.placeholder.replace(k, v)
        self.placeholder += "End"
        # print(f'self.prefix: {self.prefix}')
        # print(f'self.placeholder: {self.placeholder}')
        self.placeholder = self.prefix + self.placeholder

    def __repr__(self):
        return self.orig_string

    def __get__(self):
        return self

class FnTerm():
        """Parse a string representing a function call

        Set the font of the function name and evaluate the result for
        numeric display.  Include special processing for specific
        functions.

        Args: numeric_brackets (str): display brackets (Default value= "()")
        """
        def __init__(namespace, numeric_brackets="()", underset_function_parents=True,verbose=False):
            logging.degug(f"FnTerm({self.term_string})")
            if numeric_brackets == "{}":
                leftbrace = "\\left\\{"
                rightbrace = "\\right\\}"
            else:
                leftbrace = f"\\left{numeric_brackets[0]}"
                rightbrace = f"\\right{numeric_brackets[1]}"
            string = self.term_string
            function_name, arg = string.split("(")
            arg = arg[:-1]
            args = arg.split(",")
            logging.debug(function_name, arg)
            # store parents if function is method
            *parent_name,func = function_name.split(".")
            function_obj = eval(function_name, namespace)
            logging.debug(f'function __name__: {function_obj.__name__}')
            function_name = function_obj.__name__
            if function_name in ["Q_", "Quantity"]:
                logging.debug("Attempting to process as a quantity")
                try:
                    self.numeric = eval(self.orig_string, self.namespace)
                    if isinstance(self.numeric, units.Quantity):
                        try:
                            self.numeric = f"{leftbrace} {self.numeric:.5~L} {rightbrace}"
                        except:
                            self.numeric = f"{leftbrace} {self.numeric:~L} {rightbrace}"
                    else:
                        self.numeric = f" {self.numeric} "
                except Exception as e:
                    if self.verbose:
                        print(e)
                    logging.debug(f"Could not get numeric value: {string}")
                    self.numeric = string
                self.latex = self.numeric
            else:
                logging.debug(f"Attempting to format function: {function_name}")
                try:
                    self.latex = r"\mathrm{"
                    if underset_function_parents:
                        self.latex += r"\underset{"
                        self.latex += ",".join(parent_name)
                        self.latex += r"}{"
                    self.latex += function_name
                    if underset_function_parents:
                        self.latex += r"}"
                    
                    self.latex += r"}" + r"\left("
                    if self.verbose: print(self.latex)
                    for arg in args:
                        logging.debug(f'processing arg: {arg}')
                        if "=" in arg:
                            logging.debug('processing as key=value pairs')
                            if self.latex[-1] != "(":
                                self.latex += r" , "
                            key, value = arg.split("=")
                            self.latex += r"\mathrm{" + key + r"}="
                            self.latex += EqTerm(value,verbose=verbose).latex
                        else:
                            logging.debug(arg)
                            logging.debug('This is a logging string')
                            if self.verbose: print(f'   Processing: {arg}')
                            self.latex += EqTerm(arg).latex
                    self.latex += r"\right)"
                except Exception as e:
                    logging.debug(e)
                    self.latex = string
                self.numeric = eval(self.orig_string, self.namespace)
                self.numeric = f"{self.numeric:.5}"
            self.placeholder = "FncPlcHolder" + function_name + arg
            self.sanitize_placeholder()
    
        
class EqFormat:
    """Process a line of text as an equation 

    For lines of text that appear to be assignments in equation form,
    split into LHS and RHS, split each side into individual terms
    using order of opertations, and parse each term using the EqTerm
    class.
    """

    def __init__(self, eq_string, namespace=locals(), verbose=False, **kwargs):
        self.verbose = verbose
        self.namespace = namespace
        self.kwargs = kwargs
        self.input_string = eq_string
        # self._parse_input_string(**kwargs)
        self.parsed_input_string, self.parsed_list = self._parse_input_string(self.input_string,**kwargs)
        self.terms_list = self._process_terms(self.parsed_list, namespace, verbose,**kwargs)
        # self._process_terms(**kwargs)

    @staticmethod
    def _parse_input_string(input_string,**kwargs):
        """Parse a line of python code into terms

        Split a line of code into terms following order of operations rules

        Args:
          **kwargs: 
        """
        operators = "*/^+-="
        parens = "()"
        brackets = "[]"
        parsed_string = '["""'
        skip_next = False
        in_string = False
        function_level = 0
        index_level = 0
        for i, char in enumerate(input_string):
            if skip_next:
                skip_next = False
            elif char in operators and function_level == 0:
                if input_string[i : i + 1] == "**":
                    char = "**"
                    skip_next = True
                parsed_string += f'""","""{char}""","""'
            elif char == "(":
                if parsed_string[-1] == '"' and function_level == 0:
                    parsed_string += f'""","""{char}""","""'
                else:
                    function_level += 1
                    parsed_string += char
            elif char == ")":
                if function_level == 0:
                    parsed_string += f'""","""{char}""","""'
                elif function_level == 1:
                    parsed_string += char
                    function_level -= 1
            else:
                parsed_string += char
            parsed_string = parsed_string.strip()
        parsed_string += '"""]'
        return parsed_string, eval(parsed_string)
        # self.parsed_input_string = parsed_string
        # self.parsed_list = eval(parsed_string)

    @staticmethod
    def _process_terms(parsed_list, namespace, verbose, **kwargs):
        """Process each term in equation using EqTerm class
        
        Apply the EqTerm class to each term in the parsed equations
        and append a placeholder prefix based on the position in the
        Args:
          **kwargs: 
        """
        ret_lst = []
        for i, term in enumerate(parsed_list):
            ret_lst.append(
                EqTerm(
                    term,
                    namespace=namespace,
                    plchldr_prefix=plchldrpfixlst[i],
                    verbose=verbose,
                    **kwargs,
                )
            )
            if verbose:
                print(ret_lst[-1].placeholder)
        terms_list = ret_lst
        return terms_list

    def _sympy_formula_formatting(self, **kwargs):
        """Sympify equation expression

        Use sympify to convert an equation string into a latex expression.  Sympy tends to rearrange equations through the sympify process, so the earlier classes/funtions in this chain introduce a prefix string to minimize the impacts of the sympy rearranging by tricking it into treating the terms as alphabetical by the order they are defines.

        Args:
          **kwargs: 

        Returns:

        """
        LHS_plchldr, *MID_plchldr, RHS_plchldr = "".join(
            [i.placeholder for i in self.terms_list]
        ).split("=")
        if self.verbose:
            print(MID_plchldr)
        LHS_latex_plchldr = latex(sympify(LHS_plchldr))
        RHS_latex_plchldr = latex(sympify(RHS_plchldr), order="grevlex")
        LHS_latex_symbolic = str(LHS_latex_plchldr)
        RHS_latex_symbolic = str(RHS_latex_plchldr)
        LHS_latex_numeric = str(LHS_latex_plchldr)
        RHS_latex_numeric = str(RHS_latex_plchldr)
        for i in self.terms_list:
            LHS_latex_symbolic = LHS_latex_symbolic.replace(
                i.sympified_placeholder, i.latex
            )
            if self.verbose: print(RHS_latex_symbolic)
            RHS_latex_symbolic = RHS_latex_symbolic.replace(
                i.sympified_placeholder, i.latex
            )
            LHS_latex_numeric = LHS_latex_numeric.replace(
                i.sympified_placeholder, str(i.numeric)
            )
            RHS_latex_numeric = RHS_latex_numeric.replace(
                i.sympified_placeholder, str(i.numeric)
            )
        if len(self.terms_list) > 3 and not len(MID_plchldr):
            LHS_latex_numeric = re.sub(
                "^\\\\left\((.*)\\\\right\)$", "\g<1>", LHS_latex_numeric
            )
            latex_string = "\\[\n  \\begin{aligned}{ "
            latex_string += LHS_latex_symbolic
            latex_string += r" }&={ "
            latex_string += " }\\\\\n    &={ ".join(
                [RHS_latex_symbolic, RHS_latex_numeric, LHS_latex_numeric]
            )
            latex_string += " }\n  \\end{aligned}\n\\]\n"
        else:
            latex_string = "\\[\n  \\begin{aligned}\n    { "
            latex_string += LHS_latex_symbolic
            latex_string += " }&={ "
            if RHS_latex_symbolic.strip() != LHS_latex_numeric.strip():
                latex_string += RHS_latex_symbolic
                latex_string += r" } = {"
            LHS_latex_numeric = re.sub(
                "^\\\\left\((.*)\\\\right\)$", "\g<1>", LHS_latex_numeric
            )
            latex_string += LHS_latex_numeric
            latex_string += " }\n  \\end{aligned}\n\\]\n"
        return latex_string


class Calculations:
    """Displaye the calculations in the current cell"""

    def __init__(
        self,
        namespace=locals(),
        comments=False,
        progression=True,
        return_latex=False,
        verbose=False,
        **kwargs,
    ):
        self.namespace = namespace
        self.cell_string = self.namespace["_ih"][-1]
        self.lines = self.cell_string.split("\n")
        self.verbose = verbose
        self.output = ""
        for line in self.lines:
            self.process_line(line, comments=comments, verbose=verbose, **kwargs)

    def process_line(self, line, comments, verbose=False, **kwargs):
        """Parse a single line of Python code

        Args:
          line (str): line of Python code
          comments (bool): True to show comments (Default value = False) 
          verbose (bool): show debug information (Default value = False)
          **kwargs: 

        Returns:

        """
        try:
            if "Calculations(" in line or "SC(" in line:
                pass
            elif line.strip().startswith("print"):
                pass
            elif line.startswith("#"):
                if comments:
                    processed_string = re.sub("^#", "", line)
                    self.output += re.sub("#", "", line) + r"<br/>"  # + '\n'
                    display(Markdown(processed_string))
            elif "=" in line:
                if "#" in line:
                    line, comment = line.split("#")
                    if "hide" in comment or "noshow" in comment:
                        raise ValueError
                eq = EqFormat(line, namespace=self.namespace, verbose=verbose, **kwargs)
                processed_string = eq._sympy_formula_formatting(**kwargs)
                self.output += processed_string
                display(Latex(processed_string))
        except Exception as e:
            if verbose:
                print(e)
                print(f"Failed to format: {line}")


class PropertyTables:
    """Display all StatesTables in namespace"""

    def __init__(self, namespace, **kwargs):
        self.namespace = namespace

        for k, v in sorted(namespace.items()):
            if not k.startswith("_"):
                if isinstance(v, PropertyTable):
                    v.display()


class Quantities:
    """Display Quantities in namespace 

    If a list of variables is provided, add each variable to list for
    display.  Otherwise add all quantities in the namespace to the
    list for display.
    """

    def __init__(self, namespace, variables=None, n_col=3, style=None, **kwargs):
        self.namespace = namespace
        self.style = style
        self.n = 1
        self.n_col = n_col
        self.latex_string = r"\[\begin{aligned}{ "
        if variables is not None:
            for variable in variables:
                self.add_variable(variable, **kwargs)
        else:
            for k, v in sorted(namespace.items()):
                if not k.startswith("_"):
                    if isinstance(v, units.Quantity):
                        self.add_variable(k, **kwargs)
        self.latex_string += r" }\end{aligned}\]"
        self.latex = self.latex_string
        display(Latex(self.latex_string))

    def add_variable(self, variable, **kwargs):
        """Add a variable to the display list

        Args:
          variable: 
          **kwargs: 

        Returns:

        """
        term = EqTerm(variable, namespace=self.namespace, **kwargs)
        symbol = term.latex
        boxed_styles = ["box", "boxed", "sol", "solution"]
        value = re.sub("^\\\\left\((.*)\\\\right\)$", "\g<1>", str(term.numeric))
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

    def __init__(self, namespace, variables=None, n_col=None, style=None, **kwargs):
        self.namespace = namespace
        if variables is not None:
            if n_col is None:
                n_col = 1
            Quantities(self.namespace, variables, n_col=n_col, style=style)
        else:
            if n_col is None:
                n_col = 3
            self.quantities = Quantities(self.namespace, n_col=n_col, **kwargs)
            self.state_tables = PropertyTables(self.namespace, **kwargs)


def _parse_input_string(input_string,**kwargs):
    """Parse a line of python code into terms

    Split a line of code into terms following order of operations rules

    Args:
      **kwargs: 
    """
    operators = "*/^+-="
    parens = "()"
    brackets = "[]"
    parsed_string = '["""'
    skip_next = False
    in_string = False
    function_level = 0
    index_level = 0
    for i, char in enumerate(input_string):
        if skip_next:
            skip_next = False
        elif char in operators and function_level == 0:
            if input_string[i : i + 1] == "**":
                char = "**"
                skip_next = True
            parsed_string += f'""","""{char}""","""'
        elif char == "(":
            if parsed_string[-1] == '"' and function_level == 0:
                parsed_string += f'""","""{char}""","""'
            else:
                function_level += 1
                parsed_string += char
        elif char == ")":
            if function_level == 0:
                parsed_string += f'""","""{char}""","""'
            elif function_level == 1:
                parsed_string += char
                function_level -= 1
        else:
            parsed_string += char
        parsed_string = parsed_string.strip()
    parsed_string += '"""]'
    return parsed_string, eval(parsed_string)
    # self.parsed_input_string = parsed_string
    # self.parsed_list = eval(parsed_string)

def _process_terms(parsed_list, namespace, verbose, **kwargs):
    """Process each term in equation using EqTerm class
    
    Apply the EqTerm class to each term in the parsed equations
    and append a placeholder prefix based on the position in the
    Args:
      **kwargs: 
    """
    ret_lst = []
    for i, term in enumerate(parsed_list):
        ret_lst.append(
            EqTerm(
                term,
                namespace=namespace,
                plchldr_prefix=plchldrpfixlst[i],
                verbose=verbose,
                **kwargs,
            )
        )
        if verbose:
            print(ret_lst[-1].placeholder)
    terms_list = ret_lst
    return terms_list
