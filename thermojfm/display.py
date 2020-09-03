from IPython.display import display, HTML, Math, Latex, Markdown
from sympy import sympify, latex
import re
from .organization import PropertyTable

from thermoJFM import units


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


class EqTerm:
    def __init__(
        self,
        term_string,
        namespace=locals(),
        numeric_brackets="{}",
        verbose=False,
        **kwargs,
    ):
        if verbose:
            print(f"EqTerm({term_string})")
        self.verbose = verbose
        self.namespace = namespace
        self.orig_string = term_string
        for k, v in pre_sympy_latex_substitutions.items():
            term_string = re.sub(k, v, term_string)
        self.term_string = term_string
        if ".to(" in self.term_string:
            self.term_string = self.term_string.split(".to(")[0]
        if "(" in self.term_string and ")" in self.term_string:
            self.process_function()
        elif "[" in self.term_string and "]" in self.term_string:
            self.process_index(**kwargs)
        else:
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
        for key, value in post_sympy_latex_substitutions.items():
            self.latex = self.latex.replace(key, value)

    def to_sympy(self):
        string = self.term_string
        if string not in "**/+-=^()":
            try:
                check = float(string)
                self.sympy_expr = string
                self.latex = string
                self.placeholder = string
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
                    # self.sympified_placeholder_expr = sympify(self.placeholder)
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

    def process_function(self, numeric_brackets="()"):
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
        string = re.sub("^math.", "", string)
        # string = re.sub('^log(','ln(',string)
        string = re.sub("^np.", "", string)
        function_obj = eval(function_name, self.namespace)
        if function_name == "Q_":
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
                if verbose:
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
                print("Attempting to format function")
            try:
                self.latex = r"\mathrm{" + function_name + r"}" + r"\left("
                for arg in args:
                    if "=" in arg:
                        if self.latex[-1] != "(":
                            self.latex += r" , "
                        key, value = arg.split("=")
                        self.latex += r"\mathrm{" + key + r"}="
                        self.latex += EqTerm(value).latex
                    else:
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
        string = self.term_string
        string = string.replace("[", "_")
        for i in r""""']""":
            string = string.replace(i, "")
        self.sympy_expr = sympify(string)
        self.latex = latex(self.sympy_expr)
        self.placeholder = "PlcHldr" + string.replace("_", "Indx")
        self.sanitize_placeholder()
        self.to_numeric()
        self.apply_local_latex_subs()

    def sanitize_placeholder(self):
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

    def __repr__(self):
        return self.orig_string

    def __get__(self):
        return self


class EqFormat:
    def __init__(self, eq_string, namespace=locals(), verbose=False, **kwargs):
        self.verbose = verbose
        self.namespace = namespace
        self.kwargs = kwargs
        self.input_string = eq_string
        self._parse_input_string(**kwargs)
        self._process_terms(**kwargs)

    def _parse_input_string(self, **kwargs):
        operators = "*/^+-="
        parens = "()"
        brackets = "[]"
        parsed_string = '["""'
        skip_next = False
        in_string = False
        function_level = 0
        index_level = 0
        for i, char in enumerate(self.input_string):
            if skip_next:
                skip_next = False
            elif char in operators and function_level == 0:
                if self.input_string[i : i + 1] == "**":
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
        self.parsed_input_string = parsed_string
        self.parsed_list = eval(parsed_string)

    def _process_terms(self, **kwargs):
        ret_lst = []
        for term in self.parsed_list:
            ret_lst.append(
                EqTerm(term, namespace=self.namespace, verbose=self.verbose, **kwargs)
            )
            if self.verbose:
                print(ret_lst[-1].placeholder)
        self.terms_list = ret_lst

    def _sympy_formula_formatting(self, **kwargs):
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
        #         for i,v in enumerate(MID_plchldr):
        #             LHS_latex_plchldr += ' = '
        #             LHS_latex_plchldr +=
        #         MID_latex_symbolic = []
        #         if MID_plchldr:
        #             for i,v in enumerate(MID_plchldr):
        #                 MID_latex_symbolic[i] = str(v)
        for i in self.terms_list:
            LHS_latex_symbolic = LHS_latex_symbolic.replace(
                i.sympified_placeholder, i.latex
            )
            RHS_latex_symbolic = RHS_latex_symbolic.replace(
                i.sympified_placeholder, i.latex
            )
            LHS_latex_numeric = LHS_latex_numeric.replace(
                i.sympified_placeholder, str(i.numeric)
            )
            RHS_latex_numeric = RHS_latex_numeric.replace(
                i.sympified_placeholder, str(i.numeric)
            )
        #             if MID_plchldr:
        #                 for j,v in enumerate(MID_latex_symbolic):
        #                     MID_latex_symbolic[j] = v.replace(i.sympified_placeholder, i.latex)
        if len(self.terms_list) > 3 and not len(MID_plchldr):
            LHS_latex_numeric = re.sub(
                "^\\\\left\((.*)\\\\right\)$", "\g<1>", LHS_latex_numeric
            )
            latex_string = r"\begin{align}{ "
            latex_string += LHS_latex_symbolic
            latex_string += r" }&={ "
            latex_string += r" }\\&={ ".join(
                [RHS_latex_symbolic, RHS_latex_numeric, LHS_latex_numeric]
            )
            latex_string += r" }\end{align}"
        else:
            latex_string = r"\begin{align}{ "
            latex_string += LHS_latex_symbolic
            latex_string += r" }&={ "
            if RHS_latex_symbolic.strip() != LHS_latex_numeric.strip():
                latex_string += RHS_latex_symbolic
                latex_string += r" } = {"
            LHS_latex_numeric = re.sub(
                "^\\\\left\((.*)\\\\right\)$", "\g<1>", LHS_latex_numeric
            )
            latex_string += LHS_latex_numeric
            latex_string += r" }\end{align}"
        return latex_string


class Calculations:
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
        try:
            if "ShowCalculations(" in line or "SC(" in line:
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
    def __init__(self, namespace, **kwargs):
        self.namespace = namespace

        for k, v in sorted(namespace.items()):
            if not k.startswith("_"):
                if isinstance(v, PropertyTable):
                    v.display()


class Quantities:
    def __init__(self, namespace, variables=None, n_col=3, style=None, **kwargs):
        self.namespace = namespace
        self.style = style
        self.n = 1
        self.n_col = n_col
        self.latex_string = r"\begin{align}{ "
        if variables is not None:
            for variable in variables:
                self.add_variable(variable, **kwargs)
        else:
            for k, v in sorted(namespace.items()):
                if not k.startswith("_"):
                    if isinstance(v, units.Quantity):
                        self.add_variable(k, **kwargs)
        self.latex_string += r" }\end{align}"
        self.latex = self.latex_string
        display(Latex(self.latex_string))

    def add_variable(self, variable, **kwargs):
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
