"""kiloJoule solution_hash module

This module provides classes for storing or checking hashes of values
stored in variables.  The instructor can use these tools to create a
solution hash file that can be distributed to students.  The student
can then check their solutions against the stored hashes to verify
accuracy.
"""

from .common import get_caller_namespace
from .units import Quantity
from .display import numeric_to_string, to_latex, to_numeric, Latex
from IPython.display import display, Markdown

import hashlib
import json
from os.path import exists
import sys

import shutil
import os

from sigfig import round
import warnings
import re

default_hash_filename = ".solution_hashes"
default_student_dir = "student/"
default_sigfigs = 4
default_machine_zero = 1e-12

try:
    import emoji

    sol_symbols = {"correct": "✅", "partial": "🚧", "incorrect": "❌"}
except:
    sol_symbols = {
        "correct": "\mathrm{Correct}",
        "partial": "\mathrm{Partial}",
        "incorrect": "\mathrm{incorrect}",
    }
sol_legend = r"\begin{align*}" + "\n"
sol_legend += (
    f"{sol_symbols['correct']}&: "
    + r"\mathrm{All\, significant\, figures\, are\, correct}"
    + r"\\"
    + "\n"
)
sol_legend += (
    f"{sol_symbols['partial']}&: "
    + r"\mathrm{The\, first\, significant\, figure\, is\, correct}"
    + r"\\"
    + "\n"
)
sol_legend += (
    f"{sol_symbols['incorrect']}&: "
    + r"\mathrm{No\, significant\, figures\, are\, correct}"
    + "\n"
)
sol_legend += r"\end{align*}"


def name_and_date(Name):
    if Name == "Jane Doe":
        raise ValueError("Update the Name variable above")
    # from IPython.display import display, Markdown
    from datetime import datetime
    import pytz

    today = datetime.now(pytz.timezone("US/Central"))
    display(Markdown(Name))
    display(Markdown(today.strftime("%B %d, %Y (%I:%M %p CDT)")))


class QuietError(Exception):
    """Base class for other exceptions"""

    pass


class IncorrectValueError(QuietError):
    """Raised when solution hash doesn't match stored hash"""

    pass


def quiet_hook(kind, message, traceback):
    if QuietError in kind.__bases__:
        print(
            "{0}: {1}".format(kind.__name__, message)
        )  # Only print Error Type and Message
    else:
        sys.__excepthook__(
            kind, message, traceback
        )  # Print Error Type, Message and Traceback


sys.excepthook = quiet_hook


def hashq(obj, units=None, sigfigs=None, round_machine_zero=True, verbose=False, **kwargs):
    if isinstance(obj, Quantity):
        base = obj.to_base_units()
        base_mag = base.magnitude
        base_units = base.units
    else:
        base = base_mag = obj
        base_units = units
    if verbose:
        print(f"{base_mag=}; {base_units=}")
    if round_machine_zero:
        try:
            if abs(base_mag) < default_machine_zero:
                base_mag = 0.0
        except TypeError:
            pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(base_mag, int) and sigfigs is not None:
                base_mag = float(base_mag)
            rounded_base_mag = round(base_mag, sigfigs=sigfigs)
            if rounded_base_mag == 0:
                rounded_base_mag = (
                    0  # fix for case of `-0.0` being interpreted differently than `0.0`
                )
            str_rep = str(rounded_base_mag)
    except:
        if base_mag == 0:
            ssbase_mag = (
                0  # fix for case of `-0.0` being interpreted differently than `0.0`
            )
        str_rep = str(base_mag)
    encoded_str = str_rep.encode()
    hash_obj = hashlib.md5(encoded_str)
    hexa_value = hash_obj.hexdigest()
    if verbose:
        if isinstance(obj, Quantity):
            print(
                f"value:{obj:~} ->  base units:{base:~} -> magnitude:{base_mag} -> rounded:{str_rep}-> hash:{hexa_value}"
            )
        else:
            print(
                f"value:{obj} ->  base units:{base} -> magnitude:{base_mag} -> rounded:{str_rep}-> hash:{hexa_value}"
            )
    if units is not None:
        str_rep += f" {base_units}"
    hash_dict = dict(
        hash=hexa_value,
        units=units,
        sigfigs=sigfigs,
    )
    return hexa_value, str_rep, hash_dict


def str_to_sol_list(sol_list):
    # if the first argument is a string rather than a list, convert it to a list of variable names by splitting
    # on delimeters in the order: comma (","), semicolon (";"), or space (" ")
    if isinstance(sol_list, str):
        split_char = " "
        if "," in sol_list:
            split_char = ","
        elif ";" in sol_list:
            split_char = ";"
        sol_list = [i.strip() for i in sol_list.split(split_char) if len(i.strip()) > 0]
    return sol_list


def check_solutions(sol_list, n_col=3, namespace=None, legend=False, **kwargs):
    """Accepts a list of solution check specifications and call `check_solution()` for each.

    Accepts a list of strings or a list of dictionaries.
    """
    namespace = namespace or get_caller_namespace()
    kwargs["namespace"] = namespace
    n = 1
    result_str = r"\begin{align} "
    sol_list = str_to_sol_list(sol_list)
    for sol in sol_list:
        if isinstance(sol, str):
            result_str += check_solution(sol, single_check=False, **kwargs)
        elif isinstance(sol, dict):
            result_str += check_solution(**sol, single_check=False, **kwargs)
        if n < n_col:
            result_str += r" \quad & "
            n += 1
        else:
            result_str += r" \\ "
            n = 1
    # use regex to remove empty line from end of align environment if it exists
    result_str += r" \end{align}"
    result_str = re.sub(r"\\\\\s*{\s*}\s*\\end{align}", r"\n\\end{align}", result_str)
    display(Latex(result_str))
    if legend:
        display(Latex(sol_legend))

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


def check_solution(
    name,
    value=None,
    units=None,
    sigfigs=None,
    namespace=None,
    prefix="",
    filename=default_hash_filename,
    verbose=False,
    raise_error=False,
    single_check=True,
    legend=False,
    **kwargs,
):
    namespace = namespace or get_caller_namespace()
    key = prefix + name

    # If no value was provided, evaluate the variable name in the namespace
    try:
        value = value or eval(name, namespace)
    except (NameError, KeyError):
        # NameError if undefined variable
        # KeyError if undefined index in a dict
        value = "??"
    # if verbose:
    #     print(f"key={key}; value={value:sigfigs}")
    try:
        result_str_body = f"{to_latex(name)} &= {numeric_to_string(value)} && "
    except:
        result_str_body = f"{to_latex(name)} &= {value} && "
    # Read the corresponding entry form the hash db
    try:
        hash_db = read_solution_hash(key)
        # Set units and sigfigs to correspond to the hash db unless specified in the arguments
        units = units or hash_db["units"]
        if units == "None":
            units = None
        sigfigs = sigfigs or hash_db["sigfigs"]
        target_hashes = hash_db["hashes"]
        firt_sigfig_hashes = hash_db["first_sigfig_hashes"]
        # target_hashes = [str(i['hash']) for i in read_solution_hash(key)]
        if "round_machine_zero" in hash_db.keys():
            round_machine_zero = hash_db["round_machine_zero"]
        else:
            round_machine_zero = True
        try:
            hash_value, str_rep, hash_dict = hashq(
                value,
                units=units,
                sigfigs=sigfigs,
                round_machine_zero=round_machine_zero,
                verbose=verbose,
                **kwargs,
            )
            if verbose:
                print(f"hash: {hash_value} <-> target: {target_hashes}")
            assert hash_value in target_hashes
            result_str_body += sol_symbols["correct"]
        except AssertionError as err:
            # Try first sigfig only
            try:
                first_sigfig_hash_value, str_rep, hash_dict = hashq(
                    value, units=units, sigfigs=1, verbose=verbose, **kwargs
                )
                if verbose:
                    print(
                        f"first sigfig hash: {first_sigfig_hash_value} <-> target: {firt_sigfig_hashes}"
                    )
                assert first_sigfig_hash_value in firt_sigfig_hashes
                result_str_body += sol_symbols["partial"]
            except AssertionError as err2:
                result_str_body += sol_symbols["incorrect"]
                msg = f"Hash Mismatch for {key}: {hash_value} not in {target_hashes}"
                if raise_error:
                    raise IncorrectValueError(msg)
    except KeyError as err:
        if verbose:
            print(f"{name} not in hash database")
        if raise_error:
            raise err
    if single_check:
        result_str = f"\\begin{{align}}{result_str_body}\\end{{align}}"
        display(Latex(result_str))
    else:
        return result_str_body  # +r'\\'


def read_solution_hashes(filename=default_hash_filename):
    if exists(filename):
        # Load existing hashes if the file exits
        with open(filename, "r") as f:
            hashes = json.load(f)
    else:
        # Create an empty dict if no previous file exists
        hashes = {}
    return hashes


def store_solutions(
    sol_list=None,
    namespace=None,
    filename=default_hash_filename,
    copy_to_student=True,
    student_dir=default_student_dir,
    **kwargs,
):
    """Accepts a list of solution storage specifications and calls `store_solution()` for each.

    Accepts a list of strings or a list of dictionaries.
    """
    namespace = namespace or get_caller_namespace()
    kwargs["namespace"] = namespace
    sol_list = str_to_sol_list(sol_list)
    for sol in sol_list:
        if isinstance(sol, str):
            store_solution(sol, **kwargs)
        elif isinstance(sol, dict):
            store_solution(**sol, **kwargs)
    if copy_to_student:
        os.makedirs(student_dir, exist_ok=True)
        shutil.copy2(default_hash_filename, student_dir)


def store_solution(
    name,
    value=None,
    units=None,
    sigfigs=default_sigfigs,
    append=False,
    namespace=None,
    prefix="",
    filename=default_hash_filename,
    round_machine_zero=True,
    verbose=False,
    **kwargs,
):
    """Store the hash of a value

    Generate a hash of the value stored in `name` and store that hash in a file
    to check student solutions against.  Use the value of `name` in the calling
    namespace unless `value` is provided.  Convert the value to the specified
    units and round to the specified number of significant figures before hashing.
    `prefix` will be added to the variable name before storing to avoid conflicts
    if multiple documents are using the same storage file.
    """
    namespace = namespace or get_caller_namespace()
    # If no value was provided, evaluate the variable name in the namespace
    value = value or eval(name, namespace)
    if units is not None:
        value = value.to(units)
    if isinstance(value, Quantity):
        units = value.units
    key = prefix + name
    if verbose:
        print(f"key={key}; value={value}")
    # Read in existing hash database
    hash_db = read_solution_hashes(filename)
    if isinstance(value, list):
        hashes = [
            str(hashq(i, units, sigfigs, verbose=verbose, **kwargs)[0]) for i in value
        ]
        first_sigfig_hashes = [
            str(hashq(value, units, sigfigs=1, verbose=verbose, **kwargs)[0])
            for i in value
        ]
        for val in value.magnitude:
            if round_machine_zero and val < default_machine_zero:
                round_machine_zero = False
    else:
        hashes = [str(hashq(value, units, sigfigs, verbose=verbose, **kwargs)[0])]
        first_sigfig_hashes = [
            str(hashq(value, units, sigfigs=1, verbose=verbose, **kwargs)[0])
        ]
        # print(value)
        if round_machine_zero and value.magnitude < default_machine_zero:
            round_machine_zero = False
    if append:
        hashes.extend(hash_db[key]['hashes'])
        first_sigfig_hashes.extend(hash_db[key]['first_sigfig_hashes'])
    hash_db[key] = dict(
        hashes=unique(hashes),
        first_sigfig_hashes=unique(first_sigfig_hashes),
        units=str(units.__repr__()),
        sigfigs=sigfigs,
    )
    # Save hashes to disk
    with open(filename, "w") as f:
        json.dump(hash_db, f, indent=4)


def read_solution_hash(key, filename=default_hash_filename):
    hashes = read_solution_hashes(filename)
    return hashes[key]
