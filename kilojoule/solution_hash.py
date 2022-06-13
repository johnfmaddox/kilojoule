"""kiloJoule solution_hash module

This module provides classes for storing or checking hashes of values
stored in variables.  The instructor can use these tools to create a
solution hash file that can be distributed to students.  The student
can then check their solutions against the stored hashes to verify
accuracy.
"""

from .common import get_caller_namespace
from .units import units, Quantity
from .display import numeric_to_string, to_latex, to_numeric, Latex

import hashlib
import json
from os.path import exists
import sys

import shutil
import os

from sigfig import round
import warnings
import re

default_hash_filename = '.solution_hashes'
default_student_dir = 'student/'
default_sigfigs = 4

def name_and_date(Name):
    if Name == 'Jane Doe': raise ValueError('Update the Name variable above')
    from IPython.display import display, Markdown
    from datetime import datetime
    import pytz
    today = datetime.now(pytz.timezone('US/Central'))
    display(Markdown(Name))
    display(Markdown(today.strftime('%B %d, %Y (%-I:%M %p CDT)')))


class QuietError(Exception):
    """Base class for other exceptions"""
    pass

class IncorrectValueError(QuietError):
    """Raised when solution hash doesn't match stored hash"""
    pass

def quiet_hook(kind, message, traceback):
    if QuietError in kind.__bases__:
        print('{0}: {1}'.format(kind.__name__, message))  # Only print Error Type and Message
    else:
        sys.__excepthook__(kind, message, traceback)  # Print Error Type, Message and Traceback

sys.excepthook = quiet_hook


def hashq(obj, units=None, sigfigs=None, verbose=False):
    if isinstance(obj, Quantity):
        base = obj.to_base_units()
        base_mag = base.magnitude
        base_units = base.units
    else:
        base = base_mag = obj
        base_units = units
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            str_rep = str(round(base_mag,sigfigs=sigfigs))
    except:
        str_rep = str(base_mag)
    encoded_str = str_rep.encode()
    hash_obj = hashlib.md5(encoded_str)
    hexa_value = hash_obj.hexdigest()
    if verbose:
        if isinstance(obj,Quantity):
            print(f'value:{obj:~} ->  base units:{base:~} -> magnitude:{base_mag} -> rounded:{str_rep}-> hash:{hexa_value}')
        else:
            print(f'value:{obj} ->  base units:{base} -> magnitude:{base_mag} -> rounded:{str_rep}-> hash:{hexa_value}')
    if units is not None: str_rep += f' {base_units}'
    hash_dict = dict(hash=hexa_value, units=units, sigfigs=sigfigs)
    return hexa_value, str_rep, hash_dict


def check_solutions(sol_list, n_col=3, namespace=None, **kwargs):
    """Accepts a list of solution check specifications and call `check_solution()` for each.

    Accepts a list of strings or a list of dictionaries.
    """
    namespace = namespace or get_caller_namespace()
    kwargs['namespace'] = namespace
    n = 1
    result_str = r'\begin{align} '
    for sol in sol_list:
        if isinstance(sol,str):
            result_str += check_solution(sol,single_check=False,**kwargs)
        elif isinstance(sol,dict):
            result_str += check_solution(**sol,single_check=False,**kwargs)
        if n < n_col:
            result_str += r' \quad & '
            n += 1
        else:
            result_str += r' \\ '
            n = 1
    # use regex to remove empty line from end of align environment if it exists
    result_str += r' \end{align}'
    result_str = re.sub(r'\\\\\s*{\s*}\s*\\end{align}',r'\n\\end{align}',result_str)
    display(Latex(result_str))

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
    prefix='',
    filename=default_hash_filename,
    verbose=False,
    raise_error=False,
    single_check=True,
    **kwargs
):
    namespace = namespace or get_caller_namespace()
    key = prefix+name

    # If no value was provided, evaluate the variable name in the namespace
    try:
        value = value or eval(name, namespace)
    except (NameError, KeyError):
        # NameError if undefined variable
        # KeyError if undefined index in a dict
        value = '??'
    if verbose: print(f'{key=}; {value=}')
    try:
        result_str_body = f'{to_latex(name)} &= {numeric_to_string(value)} && '
    except:
        result_str_body = f'{to_latex(name)} &= {value} && '
    # Read the corresponding entry form the hash db
    try:
        hash_db = read_solution_hash(key)
        # Set units and sigfigs to correspond to the hash db unless specified in the arguments
        units = units or hash_db['units']
        if units=="None": units = None
        sigfigs = sigfigs or hash_db['sigfigs']
        target_hashes = hash_db['hashes']
        #target_hashes = [str(i['hash']) for i in read_solution_hash(key)]
        try:
            hash_value, str_rep, hash_dict = hashq(value, units=units, sigfigs=sigfigs, verbose=verbose, **kwargs)
            if verbose: print(f'hash: {hash_value} <-> target: {target_hashes}')
            assert hash_value in target_hashes
            try:
                import emoji
                result_str_body += '✅'
            except:
                result_str_body += '\mathrm{Correct}'
        except AssertionError as err:
            try:
                result_str_body += '❌'
            except:
                result_str_body += '\mathrm{Incorrect}'
            msg=f'Hash Mismatch for {key}: {hash_value} not in {target_hashes}'
            if raise_error:
                raise IncorrectValueError(msg)
    except KeyError as err:
        if verbose: print(f'{name} not in hash database')
        if raise_error:
            raise err
    if single_check:
        result_str = f'\\begin{{align}}{result_str_body}\\end{{align}}'
        display(Latex(result_str))
    else:
        return result_str_body#+r'\\'


def read_solution_hashes(filename=default_hash_filename):
    if exists(filename):
    # Load existing hashes if the file exits
        with open(filename, 'r') as f:
            hashes = json.load(f)
    else:
    # Create an empty dict if no previous file exists
        hashes = {}
    return hashes


def store_solutions(sol_list=None, namespace=None, filename=default_hash_filename, copy_to_student=True, student_dir=default_student_dir, **kwargs):
    """Accepts a list of solution storage specifications and calls `store_solution()` for each.

    Accepts a list of strings or a list of dictionaries.
    """
    namespace = namespace or get_caller_namespace()
    kwargs['namespace'] = namespace
    for sol in sol_list:
        if isinstance(sol,str):
            store_solution(sol,**kwargs)
        elif isinstance(sol,dict):
            store_solution(**sol,**kwargs)
    if copy_to_student:
        os.makedirs(student_dir, exist_ok=True)
        shutil.copy2(default_hash_filename, student_dir)

def store_solution(
    name,
    value=None,
    units=None,
    sigfigs=default_sigfigs,
    namespace=None,
    prefix='',
    filename=default_hash_filename,
    verbose=False,
    **kwargs
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
    if isinstance(value,Quantity):
        units = value.units
    key = prefix+name
    if verbose: print(f'{key=}; {value=}')
    # Read in existing hash database
    hash_db = read_solution_hashes(filename)
    if isinstance(value, list):
        hashes = [str(hashq(i, units, sigfigs, verbose=verbose, **kwargs)[0]) for i in value]
    else:
        hashes = [str(hashq(value,units,sigfigs, verbose=verbose, **kwargs)[0])]
    hash_db[key] = dict(hashes=hashes, units=str(units.__repr__()), sigfigs=sigfigs)
    # Save hashes to disk
    with open(filename,'w') as f:
        json.dump(hash_db,f,indent=4)

def read_solution_hash(key, filename=default_hash_filename):
    hashes = read_solution_hashes(filename)
    return hashes[key]