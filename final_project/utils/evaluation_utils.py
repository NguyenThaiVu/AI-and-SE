import os 
import numpy as np
import io
import re
import ast
import contextlib
import types
import unittest
import importlib
import sys

def _find_single_function_name(code: str) -> str:
    """Find the only top-level function name in 'code'."""
    tree = ast.parse(code)
    fns = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(fns) != 1:
        raise ValueError(f"Expected exactly one top-level function, found {len(fns)}: {fns}")
    return fns[0]


def run_tests(function_code: str,
              tests_code: str,
              *,
              libs=None,
              expected_signature=None):
    """
    Run unittest string against dynamically generated function with optional extra libs and aliases.
    """
    # ---------- 1. Prepare isolated module for candidate ----------
    candidate = types.ModuleType("candidate")
    candidate.__dict__['np'] = np  # pre-import numpy for convenience

    libs = libs or []
    loaded_libs = {}

    # Import libraries and assign common aliases
    alias_map = {
        'numpy': 'np',
        'pandas': 'pd',
        'matplotlib.pyplot': 'plt',
        'matplotlib': 'mpl',
        'os': 'os',
        're': 're'
    }

    for lib_name in libs:
        try:
            mod = importlib.import_module(lib_name)
            loaded_libs[lib_name] = mod
            candidate.__dict__[lib_name.split('.')[-1]] = mod  # e.g. 'os', 're'
            # assign alias if known
            if lib_name in alias_map:
                candidate.__dict__[alias_map[lib_name]] = mod
        except ModuleNotFoundError:
            print(f"⚠️ Warning: Library '{lib_name}' not installed, skipping import.")

    # ---------- 2. Execute generated function code ----------
    exec(function_code, candidate.__dict__)

    # ---------- 3. Extract function ----------
    gen_name = _find_single_function_name(function_code)
    gen_func = candidate.__dict__[gen_name]

    # ---------- 4. Prepare test module ----------
    tests_mod = types.ModuleType("tests_mod")
    tests_mod.__dict__['unittest'] = unittest
    tests_mod.__dict__['np'] = np

    # inject all libs + aliases into test globals
    for lib_name, mod in loaded_libs.items():
        tests_mod.__dict__[lib_name.split('.')[-1]] = mod
        if lib_name in alias_map:
            tests_mod.__dict__[alias_map[lib_name]] = mod

    # ---------- 5. Bind the candidate function ----------
    if expected_signature is None:
        expected_signature = ("task_func", ["mean", "std_dev", "n"])
    task_func_name, _ = expected_signature
    tests_mod.__dict__[task_func_name] = gen_func

    # ---------- 6. Execute test code ----------
    exec(tests_code, tests_mod.__dict__)

    # ---------- 7. Run unittest ----------
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests_mod)
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)

    return {
        "testsRun": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "wasSuccessful": result.wasSuccessful(),
        "output": stream.getvalue()
    }