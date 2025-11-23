"""
This is the utility file for evaluating models on the HumanEval dataset.
"""

import os 
import builtins
import typing
import ast

def create_namespace():
    ns = {}

    # 1. Standard builtins (print, len, etc.)
    ns.update({k: getattr(builtins, k) for k in dir(builtins)})

    # 2. Install common typing names (List, Optional, etc.)
    for name in typing.__all__:
        ns[name] = getattr(typing, name)

    # 3. (Optional) Add math, random, itertools, etc.
    import math, random, itertools, statistics
    ns.update({
        'math': math,
        'random': random,
        'itertools': itertools,
        'statistics': statistics,
    })

    return ns


def evaluate_asserts(generated_code: str, test_code: str, entry_point: str):
    # ns = {}
    ns = create_namespace()
    
    # 1. Exec both code strings
    exec(generated_code, ns)
    exec(test_code, ns)

    candidate = ns[entry_point]     # the model's function
    check_fn = ns["check"]          # original check() function
    
    # 2. Parse the test code AST
    tree = ast.parse(test_code)

    # 3. Find the check() function body
    check_body = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            check_body = node.body
            break

    if check_body is None:
        raise ValueError("check() function not found.")
    
    # 4. Evaluate each assert individually
    results = []
    for idx, stmt in enumerate(check_body):
        if isinstance(stmt, ast.Assert):
            # Convert AST back to executable code
            code = compile(ast.Module([stmt], type_ignores=[]), "<assert>", "exec")
            try:
                exec(code, {**ns, "candidate": candidate})
                results.append(("pass", None))
            except Exception as e:
                results.append(("fail", repr(e)))

    # 5. Compute pass percentage
    total = len(results)
    passed = sum(1 for r, _ in results if r == "pass")
    percentage = passed / total if total > 0 else 0.0

    return {
        "total_asserts": total,
        "passed": passed,
        "percentage": percentage,
        "detail": results
    }