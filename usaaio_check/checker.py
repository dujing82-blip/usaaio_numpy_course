"""Friendly self-study checks, not a secure exam sandbox."""
import ast
import inspect
import textwrap
import copy
import numpy as np

NO_LOOPS = dict(no_loops=True)
PURE = dict(no_loops=True, no_linalg=True, no_matmul=True, no_transpose=True)

def check_rules(fn, no_loops=False, no_linalg=False, no_matmul=False,
                no_transpose=False, no_imports=True, max_loops=None,
                no_numpy=False, no_kv_copies=False, final_dict_only=False,
                require_seed_reset=False):
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, IndentationError, SyntaxError):
        return ["source unavailable; rerun the definition cell"]
    problems = []
    nodes = list(ast.walk(tree))
    loops = [n for n in nodes if isinstance(n, (ast.For, ast.AsyncFor, ast.While,
              ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))]
    if no_loops and loops:
        problems.append("loop or comprehension")
    if max_loops is not None and len(loops) > max_loops:
        problems.append(f"more than {max_loops} loop(s)")
    if max_loops is not None and any(isinstance(n, (ast.ListComp, ast.SetComp,
                        ast.DictComp, ast.GeneratorExp)) for n in loops):
        problems.append("comprehension (only an explicit iteration loop is allowed)")
    if final_dict_only:
        fn_node = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
        last = fn_node.body[-1] if fn_node else None
        if len(loops)!=1 or not isinstance(loops[0], ast.DictComp) or not isinstance(last, ast.Return) or last.value is not loops[0]:
            problems.append("only one final dictionary comprehension is allowed")
    if require_seed_reset:
        resets = [n for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr=='seed' and not n.args and not n.keywords]
        if not resets:
            problems.append("missing np.random.seed() reset")
    for n in nodes:
        if no_imports and isinstance(n, (ast.Import, ast.ImportFrom)):
            problems.append("import inside the function")
        if no_numpy and isinstance(n, ast.Name) and n.id in ("np", "numpy"):
            problems.append("NumPy inside the function")
        if no_linalg and isinstance(n, ast.Attribute) and n.attr == "linalg":
            problems.append("np.linalg")
        if no_matmul and isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult):
            problems.append("the @ operator")
        if no_matmul and isinstance(n, ast.Attribute) and n.attr in (
                "dot", "matmul", "einsum", "inner", "tensordot", "vdot", "outer", "multi_dot", "mm", "bmm"):
            problems.append(f".{n.attr}")
        if no_transpose and isinstance(n, ast.Attribute) and n.attr in (
                "T", "transpose", "swapaxes", "mT", "moveaxis", "permute"):
            problems.append(f".{n.attr}")
        if no_kv_copies and isinstance(n, ast.Attribute) and n.attr in (
                "repeat", "tile", "repeat_interleave", "copy", "concatenate", "cat"):
            problems.append("copies of K/V")
    return sorted(set(problems))

def _equal(got, expected, atol=1e-6, rtol=1e-6):
    if expected is None or isinstance(expected, (str, bool)):
        assert type(got) is type(expected) and got == expected, "metadata differs"
    elif isinstance(expected, dict):
        assert isinstance(got, dict) and got.keys() == expected.keys(), "dictionary keys differ"
        for k in expected:
            _equal(got[k], expected[k], atol, rtol)
    elif isinstance(expected, (tuple, list)):
        assert type(got) is type(expected) and len(got) == len(expected), "container type/length differs"
        for g, e in zip(got, expected):
            _equal(g, e, atol, rtol)
    elif hasattr(expected, "detach"):
        assert hasattr(got, "detach"), "expected a PyTorch tensor"
        assert got.dtype == expected.dtype, f"dtype {got.dtype}, expected {expected.dtype}"
        _equal(got.detach().cpu().numpy(), expected.detach().cpu().numpy(), atol, rtol)
    elif hasattr(expected, "index") and hasattr(expected, "to_numpy"):
        assert type(got) is type(expected), "expected a pandas result"
        assert got.index.equals(expected.index), "pandas index differs"
        _equal(got.to_numpy(), expected.to_numpy(), atol, rtol)
    elif isinstance(expected, np.ndarray):
        assert isinstance(got, np.ndarray), "expected a NumPy array"
        assert got.shape == expected.shape, f"shape {got.shape}, expected {expected.shape}"
        if expected.dtype.kind in "iub":
            assert got.dtype == expected.dtype, f"dtype {got.dtype}, expected {expected.dtype}"
        if expected.dtype.kind in "OUS":
            assert np.array_equal(got, expected), "values differ"
        else:
            assert np.allclose(got, expected, atol=atol, rtol=rtol, equal_nan=False), "values differ"
    else:
        assert np.isscalar(got) and not isinstance(got, (str, bool)), "expected a numeric scalar"
        assert np.allclose(got, expected, atol=atol, rtol=rtol), "values differ"

def grade(fn, cases, rules=None, atol=1e-6, rtol=1e-6, immutable=False):
    bad = check_rules(fn, **(rules or {}))
    if bad:
        print(f"RULE VIOLATION in {fn.__name__}: {', '.join(bad)} -> 0 points on the real exam")
        return False
    ok = True
    for i, (args, expected) in enumerate(cases):
        try:
            before = copy.deepcopy(args)
            got = fn(*args)
            _equal(got, expected, atol, rtol)
            if immutable:
                _equal(args, before, 0, 0)
            print(f"  case {i+1}: PASS")
        except Exception as e:
            ok = False
            print(f"  case {i+1}: FAIL - {type(e).__name__}: {e}")
    print("ALL PASS" if ok else "not yet")
    return ok

def show_shape(**arrays):
    for k, v in arrays.items():
        print(f"{k:>12s}: shape {tuple(v.shape)}  ndim {v.ndim}  dtype {v.dtype}")
