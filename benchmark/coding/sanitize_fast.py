"""Exact pruning of EvalPlus 0.3.1 code_extract; official filtering is unchanged.

The original scans (i,j) lexicographically and only replaces its answer on a
strictly larger number of nonblank lines. Candidates that cannot improve that
number need no AST parse. An earliest valid window containing every remaining
nonblank line reaches the upper bound and terminates the scan exactly.
"""
from functools import lru_cache
from types import FunctionType
import ast
import codeop


def _ast_compile(source, filename, symbol, incomplete_input=True):
    flags = ast.PyCF_ONLY_AST
    if incomplete_input:
        flags |= codeop.PyCF_ALLOW_INCOMPLETE_INPUT | codeop.PyCF_DONT_IMPLY_DEDENT
    return compile(source, filename, symbol, flags, dont_inherit=True)


@lru_cache(maxsize=16384)
def _invalid_prefix(source):
    # AST-only is essential: bytecode compilation also rejects valid ASTs such
    # as misplaced future imports, break outside loops, or return outside defs.
    # Incomplete prefixes remain candidates; only definitive syntax errors skip.
    try:
        codeop._maybe_compile(_ast_compile, source, "<code-extract>", "exec")
    except SyntaxError:
        return True
    except Exception:
        return False
    return False


def code_extract(text):
    from evalplus.syncheck import syntax_check

    lines = text.split("\n")
    prefix = [0]
    last_nonblank = -1
    for index, line in enumerate(lines):
        prefix.append(prefix[-1] + bool(line.strip()))
        if line.strip():
            last_nonblank = index
    best_pair = (0, 0)
    best = 0
    for i in range(len(lines)):
        remaining = prefix[-1] - prefix[i]
        if remaining <= best:
            break
        if i + 1 < len(lines) and _invalid_prefix("\n".join(lines[i:i + 2])):
            continue
        # Earliest legal j attaining this i's upper bound (original j >= i+1).
        bound_j = max(i + 1, last_nonblank)
        if bound_j < len(lines):
            candidate = "\n".join(lines[i:bound_j + 1])
            if syntax_check(candidate):
                return candidate
        for j in range(i + 1, len(lines)):
            count = prefix[j + 1] - prefix[i]
            if count <= best:
                continue
            candidate = "\n".join(lines[i:j + 1])
            if syntax_check(candidate):
                best = count
                best_pair = (i, j)
    return "\n".join(lines[best_pair[0]:best_pair[1] + 1])


@lru_cache(maxsize=1)
def _extractor():
    from evalplus import sanitize as official

    # Clone only the function globals; never patch the installed dependency.
    namespace = {**vars(official), "code_extract": lambda text: text}
    return FunctionType(official.extract_target_code_or_empty.__code__, namespace)


def sanitize(code, entrypoint=None):
    extracted = code_extract(code)
    result = _extractor()(extracted, entrypoint).strip()
    return result if result else extracted
