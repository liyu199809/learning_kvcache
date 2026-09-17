"""Small JSON repair compatibility layer with no mandatory dependency."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

try:
    import json_repair as _json_repair
except ModuleNotFoundError:  # The B200 base image intentionally stays immutable.
    _json_repair = None


def loads(text: str) -> Any:
    """Parse JSON and repair common LLM formatting mistakes when possible."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as original_error:
        if _json_repair is not None:
            return _json_repair.loads(text)

        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
            candidate = re.sub(r"\s*```$", "", candidate)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        candidate = re.sub(
            r'([{,]\s*)([A-Za-z_][A-Za-z0-9_.-]*)(\s*:)',
            r'\1"\2"\3',
            candidate,
        )

        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass

        # Truncated object/array output is common near generation limits.
        balanced = candidate + "}" * max(0, candidate.count("{") - candidate.count("}"))
        balanced += "]" * max(0, candidate.count("[") - candidate.count("]"))
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            raise original_error


__all__ = ["loads"]
