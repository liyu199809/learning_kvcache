"""
Method Registry - Register and retrieve different memory methods
"""

import importlib
import inspect
from typing import Dict, Type, Any, Optional

from src.method.base_method import BaseMethod


# Lazy registry: each entry is (module_path, class_name). The module is imported
# only when get_method() / get_method_class() is called for that name. This keeps
# optional heavy dependencies (rank_bm25, ray, transformers, embedding stacks)
# off the import path for users who only need a subset of methods (e.g.
# longcontext + OpenAI API).
_LAZY_REGISTRY: Dict[str, tuple[str, str]] = {
    "bm25": ("src.method.bm25", "BM25Method"),
    "embedding": ("src.method.embedding_mem", "EmbeddingMethod"),
    "longcontext": ("src.method.longcontext", "LongContextMethod"),
    "ama_agent": ("src.method.ama_agent", "AMAAgentMethod"),
    "codex": ("src.method.agent_method", "CodexAgentMethod"),
    "claude_code": ("src.method.agent_method", "ClaudeCodeAgentMethod"),
}

_METHOD_REGISTRY: Dict[str, Type[BaseMethod]] = {}


def _resolve(name: str) -> Type[BaseMethod]:
    """Import + cache the method class for *name*."""
    if name in _METHOD_REGISTRY:
        return _METHOD_REGISTRY[name]
    if name not in _LAZY_REGISTRY:
        available = ", ".join(_LAZY_REGISTRY)
        raise ValueError(f"Method '{name}' not found. Available methods: {available}")
    module_path, class_name = _LAZY_REGISTRY[name]
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    _METHOD_REGISTRY[name] = cls
    return cls


def register_method(name: str, method_class: Type[BaseMethod]) -> None:
    """
    Register a new method.

    Args:
        name: Name of the method
        method_class: Class implementing BaseMethod interface
    """
    if not issubclass(method_class, BaseMethod):
        raise ValueError(f"Method class must inherit from BaseMethod, got {method_class}")

    _METHOD_REGISTRY[name] = method_class
    print(f"✅ Registered method: {name}")


def get_method(name: str, **kwargs) -> BaseMethod:
    """
    Get a method instance by name.

    Args:
        name: Name of the method
        **kwargs: Additional arguments to pass to the method constructor

    Returns:
        Instance of the requested method

    Raises:
        ValueError: If method name is not registered
    """
    method_class = _resolve(name)

    # Filter kwargs based on method's __init__ signature
    init_params = inspect.signature(method_class.__init__).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in init_params}

    return method_class(**filtered_kwargs)


def list_methods() -> list:
    """
    List all registered methods.

    Returns:
        List of method names
    """
    return list(_LAZY_REGISTRY.keys())


def get_method_class(name: str) -> Type[BaseMethod]:
    """
    Get the method class by name without instantiating.

    Args:
        name: Name of the method

    Returns:
        Method class

    Raises:
        ValueError: If method name is not registered
    """
    return _resolve(name)
