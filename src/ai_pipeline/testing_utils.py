"""Utilities to annotate tests with human-readable metadata for reporting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence
import inspect


@dataclass(frozen=True)
class TargetDetails:
    """Lightweight description of a code object exercised by a test."""

    display_name: str
    module: Optional[str]
    file: Optional[str]
    doc: Optional[str]


@dataclass(frozen=True)
class TestMetadata:
    """Container for descriptive information about a test."""

    purpose: str
    notes: Optional[str]
    targets: Sequence[Any]


def describe_test(*, purpose: str, targets: Sequence[Any] | None = None, notes: str | None = None):
    """Attach human-friendly context to a pytest test function.

    Parameters
    ----------
    purpose:
        A concise sentence that explains what behaviour the test verifies.
    targets:
        Optional iterable of code objects (functions, methods, or classes) that the
        test focuses on. These are used to pull docstrings and source file paths for
        reporting purposes.
    notes:
        Additional free-form notes that should appear in the generated report.
    """

    if not purpose:
        raise ValueError("describe_test requires a non-empty purpose")

    resolved_targets: Sequence[Any] = list(targets or [])

    def _decorator(func):
        setattr(func, "__test_metadata__", TestMetadata(purpose=purpose, notes=notes, targets=resolved_targets))
        return func

    return _decorator


def resolve_target_details(targets: Iterable[Any]) -> List[TargetDetails]:
    """Convert raw target references into serialisable descriptions."""

    details: List[TargetDetails] = []
    for target in targets:
        if isinstance(target, TargetDetails):
            details.append(target)
            continue
        if isinstance(target, str):
            details.append(TargetDetails(display_name=target, module=None, file=None, doc=None))
            continue
        try:
            module = getattr(target, "__module__", None)
            qualname = getattr(target, "__qualname__", getattr(target, "__name__", repr(target)))
            source = inspect.getsourcefile(target)
            doc = inspect.getdoc(target)
            details.append(TargetDetails(display_name=qualname, module=module, file=source, doc=doc))
        except Exception:
            details.append(TargetDetails(display_name=repr(target), module=None, file=None, doc=None))
    return details


__all__ = ["describe_test", "resolve_target_details", "TargetDetails", "TestMetadata"]
