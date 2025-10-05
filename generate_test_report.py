"""Generate a Markdown report with pytest results and rich explanations."""
from __future__ import annotations

import argparse
import datetime as dt
import inspect
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import pytest

from ai_pipeline.testing_utils import TestMetadata, resolve_target_details


class ReportingPlugin:
    """Pytest hook implementation that captures metadata and outcomes."""

    def __init__(self) -> None:
        self.collected: Dict[str, Dict[str, object]] = {}
        self.execution: Dict[str, Dict[str, object]] = {}

    # Collection phase -------------------------------------------------
    def pytest_collection_modifyitems(self, session, config, items):  # type: ignore[override]
        for item in items:
            metadata: Optional[TestMetadata] = getattr(item.obj, "__test_metadata__", None)
            doc = inspect.getdoc(item.obj)
            self.collected[item.nodeid] = {
                "nodeid": item.nodeid,
                "name": item.name,
                "path": str(item.fspath),
                "doc": doc,
                "metadata": metadata,
            }

    # Execution phase --------------------------------------------------
    def pytest_runtest_logreport(self, report):  # type: ignore[override]
        entry = self.execution.setdefault(
            report.nodeid,
            {"duration": 0.0, "details": None, "outcome": None, "stage": None, "xfail": False},
        )
        entry["duration"] = float(entry["duration"]) + float(report.duration or 0.0)
        entry["stage"] = report.when

        if getattr(report, "wasxfail", False):
            entry["xfail"] = True

        if report.skipped:
            entry["outcome"] = "skipped"
            try:
                # Skip reports often expose (filename, lineno, reason)
                entry["details"] = report.longrepr[2]  # type: ignore[index]
            except Exception:
                entry["details"] = str(report.longrepr)
            return

        if report.failed:
            entry["outcome"] = "failed"
            entry["details"] = str(report.longrepr)
            return

        if report.passed and report.when == "call":
            entry["outcome"] = "passed"
            entry["details"] = None

    # Utility ----------------------------------------------------------
    def iter_tests(self) -> Iterable[Dict[str, object]]:
        for nodeid, meta in self.collected.items():
            result = self.execution.get(nodeid, {})
            merged = {**meta, **result}
            yield merged


def canonical_outcome(entry: Dict[str, object]) -> str:
    outcome = entry.get("outcome")
    stage = entry.get("stage")
    if outcome is None:
        return "not-run"
    if outcome == "failed" and stage != "call":
        return f"error ({stage})"
    return str(outcome)


def format_duration(seconds: Optional[float]) -> str:
    if not seconds:
        return "0.00s"
    return f"{seconds:.2f}s"


def build_markdown(plugin: ReportingPlugin, exit_code: int) -> str:
    tests = list(plugin.iter_tests())
    tests.sort(key=lambda entry: (entry.get("path", ""), entry.get("name", "")))

    summary_counts: Dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0, "error": 0, "not-run": 0}
    total_duration = 0.0

    normalised_entries: List[Dict[str, object]] = []
    for entry in tests:
        outcome_label = canonical_outcome(entry)
        total_duration += float(entry.get("duration") or 0.0)
        if outcome_label.startswith("error"):
            summary_counts["error"] += 1
        elif outcome_label in summary_counts:
            summary_counts[outcome_label] += 1
        else:
            summary_counts[outcome_label] = summary_counts.get(outcome_label, 0) + 1
        normalised_entries.append({**entry, "outcome_label": outcome_label})

    total_tests = len(tests)
    now = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    overall_status = "SUCCESS" if exit_code == 0 else "FAILURE"

    lines: List[str] = []
    lines.append(f"# Test Report - {now}")
    lines.append("")
    lines.append(f"- Overall status: {overall_status}")
    lines.append(f"- Total tests: {total_tests}")
    lines.append(
        "- Outcomes: "
        f"passed={summary_counts.get('passed', 0)}, "
        f"failed={summary_counts.get('failed', 0)}, "
        f"error={summary_counts.get('error', 0)}, "
        f"skipped={summary_counts.get('skipped', 0)}"
    )
    lines.append(f"- Aggregate duration: {format_duration(total_duration)}")
    lines.append("")

    for entry in normalised_entries:
        nodeid = entry["nodeid"]
        lines.append(f"## {nodeid}")
        lines.append("")
        outcome_label = entry["outcome_label"]
        duration = format_duration(entry.get("duration"))
        lines.append(f"- Outcome: {outcome_label.upper()} ({duration})")

        test_doc = entry.get("doc")
        if test_doc:
            lines.append(f"- Test intent: {test_doc}")

        metadata: Optional[TestMetadata] = entry.get("metadata")  # type: ignore[assignment]
        if metadata:
            lines.append(f"- Purpose: {metadata.purpose}")
            if metadata.notes:
                lines.append(f"- Notes: {metadata.notes}")
            targets = resolve_target_details(metadata.targets)
            if targets:
                lines.append("- Code under test:")
                for target in targets:
                    descriptor = target.display_name
                    location = target.file or "(source unknown)"
                    module = target.module or "(module unknown)"
                    doc = target.doc or "No docstring available."
                    lines.append(f"  - `{descriptor}` | module={module} | source={location}")
                    lines.append(f"    - {doc}")
        details = entry.get("details")
        if details:
            lines.append(f"- Details: {details}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(content: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pytest and emit a Markdown report")
    parser.add_argument("--output", default="reports/test_report.md", help="Relative path for the generated report")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Additional arguments forwarded to pytest")
    args = parser.parse_args()

    plugin = ReportingPlugin()
    pytest_args = ["tests"]
    pytest_args.extend(args.pytest_args)

    exit_code = pytest.main(pytest_args, plugins=[plugin])
    report = build_markdown(plugin, exit_code)
    write_report(report, Path(args.output))
    print(f"Test report written to {args.output}")
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
