# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Count core and registry-implementation Python LOC inside ``xanesnet``.

Files with ``*.Registry.register`` calls are implementations. When such a file
lives below the nearest package containing ``registry.py``, its subpackage is
also counted so model-specific helpers are included. Everything else is core.
"""

import argparse
import ast
from pathlib import Path

DOCSTRING_NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def inspect_source(source_file: Path) -> tuple[int, set[str]]:
    """Count code lines and find registry registrations in a Python file.

    Args:
        source_file: Python source file to inspect.

    Returns:
        A tuple containing the nonblank code-line count, excluding comment-only
        lines and docstrings, and the registry names found in registration calls.
    """

    source = source_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_file))
    docstring_lines: set[int] = set()
    registry_names: set[str] = set()

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "register"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id.endswith("Registry")
        ):
            registry_names.add(node.func.value.id)

        if not isinstance(node, DOCSTRING_NODES) or not node.body:
            continue
        first_statement = node.body[0]
        if (
            not isinstance(first_statement, ast.Expr)
            or not isinstance(first_statement.value, ast.Constant)
            or not isinstance(first_statement.value.value, str)
        ):
            continue
        end_lineno = first_statement.end_lineno or first_statement.lineno
        docstring_lines.update(range(first_statement.lineno, end_lineno + 1))

    code_lines = sum(
        1
        for line_number, line in enumerate(source.splitlines(), start=1)
        if line.strip() and not line.lstrip().startswith("#") and line_number not in docstring_lines
    )
    return code_lines, registry_names


def find_registry_root(source_file: Path, package_root: Path) -> Path | None:
    """Find the nearest package ancestor that defines a ``registry.py``.

    Args:
        source_file: Python source file whose registry package should be found.
        package_root: Root directory that bounds the ancestor search.

    Returns:
        The nearest ancestor containing ``registry.py``, or ``None`` if no such ancestor exists.
    """

    relative_parts = source_file.relative_to(package_root).parts
    for depth in range(len(relative_parts) - 1, -1, -1):
        candidate = package_root.joinpath(*relative_parts[:depth])
        if (candidate / "registry.py").is_file():
            return candidate
    return None


def collect_module_groups(
    package_root: Path,
    registered_sources: dict[Path, set[str]],
    plotters_common_as_registry: bool = False,
) -> dict[str, set[Path]]:
    """Group registered sources with their nested implementation helpers.

    Args:
        package_root: Root directory of the package being measured.
        registered_sources: Registered source paths mapped to their registry class names.
        plotters_common_as_registry: Whether to count shared plotter helpers as
            plotter implementation code.

    Returns:
        Registry labels mapped to sets of registered source and implementation helper files.
    """

    module_registries = {path: set(names) for path, names in registered_sources.items()}
    implementation_roots: dict[Path, set[str]] = {}

    for source_file, registry_names in registered_sources.items():
        registry_root = find_registry_root(source_file, package_root)
        if registry_root is not None and source_file.parent != registry_root:
            implementation_roots.setdefault(source_file.parent, set()).update(registry_names)

    for implementation_root, registry_names in implementation_roots.items():
        for source_file in implementation_root.rglob("*.py"):
            if (
                source_file.parent == implementation_root
                and source_file.name in {"base.py", "registry.py"}
                and source_file not in registered_sources
            ):
                continue
            module_registries.setdefault(source_file, set()).update(registry_names)

    if plotters_common_as_registry:
        common_root = package_root / "analysis" / "plotters" / "common"
        if not common_root.is_dir():
            raise FileNotFoundError(f"Plotter common package does not exist: {common_root}")
        for source_file in common_root.rglob("*.py"):
            module_registries.setdefault(source_file, set()).add("PlotterRegistry")

    groups: dict[str, set[Path]] = {}
    for source_file, registry_names in module_registries.items():
        label = " + ".join(sorted(name.removesuffix("Registry") for name in registry_names))
        groups.setdefault(label, set()).add(source_file)
    return groups


def main() -> None:
    """Print core and per-registry Python LOC for the xanesnet package."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plotters-common",
        dest="plotters_common_as_registry",
        action="store_true",
        help="Count analysis/plotters/common as Plotter registry implementation code.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    package_root = repo_root / "xanesnet"
    if not package_root.is_dir():
        raise SystemExit(f"Could not find the xanesnet package at {package_root}")

    source_files = sorted(package_root.rglob("*.py"))
    source_info = {source_file: inspect_source(source_file) for source_file in source_files}
    registered_sources = {
        source_file: registry_names for source_file, (_, registry_names) in source_info.items() if registry_names
    }
    module_groups = collect_module_groups(
        package_root,
        registered_sources,
        plotters_common_as_registry=args.plotters_common_as_registry,
    )
    module_files = set().union(*module_groups.values()) if module_groups else set()
    source_file_set = set(source_files)
    if not module_files.issubset(source_file_set):
        raise ValueError("A registry implementation file is outside the xanesnet source tree.")
    core_files = source_file_set - module_files
    core_groups: dict[str, set[Path]] = {}
    for source_file in core_files:
        relative_parts = source_file.relative_to(package_root).parts
        group = relative_parts[0] if len(relative_parts) > 1 else "package root"
        core_groups.setdefault(group, set()).add(source_file)
    line_counts = {source_file: lines for source_file, (lines, _) in source_info.items()}

    core_lines = sum(line_counts[path] for path in core_files)
    core_group_lines = {group: sum(line_counts[path] for path in files) for group, files in core_groups.items()}
    if sum(core_group_lines.values()) != core_lines:
        raise ValueError("Core breakdown does not sum to the core LOC total.")
    module_lines = sum(line_counts[path] for path in module_files)
    module_group_lines = {group: sum(line_counts[path] for path in files) for group, files in module_groups.items()}
    if sum(module_group_lines.values()) != module_lines:
        raise ValueError("Registry implementation breakdown does not sum to its LOC total.")
    total_lines = sum(line_counts.values())

    print(f"Python source root: {package_root.relative_to(repo_root)}")
    print("LOC excludes blank lines, comment-only lines, and docstrings.")
    print("Implementation LOC includes registered sources and nested package helpers.")
    print()
    print(f"{'Core functionality':<34} {core_lines:>9,} LOC ({len(core_files):>3} files)")
    print(
        f"{'Registry implementations':<34} {module_lines:>9,} LOC "
        f"({len(module_files):>3} files; {len(registered_sources)} registered sources)"
    )
    print(f"{'Total':<34} {total_lines:>9,} LOC ({len(source_files):>3} files)")
    print()
    print("Core functionality breakdown:")
    for group, files in sorted(core_groups.items()):
        print(f"  {group:<32} {core_group_lines[group]:>8,} LOC ({len(files):>3} files)")
    print(f"  {'Breakdown total':<32} {core_lines:>8,} LOC ({len(core_files):>3} files)")
    print()
    print("Registry implementation breakdown:")
    for group, files in sorted(module_groups.items()):
        print(f"  {group:<32} {module_group_lines[group]:>8,} LOC ({len(files):>3} files)")
    print(f"  {'Breakdown total':<32} {module_lines:>8,} LOC ({len(module_files):>3} files)")


if __name__ == "__main__":
    main()
