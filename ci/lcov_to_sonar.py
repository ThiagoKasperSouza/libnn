#!/usr/bin/env python3
"""Converte um relatório lcov (.info) para SonarQube Generic Coverage XML.

Uso::

    python3 ci/lcov_to_sonar.py coverage.info coverage.xml [root]

O formato esperado por ``sonar.coverageReportPaths`` é:

    <coverage version="1">
      <file path="src/foo.c">
        <lineToCover lineNumber="10" covered="true"
                     branchesToCover="2" coveredBranches="1"/>
      </file>
    </coverage>

O argumento opcional ``root`` é removido do início de cada caminho SF para
que os paths no XML sejam relativos ao workspace, como o SonarQube espera.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote


def parse_lcov_clean(text: str, root: str | None):
    files: dict[str, dict[int, dict]] = {}
    branches_by_file: dict[str, dict[int, list[str]]] = {}
    current_path: str | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if line == "end_of_record":
            current_path = None
            continue
        if line.startswith("SF:"):
            path = unquote(line[3:])
            p = Path(path)
            if root:
                try:
                    path = str(p.relative_to(Path(root).resolve()))
                except ValueError:
                    path = str(p)
            current_path = path
            files.setdefault(path, {})
            branches_by_file.setdefault(path, {})
            continue
        if current_path is None:
            continue

        data = files[current_path]
        br = branches_by_file[current_path]

        if line.startswith("DA:"):
            parts = line[3:].split(",")
            try:
                no, count = int(parts[0]), int(parts[1])
            except (ValueError, IndexError):
                continue
            entry = data.setdefault(no, {"covered": False})
            entry["covered"] = count > 0
        elif line.startswith("BRDA:"):
            parts = line[5:].split(",")
            try:
                no = int(parts[0])
                taken = parts[3]
            except (ValueError, IndexError):
                continue
            br.setdefault(no, []).append(taken)

    for path, data in files.items():
        for no, taken_list in branches_by_file.get(path, {}).items():
            entry = data.setdefault(no, {"covered": False})
            entry["branches"] = len(taken_list)
            entry["covered_branches"] = sum(
                1 for t in taken_list if t != "-" and t.isdigit() and int(t) > 0
            )
    return files


def write_sonar_xml(files, output: Path) -> None:
    root = ET.Element("coverage", version="1")
    for path in sorted(files):
        f = ET.SubElement(root, "file", path=path)
        for no in sorted(files[path]):
            info = files[path][no]
            attrs = {
                "lineNumber": str(no),
                "covered": "true" if info.get("covered") else "false",
            }
            if "branches" in info:
                attrs["branchesToCover"] = str(info["branches"])
                attrs["coveredBranches"] = str(info["covered_branches"])
            ET.SubElement(f, "lineToCover", attrs)

    output.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    # ElementTree escreve um XML de uma linha; adiciona newline final legível.
    with output.open("a", encoding="utf-8") as fh:
        fh.write("\n")


def main(argv) -> int:
    if len(argv) < 3:
        print(f"Uso: {argv[0]} <coverage.info> <coverage.xml> [root]", file=sys.stderr)
        return 2
    info = Path(argv[1])
    output = Path(argv[2])
    root = argv[3] if len(argv) > 3 else None

    if not info.is_file():
        print(f"Arquivo não encontrado: {info}", file=sys.stderr)
        return 1

    text = info.read_text(encoding="utf-8", errors="replace")
    files = parse_lcov_clean(text, root)
    write_sonar_xml(files, output)

    lines = sum(len(v) for v in files.values())
    covered = sum(1 for v in files.values() for i in v.values() if i.get("covered"))
    pct = (100.0 * covered / lines) if lines else 0.0
    print(f"{output}: {len(files)} arquivos, {lines} linhas, {pct:.1f}% cobertas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
