from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"main": MAIN_NS}


@dataclass
class RegressionData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _load_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for si in root.findall("main:si", NS):
        parts = [text.text or "" for text in si.iter(f"{{{MAIN_NS}}}t")]
        values.append("".join(parts))
    return values


def _sheet_path_map(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
    rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: f"xl/{rel.attrib['Target']}"
        for rel in rel_root
        if rel.tag.endswith("Relationship")
    }

    sheet_paths: dict[str, str] = {}
    for sheet in workbook_root.find("main:sheets", NS):
        rel_id = sheet.attrib[f"{{{REL_NS}}}id"]
        sheet_paths[sheet.attrib["name"]] = rel_map[rel_id]
    return sheet_paths


def _parse_sheet_rows(
    archive: zipfile.ZipFile,
    sheet_path: str,
    shared_strings: list[str],
) -> list[tuple[float, float]]:
    root = ET.fromstring(archive.read(sheet_path))
    rows = root.findall(".//main:sheetData/main:row", NS)
    parsed: list[tuple[float, float]] = []

    for row in rows[1:]:
        values: list[float] = []
        for cell in row.findall("main:c", NS):
            value_node = cell.find("main:v", NS)
            if value_node is None:
                continue
            raw_value = value_node.text or ""
            if cell.attrib.get("t") == "s":
                raw_value = shared_strings[int(raw_value)]
            values.append(float(raw_value))
        if len(values) >= 2:
            parsed.append((values[0], values[1]))
    return parsed


def _to_arrays(rows: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    xs = np.array([row[0] for row in rows], dtype=np.float64)
    ys = np.array([row[1] for row in rows], dtype=np.float64)
    return xs, ys


def load_regression_data(path: Path) -> RegressionData:
    if not path.exists():
        raise FileNotFoundError(f"Excel data file not found: {path}")

    with zipfile.ZipFile(path) as archive:
        shared_strings = _load_shared_strings(archive)
        sheet_paths = _sheet_path_map(archive)
        required_sheets = ("Training Data", "Test Data")
        missing = [name for name in required_sheets if name not in sheet_paths]
        if missing:
            missing_names = ", ".join(missing)
            raise ValueError(f"Missing required worksheets: {missing_names}")
        train_rows = _parse_sheet_rows(
            archive,
            sheet_paths["Training Data"],
            shared_strings,
        )
        test_rows = _parse_sheet_rows(
            archive,
            sheet_paths["Test Data"],
            shared_strings,
        )

    x_train, y_train = _to_arrays(train_rows)
    x_test, y_test = _to_arrays(test_rows)
    return RegressionData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
