"""Import two-column experimental spectra from text files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


class ExperimentalImportError(ValueError):
    """The selected experimental data file is not a usable XY spectrum."""


@dataclass(frozen=True)
class ExperimentalSpectrum:
    """An experimental spectrum using wavenumber and measured intensity."""

    source: Path
    x: np.ndarray
    y: np.ndarray


def _as_float(token: str) -> float:
    return float(token.strip().replace(",", ".").replace("D", "E").replace("d", "e"))


def _tokens(line: str) -> list[str]:
    if ";" in line:
        return [part.strip() for part in line.split(";") if part.strip()]
    if "\t" in line:
        return [part.strip() for part in line.split("\t") if part.strip()]
    whitespace = line.split()
    if len(whitespace) >= 2:
        return whitespace
    return [part.strip() for part in line.split(",") if part.strip()]


def parse_experimental_text(text: str, source: Path | str = Path("<memory>")) -> ExperimentalSpectrum:
    """Parse numeric two-column data separated by whitespace, tabs, comma or semicolon."""

    points: list[tuple[float, float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "//", ";")):
            continue
        tokens = _tokens(line)
        if len(tokens) < 2:
            continue
        try:
            points.append((_as_float(tokens[0]), _as_float(tokens[1])))
        except ValueError:
            # Header and annotation lines are expected in laboratory exports.
            continue

    if len(points) < 3:
        raise ExperimentalImportError("Нужно не менее трёх числовых точек X/Y.")
    data = np.asarray(points, dtype=float)
    if not np.isfinite(data).all():
        raise ExperimentalImportError("Файл содержит нечисловые или бесконечные значения.")
    order = np.argsort(data[:, 0], kind="stable")
    return ExperimentalSpectrum(Path(source), data[order, 0], data[order, 1])


def parse_experimental_file(path: str | Path) -> ExperimentalSpectrum:
    """Read an experimental ``.csv``, ``.txt`` or ``.xy`` spectrum."""

    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise ExperimentalImportError(f"Не удалось прочитать файл: {error}") from error
    for encoding in ("utf-8-sig", "cp1251", "latin-1"):
        try:
            return parse_experimental_text(raw.decode(encoding), source)
        except UnicodeDecodeError:
            continue
    raise ExperimentalImportError("Не удалось определить текстовую кодировку файла.")
