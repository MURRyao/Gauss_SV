"""Read and write self-contained Gaussian Spectra project files."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from experimental_parser import ExperimentalSpectrum
from gaussian_parser import GaussianCalculation, VibrationalMode


PROJECT_FORMAT = "gaussian-spectra-project"
PROJECT_VERSION = 1


class ProjectFormatError(ValueError):
    """A project file cannot be read safely."""


@dataclass(frozen=True)
class ProjectState:
    """Data and display settings restored from a project file."""

    calculation: GaussianCalculation | None
    experimental: ExperimentalSpectrum | None
    settings: dict[str, Any]


def _mode_to_dict(mode: VibrationalMode) -> dict[str, float | int | None]:
    return {
        "number": mode.number,
        "frequency": mode.frequency,
        "ir_intensity": mode.ir_intensity,
        "raman_activity": mode.raman_activity,
        "depolarization_p": mode.depolarization_p,
        "depolarization_u": mode.depolarization_u,
    }


def _mode_from_dict(value: object) -> VibrationalMode:
    if not isinstance(value, dict):
        raise ProjectFormatError("Некорректная запись колебательной моды в проекте.")
    try:
        return VibrationalMode(
            int(value["number"]),
            float(value["frequency"]),
            float(value["ir_intensity"]),
            float(value["raman_activity"]),
            None if value.get("depolarization_p") is None else float(value["depolarization_p"]),
            None if value.get("depolarization_u") is None else float(value["depolarization_u"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectFormatError("Некорректные значения колебательной моды в проекте.") from error


def _calculation_to_dict(calculation: GaussianCalculation | None) -> dict[str, object] | None:
    if calculation is None:
        return None
    return {
        "source": str(calculation.source),
        "route": calculation.route,
        "title": calculation.title,
        "modes": [_mode_to_dict(mode) for mode in calculation.modes],
    }


def _calculation_from_dict(value: object) -> GaussianCalculation | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not isinstance(value.get("modes"), list):
        raise ProjectFormatError("В проекте отсутствуют корректные данные Gaussian.")
    modes = tuple(_mode_from_dict(mode) for mode in value["modes"])
    if not modes:
        raise ProjectFormatError("Проект Gaussian не содержит колебательных мод.")
    source = value.get("source", "<project>")
    route, title = value.get("route"), value.get("title")
    if not isinstance(source, str) or route is not None and not isinstance(route, str) or title is not None and not isinstance(title, str):
        raise ProjectFormatError("Некорректные метаданные Gaussian в проекте.")
    return GaussianCalculation(Path(source), route, title, modes)


def _experimental_to_dict(spectrum: ExperimentalSpectrum | None) -> dict[str, object] | None:
    if spectrum is None:
        return None
    return {"source": str(spectrum.source), "x": spectrum.x.tolist(), "y": spectrum.y.tolist()}


def _experimental_from_dict(value: object) -> ExperimentalSpectrum | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ProjectFormatError("Некорректные экспериментальные данные в проекте.")
    try:
        source = value.get("source", "<project>")
        x, y = np.asarray(value["x"], dtype=float), np.asarray(value["y"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ProjectFormatError("Некорректные точки экспериментального спектра в проекте.") from error
    if not isinstance(source, str) or x.ndim != 1 or y.ndim != 1 or len(x) < 3 or len(x) != len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ProjectFormatError("Экспериментальный спектр в проекте повреждён.")
    return ExperimentalSpectrum(Path(source), x, y)


def save_project(
    path: str | Path,
    calculation: GaussianCalculation | None,
    experimental: ExperimentalSpectrum | None,
    settings: dict[str, Any],
) -> None:
    """Save imported data and display settings without depending on source files."""

    if calculation is None and experimental is None:
        raise ProjectFormatError("Нечего сохранять: импортируйте Gaussian- или экспериментальный спектр.")
    data = {
        "format": PROJECT_FORMAT,
        "version": PROJECT_VERSION,
        "calculation": _calculation_to_dict(calculation),
        "experimental": _experimental_to_dict(experimental),
        "settings": settings,
    }
    target = Path(path)
    try:
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as error:
        raise ProjectFormatError(f"Не удалось сохранить проект: {error}") from error


def load_project(path: str | Path) -> ProjectState:
    """Load a versioned, self-contained project file."""

    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProjectFormatError(f"Не удалось прочитать проект: {error}") from error
    if not isinstance(raw, dict) or raw.get("format") != PROJECT_FORMAT or raw.get("version") != PROJECT_VERSION:
        raise ProjectFormatError("Это не поддерживаемый файл проекта Gaussian Spectra.")
    settings = raw.get("settings")
    if not isinstance(settings, dict):
        raise ProjectFormatError("В проекте отсутствуют настройки отображения.")
    calculation = _calculation_from_dict(raw.get("calculation"))
    experimental = _experimental_from_dict(raw.get("experimental"))
    if calculation is None and experimental is None:
        raise ProjectFormatError("Проект не содержит спектральных данных.")
    return ProjectState(calculation, experimental, settings)
