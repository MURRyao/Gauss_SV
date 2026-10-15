"""Parser for vibrational spectra printed by Gaussian log files.

Only plain-text Gaussian output is handled here.  Binary checkpoint files are
intentionally outside this module's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DdEe][-+]?\d+)?")
FREQUENCIES = re.compile(r"^\s*Frequencies\s+--\s*(.+?)\s*$")
IR_INTENSITIES = re.compile(r"^\s*IR Inten\s+--\s*(.+?)\s*$")
RAMAN_ACTIVITIES = re.compile(r"^\s*Raman Activ\s+--\s*(.+?)\s*$")
DEPOLARIZATION_P = re.compile(r"^\s*Depolar \(P\)\s+--\s*(.+?)\s*$")
DEPOLARIZATION_U = re.compile(r"^\s*Depolar \(U\)\s+--\s*(.+?)\s*$")
TERMINATION = re.compile(r"^\s*(Normal|Error) termination of Gaussian\b", re.MULTILINE)


class GaussianImportError(ValueError):
    """The selected file cannot safely supply a completed vibrational spectrum."""


@dataclass(frozen=True)
class VibrationalMode:
    """One normal mode and the quantities used for IR/Raman spectra."""

    number: int
    frequency: float
    ir_intensity: float
    raman_activity: float
    depolarization_p: float | None = None
    depolarization_u: float | None = None


@dataclass(frozen=True)
class GaussianCalculation:
    """Validated import result from a completed Gaussian frequency calculation."""

    source: Path
    route: str | None
    title: str | None
    modes: tuple[VibrationalMode, ...]


def _numbers(text: str) -> list[float]:
    return [float(match.group(0).replace("D", "E").replace("d", "e")) for match in NUMBER.finditer(text)]


def _route_and_title(lines: list[str]) -> tuple[str | None, str | None]:
    """Return the first Gaussian route section and its title, when present."""

    for index, line in enumerate(lines):
        if not line.lstrip().startswith("#"):
            continue
        route_lines = [line.strip()]
        cursor = index + 1
        while cursor < len(lines) and lines[cursor].strip():
            route_lines.append(lines[cursor].strip())
            cursor += 1
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        title = lines[cursor].strip() if cursor < len(lines) and lines[cursor].strip() else None
        return " ".join(route_lines), title
    return None, None


def parse_gaussian_text(text: str, source: Path | str = Path("<memory>")) -> GaussianCalculation:
    """Validate Gaussian output and extract frequency, IR and Raman triplets.

    Gaussian prints up to three normal modes in one block.  The parser joins
    the ``Frequencies --``, ``IR Inten --`` and ``Raman Activ --`` records from
    each block.  An incomplete group is treated as an import error rather than
    silently producing an incorrectly aligned spectrum.
    """

    terminations = list(TERMINATION.finditer(text))
    if not terminations:
        raise GaussianImportError("В файле не найдено сообщение Normal termination of Gaussian.")
    if terminations[-1].group(1) != "Normal":
        raise GaussianImportError("Последний запуск Gaussian завершился с ошибкой (Error termination).")

    lines = text.splitlines()
    route, title = _route_and_title(lines)
    modes: list[VibrationalMode] = []
    frequencies: list[float] | None = None
    ir_intensities: list[float] | None = None
    raman_activities: list[float] | None = None
    depolarization_p: list[float] | None = None
    depolarization_u: list[float] | None = None

    def append_group() -> None:
        nonlocal frequencies, ir_intensities, raman_activities, depolarization_p, depolarization_u
        if frequencies is None:
            return
        if ir_intensities is None or raman_activities is None:
            raise GaussianImportError("Найдена неполная группа частот: отсутствуют ИК- или Raman-интенсивности.")
        if not (len(frequencies) == len(ir_intensities) == len(raman_activities)):
            raise GaussianImportError("Число частот и интенсивностей в группе Gaussian не совпадает.")
        if depolarization_p is not None and len(depolarization_p) != len(frequencies):
            raise GaussianImportError("Число коэффициентов Depolar (P) не совпадает с числом частот.")
        if depolarization_u is not None and len(depolarization_u) != len(frequencies):
            raise GaussianImportError("Число коэффициентов Depolar (U) не совпадает с числом частот.")
        for index, (frequency, ir_intensity, raman_activity) in enumerate(zip(frequencies, ir_intensities, raman_activities)):
            p_value = depolarization_p[index] if depolarization_p is not None else None
            u_value = depolarization_u[index] if depolarization_u is not None else None
            modes.append(VibrationalMode(len(modes) + 1, frequency, ir_intensity, raman_activity, p_value, u_value))
        frequencies = ir_intensities = raman_activities = depolarization_p = depolarization_u = None

    for line in lines:
        match = FREQUENCIES.match(line)
        if match:
            append_group()
            frequencies = _numbers(match.group(1))
            continue
        match = IR_INTENSITIES.match(line)
        if match and frequencies is not None:
            ir_intensities = _numbers(match.group(1))
            continue
        match = RAMAN_ACTIVITIES.match(line)
        if match and frequencies is not None:
            raman_activities = _numbers(match.group(1))
            continue
        match = DEPOLARIZATION_P.match(line)
        if match and frequencies is not None:
            depolarization_p = _numbers(match.group(1))
            continue
        match = DEPOLARIZATION_U.match(line)
        if match and frequencies is not None:
            depolarization_u = _numbers(match.group(1))

    append_group()
    if not modes:
        raise GaussianImportError("В завершённом расчёте не найдены частоты с ИК- и Raman-интенсивностями.")

    return GaussianCalculation(Path(source), route, title, tuple(modes))


def parse_gaussian_log(path: str | Path) -> GaussianCalculation:
    """Read and parse a Gaussian ``.log`` or ``.out`` file."""

    source = Path(path)
    try:
        text = source.read_bytes().decode("utf-8", errors="replace")
    except OSError as error:
        raise GaussianImportError(f"Не удалось прочитать файл: {error}") from error
    return parse_gaussian_text(text, source)
