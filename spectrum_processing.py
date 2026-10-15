"""Numerical operations for theoretical and experimental spectra."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from gaussian_parser import VibrationalMode


Profile = Literal["gaussian", "lorentzian", "voigt"]
Normalization = Literal["none", "unit", "percent"]
Polarization = Literal["unpolarized", "parallel", "perpendicular"]
BaselineMethod = Literal["als", "airpls", "linear", "polynomial"]


@dataclass(frozen=True)
class TheoreticalSettings:
    profile: Profile = "gaussian"
    width: float = 18.0
    frequency_scale: float = 1.0
    normalization: Normalization = "unit"
    laser_nm: float = 514.5
    temperature_k: float = 300.0
    polarization: Polarization = "unpolarized"


@dataclass(frozen=True)
class BaselineResult:
    method: BaselineMethod
    baseline: np.ndarray
    corrected: np.ndarray


def normalize_intensity(values: np.ndarray, mode: Normalization) -> np.ndarray:
    """Return values in the requested display normalization."""

    values = np.asarray(values, dtype=float)
    if mode == "none":
        return values.copy()
    maximum = float(np.max(np.abs(values))) if values.size else 0.0
    if maximum == 0.0:
        return values.copy()
    return values / maximum * (100.0 if mode == "percent" else 1.0)


def _raman_weight(mode: VibrationalMode, settings: TheoreticalSettings) -> float:
    """Calculate a relative Stokes Raman display weight.

    The polarization modes distribute the activity using Gaussian's
    depolarization ratio. They are suitable for comparative display, not for
    absolute cross-section reporting.
    """

    frequency = mode.frequency * settings.frequency_scale
    laser_wavenumber = 1e7 / settings.laser_nm
    shifted = laser_wavenumber - frequency
    if shifted <= 0 or frequency <= 0:
        return 0.0
    thermal_factor = 1.0 / (1.0 - np.exp(-1.438776877 * frequency / settings.temperature_k))
    activity = mode.raman_activity
    rho = mode.depolarization_p if mode.depolarization_p is not None else 0.75
    if settings.polarization == "parallel":
        activity /= 1.0 + rho
    elif settings.polarization == "perpendicular":
        activity *= rho / (1.0 + rho)
    return activity * shifted**4 * thermal_factor / frequency


def calculate_theoretical_spectrum(
    modes: Sequence[VibrationalMode],
    x_min: float,
    x_max: float,
    settings: TheoreticalSettings,
    kind: Literal["ir", "raman"],
    points: int = 2400,
) -> tuple[np.ndarray, np.ndarray]:
    """Broaden Gaussian normal modes over a wavenumber grid."""

    if x_max <= x_min:
        raise ValueError("Верхняя граница диапазона должна быть больше нижней.")
    if settings.width <= 0:
        raise ValueError("Ширина пика должна быть положительной.")
    x = np.linspace(x_min, x_max, max(points, 64))
    y = np.zeros_like(x)
    sigma = settings.width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = settings.width / 2.0
    for mode in modes:
        center = mode.frequency * settings.frequency_scale
        intensity = mode.ir_intensity if kind == "ir" else _raman_weight(mode, settings)
        delta = x - center
        gaussian = np.exp(-0.5 * (delta / sigma) ** 2)
        lorentzian = gamma**2 / (delta**2 + gamma**2)
        if settings.profile == "lorentzian":
            profile = lorentzian
        elif settings.profile == "voigt":
            profile = 0.5 * gaussian + 0.5 * lorentzian
        else:
            profile = gaussian
        y += intensity * profile
    return x, normalize_intensity(y, settings.normalization)


def _difference_matrix(length: int) -> sparse.csc_matrix:
    if length < 3:
        raise ValueError("Для baseline-коррекции нужно не менее трёх точек.")
    return sparse.diags((np.ones(length - 2), -2 * np.ones(length - 2), np.ones(length - 2)), (0, 1, 2), shape=(length - 2, length), format="csc")


def als_baseline(y: np.ndarray, smoothing: float = 1e6, asymmetry: float = 0.01, iterations: int = 12) -> np.ndarray:
    """Asymmetric least squares baseline (Eilers–Boelens)."""

    values = np.asarray(y, dtype=float)
    if smoothing <= 0 or not 0 < asymmetry < 1 or iterations < 1:
        raise ValueError("Недопустимые параметры ALS.")
    difference = _difference_matrix(len(values))
    penalty = smoothing * (difference.T @ difference)
    weights = np.ones(len(values))
    baseline = values.copy()
    for _ in range(iterations):
        system = sparse.spdiags(weights, 0, len(values), len(values), format="csc") + penalty
        baseline = np.asarray(spsolve(system, weights * values))
        weights = np.where(values > baseline, asymmetry, 1.0 - asymmetry)
    return baseline


def airpls_baseline(y: np.ndarray, smoothing: float = 1e6, iterations: int = 25) -> np.ndarray:
    """Adaptive iteratively reweighted penalized least squares baseline."""

    values = np.asarray(y, dtype=float)
    if smoothing <= 0 or iterations < 1:
        raise ValueError("Недопустимые параметры airPLS.")
    difference = _difference_matrix(len(values))
    penalty = smoothing * (difference.T @ difference)
    weights = np.ones(len(values))
    baseline = values.copy()
    for iteration in range(1, iterations + 1):
        system = sparse.spdiags(weights, 0, len(values), len(values), format="csc") + penalty
        baseline = np.asarray(spsolve(system, weights * values))
        residual = values - baseline
        negative = residual[residual < 0]
        negative_sum = float(np.abs(negative).sum())
        if negative_sum < 1e-12:
            break
        weights = np.zeros(len(values))
        below = residual < 0
        weights[below] = np.exp(iteration * np.abs(residual[below]) / negative_sum)
        edge_weight = float(np.exp(iteration * max(np.abs(negative).max(), 1e-12) / negative_sum))
        weights[0] = edge_weight
        weights[-1] = edge_weight
    return baseline


def estimate_baseline(
    x: np.ndarray,
    y: np.ndarray,
    method: BaselineMethod,
    smoothing: float = 1e6,
    asymmetry: float = 0.01,
    iterations: int = 12,
    polynomial_order: int = 2,
) -> BaselineResult:
    """Fit a baseline and return it together with the corrected signal."""

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if len(x_values) != len(y_values):
        raise ValueError("Размеры массивов X и Y не совпадают.")
    if method == "linear":
        baseline = np.linspace(y_values[0], y_values[-1], len(y_values))
    elif method == "polynomial":
        order = min(max(1, polynomial_order), len(y_values) - 1)
        baseline = np.polyval(np.polyfit(x_values, y_values, order), x_values)
    elif method == "airpls":
        baseline = airpls_baseline(y_values, smoothing, iterations)
    else:
        baseline = als_baseline(y_values, smoothing, asymmetry, iterations)
    return BaselineResult(method, baseline, y_values - baseline)
