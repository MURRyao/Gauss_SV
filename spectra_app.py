"""Gaussian Spectra desktop workspace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QFrame, QGroupBox, QHeaderView, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QPushButton, QRadioButton,
    QScrollArea, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from experimental_parser import ExperimentalImportError, ExperimentalSpectrum, parse_experimental_file
from gaussian_parser import GaussianCalculation, GaussianImportError, VibrationalMode, parse_gaussian_log
from project_io import ProjectFormatError, load_project as read_project, save_project as write_project
from spectrum_processing import BaselineResult, TheoreticalSettings, calculate_theoretical_spectrum, estimate_baseline


class SpectrumPreview(QWidget):
    """Draw theoretical and experimental spectra in a shared scientific plot."""

    peak_selected = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(330)
        self.setMouseTracking(True)
        self._modes: list[VibrationalMode] = []
        self._experimental: ExperimentalSpectrum | None = None
        self._baseline: BaselineResult | None = None
        self._kind = "raman"
        self._settings = TheoreticalSettings()
        self._x_min, self._x_max = 0.0, 4000.0
        self._reverse_x = False
        self._selected = 0

    def configure(self, kind: str, settings: TheoreticalSettings, x_min: float, x_max: float, reverse_x: bool) -> None:
        self._kind, self._settings = kind, settings
        self._x_min, self._x_max, self._reverse_x = x_min, x_max, reverse_x
        self.update()

    def set_modes(self, modes: list[VibrationalMode]) -> None:
        self._modes = modes
        self._selected = min(self._selected, len(modes) - 1) if modes else 0
        self.update()

    def set_experimental(self, spectrum: ExperimentalSpectrum | None) -> None:
        self._experimental = spectrum
        self.update()

    def set_baseline(self, result: BaselineResult | None) -> None:
        self._baseline = result
        self.update()

    def select_peak(self, row: int) -> None:
        if self._modes:
            self._selected = max(0, min(row, len(self._modes) - 1))
        self.update()

    def _map_x(self, value: np.ndarray | float, plot: QRectF) -> np.ndarray | float:
        fraction = (np.asarray(value) - self._x_min) / (self._x_max - self._x_min)
        if self._reverse_x:
            fraction = 1.0 - fraction
        result = plot.left() + fraction * plot.width()
        return float(result) if np.ndim(result) == 0 else result

    @staticmethod
    def _path(xs: np.ndarray, ys: np.ndarray) -> QPainterPath:
        path = QPainterPath()
        if len(xs):
            path.moveTo(float(xs[0]), float(ys[0]))
            for x, y in zip(xs[1:], ys[1:]):
                path.lineTo(float(x), float(y))
        return path

    def _draw_legend(self, painter: QPainter, plot: QRectF, has_experimental: bool) -> None:
        entries = [("Расчёт", QColor("#1f5b8c"), Qt.PenStyle.SolidLine)]
        if has_experimental:
            entries += [
                ("Эксперимент", QColor("#606060"), Qt.PenStyle.SolidLine),
                ("Baseline", QColor("#a35a22"), Qt.PenStyle.DashLine),
                ("После коррекции", QColor("#7a2830"), Qt.PenStyle.SolidLine),
            ]
        painter.setFont(QFont("Arial", 8))
        x, y = plot.right() - 142, plot.top() + 12
        for label, color, style in entries:
            painter.setPen(QPen(color, 1.4, style))
            painter.drawLine(x, y, x + 16, y)
            painter.setPen(QColor("#303030"))
            painter.drawText(x + 21, y + 3, label)
            y += 14

    def _draw_experimental(self, painter: QPainter, plot: QRectF) -> None:
        if self._experimental is None:
            return
        x, raw = self._experimental.x, self._experimental.y
        baseline = self._baseline.baseline if self._baseline is not None else None
        corrected = self._baseline.corrected if self._baseline is not None else None
        arrays = [raw] + ([baseline, corrected] if baseline is not None and corrected is not None else [])
        scale = max(float(np.max(np.abs(values))) for values in arrays) or 1.0
        inside = (x >= self._x_min) & (x <= self._x_max)
        if not np.any(inside):
            return
        xs = self._map_x(x[inside], plot)
        y_bottom, height = plot.bottom(), plot.height() * 0.84
        painter.setClipRect(plot)
        painter.setPen(QPen(QColor("#6a6a6a"), 1.0))
        painter.drawPath(self._path(xs, y_bottom - raw[inside] / scale * height))
        if baseline is not None and corrected is not None:
            painter.setPen(QPen(QColor("#a35a22"), 1.0, Qt.PenStyle.DashLine))
            painter.drawPath(self._path(xs, y_bottom - baseline[inside] / scale * height))
            painter.setPen(QPen(QColor("#7a2830"), 1.2))
            painter.drawPath(self._path(xs, y_bottom - corrected[inside] / scale * height))
        painter.setClipping(False)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        plot = QRectF(72, 28, self.width() - 104, self.height() - 86)
        if plot.width() <= 0 or plot.height() <= 0 or self._x_max <= self._x_min:
            return

        painter.setPen(QPen(QColor("#d9d9d9"), 1))
        for index in range(5):
            painter.drawLine(plot.left(), plot.top() + index * plot.height() / 4, plot.right(), plot.top() + index * plot.height() / 4)
        for value in np.linspace(self._x_min, self._x_max, 5):
            x = self._map_x(value, plot)
            painter.drawLine(x, plot.top(), x, plot.bottom())
        painter.setPen(QPen(QColor("#727272"), 1))
        painter.drawLine(plot.left(), plot.bottom(), plot.right(), plot.bottom())
        painter.drawLine(plot.left(), plot.top(), plot.left(), plot.bottom())
        painter.setFont(QFont("Arial", 9))
        painter.setPen(QColor("#4d4d4d"))
        for value in np.linspace(self._x_min, self._x_max, 5):
            x = self._map_x(value, plot)
            painter.drawText(QRectF(x - 28, plot.bottom() + 11, 56, 18), Qt.AlignmentFlag.AlignCenter, f"{value:.0f}")
        painter.drawText(QRectF(plot.left(), plot.bottom() + 36, plot.width(), 18), Qt.AlignmentFlag.AlignCenter, "Волновое число, см⁻¹")
        painter.save()
        painter.translate(18, plot.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-plot.height() / 2, -10, plot.height(), 18), Qt.AlignmentFlag.AlignCenter, "Относительная интенсивность")
        painter.restore()

        self._draw_experimental(painter, plot)
        if self._modes:
            x_values, y_values = calculate_theoretical_spectrum(
                self._modes, self._x_min, self._x_max, self._settings, self._kind, max(800, int(plot.width() * 1.8))
            )
            scale = float(np.max(np.abs(y_values))) or 1.0
            xs = self._map_x(x_values, plot)
            ys = plot.bottom() - y_values / scale * (plot.height() * 0.91)
            painter.setClipRect(plot)
            painter.setPen(QPen(QColor("#1f5b8c"), 1.8))
            painter.drawPath(self._path(xs, ys))
            painter.setClipping(False)
            selected = self._modes[self._selected]
            center = selected.frequency * self._settings.frequency_scale
            if self._x_min <= center <= self._x_max:
                x = self._map_x(center, plot)
                index = int(np.argmin(np.abs(x_values - center)))
                y = float(ys[index])
                painter.setPen(QPen(QColor("#a53a2a"), 1.0, Qt.PenStyle.DashLine))
                painter.drawLine(x, plot.top() + 4, x, plot.bottom())
                painter.setBrush(QColor("#ffffff"))
                painter.setPen(QPen(QColor("#a53a2a"), 1.5))
                painter.drawEllipse(x - 4, y - 4, 8, 8)
                painter.setPen(QColor("#8b3023"))
                painter.drawText(QRectF(x + 8, max(plot.top() + 4, y - 25), 135, 18), f"{center:.1f} см⁻¹")
        self._draw_legend(painter, plot, self._experimental is not None)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or not self._modes:
            return
        plot = QRectF(72, 28, self.width() - 104, self.height() - 86)
        fraction = (event.position().x() - plot.left()) / plot.width()
        if self._reverse_x:
            fraction = 1.0 - fraction
        frequency = self._x_min + fraction * (self._x_max - self._x_min)
        nearest = min(range(len(self._modes)), key=lambda i: abs(self._modes[i].frequency * self._settings.frequency_scale - frequency))
        self.peak_selected.emit(nearest)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gaussian Spectra")
        self.resize(1500, 920)
        self.setMinimumSize(1120, 700)
        self.modes: list[VibrationalMode] = []
        self.calculation: GaussianCalculation | None = None
        self.experimental: ExperimentalSpectrum | None = None
        self.baseline: BaselineResult | None = None
        self.preview = SpectrumPreview()
        self.table = QTableWidget(0, 6)
        self._build_ui()
        self._refresh_preview()
        self._sync_actions()

    def _build_ui(self) -> None:
        self._build_menu()
        outer = QWidget(objectName="root")
        self.setCentralWidget(outer)
        layout = QVBoxLayout(outer)
        layout.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._controls_panel())
        splitter.addWidget(self._work_area())
        splitter.addWidget(self._actions_panel())
        splitter.setSizes([285, 925, 240])
        layout.addWidget(splitter, 1)
        self.statusBar().showMessage("Готово")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("Файл")
        file_menu.addAction("Открыть Gaussian log…", self._open_gaussian_log)
        file_menu.addAction("Импортировать экспериментальный спектр…", self._open_experimental)
        file_menu.addAction("Открыть проект…", self._open_project)
        self.menu_save_project = file_menu.addAction("Сохранить проект…", self._save_project_dialog)
        file_menu.addSeparator()
        file_menu.addAction("Выход", self.close)

        spectrum_menu = self.menuBar().addMenu("Спектр")
        self.menu_export_graph = spectrum_menu.addAction("Экспортировать график…", self._export_graph_dialog)
        spectrum_menu.addAction("Сбросить параметры спектра", self._reset_spectrum_settings)
        self.menu_clear_experimental = spectrum_menu.addAction("Убрать экспериментальный спектр", self._clear_experimental)
        view_menu = self.menuBar().addMenu("Вид")
        view_menu.addAction("Сбросить выделение", lambda: self._select_mode(0))
        help_menu = self.menuBar().addMenu("Справка")
        help_menu.addAction("Краткое руководство", self._show_quick_help)
        help_menu.addAction("О программе", self._show_about)

    @staticmethod
    def _double(value: float, low: float, high: float, decimals: int = 2) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(low, high)
        widget.setDecimals(decimals)
        widget.setValue(value)
        return widget

    def _controls_panel(self) -> QWidget:
        scroll = QScrollArea(objectName="control_scroll")
        scroll.setWidgetResizable(True)
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        spectrum_box = QGroupBox("Построение спектра")
        spectrum_layout = QVBoxLayout(spectrum_box)
        spectrum_layout.setContentsMargins(12, 18, 12, 12)
        kind_row = QHBoxLayout()
        self.raman_radio, self.ir_radio = QRadioButton("Raman"), QRadioButton("ИК")
        self.raman_radio.setChecked(True)
        kind_group = QButtonGroup(self)
        kind_group.addButton(self.raman_radio)
        kind_group.addButton(self.ir_radio)
        self.raman_radio.toggled.connect(self._on_kind_changed)
        self.ir_radio.toggled.connect(self._on_kind_changed)
        kind_row.addWidget(self.raman_radio)
        kind_row.addWidget(self.ir_radio)
        kind_row.addStretch()
        spectrum_layout.addLayout(kind_row)
        form = QFormLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["Гауссовский", "Лоренцевский", "Voigt"])
        self.profile_combo.currentTextChanged.connect(self._refresh_preview)
        form.addRow("Контур", self.profile_combo)
        self.width_spin = self._double(18.0, 1.0, 200.0, 1)
        self.width_spin.setSuffix(" см⁻¹")
        self.width_spin.valueChanged.connect(self._refresh_preview)
        form.addRow("Ширина пика", self.width_spin)
        self.scale_spin = self._double(1.0, 0.5, 1.5, 4)
        self.scale_spin.valueChanged.connect(self._refresh_preview)
        form.addRow("Масштаб частот", self.scale_spin)
        self.normalization_combo = QComboBox()
        self.normalization_combo.addItems(["Максимум = 1", "Максимум = 100", "Без нормировки"])
        self.normalization_combo.currentTextChanged.connect(self._refresh_preview)
        form.addRow("Нормализация", self.normalization_combo)
        self.x_min_spin, self.x_max_spin = self._double(0.0, -1000.0, 50000.0, 1), self._double(4000.0, -999.0, 50000.0, 1)
        self.x_min_spin.valueChanged.connect(self._refresh_preview)
        self.x_max_spin.valueChanged.connect(self._refresh_preview)
        form.addRow("От, см⁻¹", self.x_min_spin)
        form.addRow("До, см⁻¹", self.x_max_spin)
        self.reverse_axis = QCheckBox("Обратное направление оси X")
        self.reverse_axis.toggled.connect(self._refresh_preview)
        form.addRow(self.reverse_axis)
        spectrum_layout.addLayout(form)
        layout.addWidget(spectrum_box)

        self.raman_box = QGroupBox("Параметры Raman")
        raman_form = QFormLayout(self.raman_box)
        raman_form.setContentsMargins(12, 18, 12, 12)
        self.laser_spin = self._double(514.5, 100.0, 5000.0, 1)
        self.laser_spin.setSuffix(" нм")
        self.laser_spin.valueChanged.connect(self._refresh_preview)
        raman_form.addRow("Лазер", self.laser_spin)
        self.temperature_spin = self._double(300.0, 10.0, 2000.0, 1)
        self.temperature_spin.setSuffix(" K")
        self.temperature_spin.valueChanged.connect(self._refresh_preview)
        raman_form.addRow("Температура", self.temperature_spin)
        self.polarization_combo = QComboBox()
        self.polarization_combo.addItems(["Неполяризованный", "Параллельный", "Перпендикулярный"])
        self.polarization_combo.currentTextChanged.connect(self._refresh_preview)
        raman_form.addRow("Поляризация", self.polarization_combo)
        layout.addWidget(self.raman_box)

        self.baseline_box = QGroupBox("Baseline-коррекция")
        baseline_layout = QVBoxLayout(self.baseline_box)
        baseline_layout.setContentsMargins(12, 18, 12, 12)
        baseline_form = QFormLayout()
        self.baseline_method = QComboBox()
        self.baseline_method.addItems(["ALS", "airPLS", "Линейный", "Полином 2-й степени"])
        baseline_form.addRow("Метод", self.baseline_method)
        self.lambda_spin = QSpinBox()
        self.lambda_spin.setRange(2, 9)
        self.lambda_spin.setValue(6)
        self.lambda_spin.setPrefix("10^")
        baseline_form.addRow("λ", self.lambda_spin)
        self.asymmetry_spin = self._double(0.01, 0.001, 0.499, 3)
        baseline_form.addRow("Асимметрия", self.asymmetry_spin)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(1, 100)
        self.iterations_spin.setValue(12)
        baseline_form.addRow("Итерации", self.iterations_spin)
        baseline_layout.addLayout(baseline_form)
        apply_button = QPushButton("Применить baseline")
        apply_button.clicked.connect(self._apply_baseline)
        baseline_layout.addWidget(apply_button)
        self.baseline_box.setEnabled(False)
        layout.addWidget(self.baseline_box)
        layout.addStretch()
        scroll.setWidget(root)
        return scroll

    def _work_area(self) -> QWidget:
        workspace = QWidget(objectName="workspace")
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.table.setObjectName("mode_table")
        self.table.setHorizontalHeaderLabels(["Мода", "Частота, см⁻¹", "ИК, KM/mol", "Raman, Å⁴/amu", "Depol. P", "Depol. U"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(205)
        self.table.itemSelectionChanged.connect(self._table_selection_changed)
        table_box = QGroupBox("Колебательные моды")
        table_layout = QVBoxLayout(table_box)
        table_layout.setContentsMargins(7, 18, 7, 7)
        table_layout.addWidget(self.table)
        layout.addWidget(table_box)
        graph_box = QGroupBox("Спектр")
        graph_layout = QVBoxLayout(graph_box)
        graph_layout.setContentsMargins(7, 18, 7, 7)
        graph_layout.addWidget(self.preview)
        layout.addWidget(graph_box, 1)
        self.preview.peak_selected.connect(self._select_mode)
        return workspace

    def _actions_panel(self) -> QWidget:
        panel = QGroupBox("Действия")
        panel.setMinimumWidth(205)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 18, 12, 12)
        layout.setSpacing(8)
        gaussian_button = QPushButton("Открыть Gaussian log")
        gaussian_button.setObjectName("primary")
        gaussian_button.clicked.connect(self._open_gaussian_log)
        layout.addWidget(gaussian_button)
        experimental_button = QPushButton("Импортировать XY")
        experimental_button.clicked.connect(self._open_experimental)
        layout.addWidget(experimental_button)
        refresh_button = QPushButton("Обновить график")
        refresh_button.clicked.connect(self._refresh_preview)
        layout.addWidget(refresh_button)
        self.export_button = QPushButton("Экспорт графика")
        self.export_button.clicked.connect(self._export_graph_dialog)
        layout.addWidget(self.export_button)
        self.save_project_button = QPushButton("Сохранить проект")
        self.save_project_button.clicked.connect(self._save_project_dialog)
        layout.addWidget(self.save_project_button)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)
        layout.addWidget(QLabel("СОСТОЯНИЕ", objectName="field_label"))
        self.status_label = QLabel("Файлы не выбраны", objectName="status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch()
        return panel

    def _on_kind_changed(self) -> None:
        self.raman_box.setEnabled(self.raman_radio.isChecked())
        self._refresh_preview()

    def _settings(self) -> TheoreticalSettings:
        profile = {"Гауссовский": "gaussian", "Лоренцевский": "lorentzian", "Voigt": "voigt"}[self.profile_combo.currentText()]
        normalization = {"Максимум = 1": "unit", "Максимум = 100": "percent", "Без нормировки": "none"}[self.normalization_combo.currentText()]
        polarization = {"Неполяризованный": "unpolarized", "Параллельный": "parallel", "Перпендикулярный": "perpendicular"}[self.polarization_combo.currentText()]
        return TheoreticalSettings(profile, self.width_spin.value(), self.scale_spin.value(), normalization, self.laser_spin.value(), self.temperature_spin.value(), polarization)

    def _refresh_preview(self) -> None:
        self.preview.configure(
            "raman" if self.raman_radio.isChecked() else "ir",
            self._settings(),
            self.x_min_spin.value(),
            self.x_max_spin.value(),
            self.reverse_axis.isChecked(),
        )

    def _sync_actions(self) -> None:
        has_data = bool(self.modes) or self.experimental is not None
        self.export_button.setEnabled(has_data)
        self.save_project_button.setEnabled(has_data)
        self.menu_export_graph.setEnabled(has_data)
        self.menu_save_project.setEnabled(has_data)
        self.menu_clear_experimental.setEnabled(self.experimental is not None)

    def _set_calculation(self, calculation: GaussianCalculation | None) -> None:
        self.calculation = calculation
        self.modes = list(calculation.modes) if calculation is not None else []
        self.preview.set_modes(self.modes)
        self._populate_table()
        if self.modes:
            self._select_mode(0)
        self._sync_actions()

    def _set_experimental(self, spectrum: ExperimentalSpectrum | None) -> None:
        self.experimental = spectrum
        self.baseline = None
        self.preview.set_experimental(spectrum)
        self.preview.set_baseline(None)
        self.baseline_box.setEnabled(spectrum is not None)
        self._sync_actions()

    def _open_gaussian_log(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Открыть Gaussian log", "", "Gaussian output (*.log *.out);;Все файлы (*)")
        if not path:
            return
        try:
            calculation = parse_gaussian_log(path)
        except GaussianImportError as error:
            self._show_error("Gaussian log", str(error))
            return
        self._set_calculation(calculation)
        self._update_status(f"Gaussian: {Path(path).name}\n{len(self.modes)} мод")

    def _open_experimental(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Импортировать экспериментальный спектр", "", "Спектры (*.csv *.txt *.xy);;Все файлы (*)")
        if not path:
            return
        try:
            spectrum = parse_experimental_file(path)
        except ExperimentalImportError as error:
            self._show_error("Экспериментальный спектр", str(error))
            return
        self._set_experimental(spectrum)
        if not self.modes:
            self.x_min_spin.setValue(float(self.experimental.x.min()))
            self.x_max_spin.setValue(float(self.experimental.x.max()))
        self._apply_baseline()
        self._update_status(f"Эксперимент: {Path(path).name}\n{len(self.experimental.x)} точек")

    def _project_settings(self) -> dict[str, object]:
        return {
            "kind": "raman" if self.raman_radio.isChecked() else "ir",
            "profile": self.profile_combo.currentText(),
            "width": self.width_spin.value(),
            "frequency_scale": self.scale_spin.value(),
            "normalization": self.normalization_combo.currentText(),
            "x_min": self.x_min_spin.value(),
            "x_max": self.x_max_spin.value(),
            "reverse_x": self.reverse_axis.isChecked(),
            "laser_nm": self.laser_spin.value(),
            "temperature_k": self.temperature_spin.value(),
            "polarization": self.polarization_combo.currentText(),
            "baseline_method": self.baseline_method.currentText(),
            "baseline_lambda_power": self.lambda_spin.value(),
            "baseline_asymmetry": self.asymmetry_spin.value(),
            "baseline_iterations": self.iterations_spin.value(),
        }

    @staticmethod
    def _restore_combo(combo: QComboBox, value: object) -> None:
        if isinstance(value, str):
            index = combo.findText(value)
            if index >= 0:
                combo.setCurrentIndex(index)

    @staticmethod
    def _restore_spin(spin: QDoubleSpinBox | QSpinBox, value: object) -> None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            spin.setValue(int(value) if isinstance(spin, QSpinBox) else float(value))

    def _restore_project_settings(self, settings: dict[str, object]) -> None:
        self.raman_radio.setChecked(settings.get("kind") != "ir")
        self._restore_combo(self.profile_combo, settings.get("profile"))
        self._restore_spin(self.width_spin, settings.get("width"))
        self._restore_spin(self.scale_spin, settings.get("frequency_scale"))
        self._restore_combo(self.normalization_combo, settings.get("normalization"))
        self._restore_spin(self.x_min_spin, settings.get("x_min"))
        self._restore_spin(self.x_max_spin, settings.get("x_max"))
        self.reverse_axis.setChecked(settings.get("reverse_x") is True)
        self._restore_spin(self.laser_spin, settings.get("laser_nm"))
        self._restore_spin(self.temperature_spin, settings.get("temperature_k"))
        self._restore_combo(self.polarization_combo, settings.get("polarization"))
        self._restore_combo(self.baseline_method, settings.get("baseline_method"))
        self._restore_spin(self.lambda_spin, settings.get("baseline_lambda_power"))
        self._restore_spin(self.asymmetry_spin, settings.get("baseline_asymmetry"))
        self._restore_spin(self.iterations_spin, settings.get("baseline_iterations"))
        self._refresh_preview()

    def save_project(self, path: str | Path) -> None:
        """Persist the imported data and current UI state as a JSON project."""

        write_project(path, self.calculation, self.experimental, self._project_settings())

    def load_project(self, path: str | Path) -> None:
        """Restore a saved project without requiring its original source files."""

        state = read_project(path)
        self._set_calculation(state.calculation)
        self._set_experimental(state.experimental)
        self._restore_project_settings(state.settings)
        if self.experimental is not None:
            self._apply_baseline()
        self._update_status(f"Проект: {Path(path).name}")

    def export_graph(self, path: str | Path) -> None:
        """Write the currently visible spectrum plot to a raster image."""

        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".png")
        image = self.preview.grab()
        if image.isNull() or not image.save(str(target)):
            raise OSError(f"Не удалось сохранить изображение: {target}")

    def _save_project_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", "spectrum-project.json", "Проект Gaussian Spectra (*.json)")
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".json")
        try:
            self.save_project(target)
        except ProjectFormatError as error:
            self._show_error("Сохранение проекта", str(error))
            return
        self.statusBar().showMessage(f"Проект сохранён: {target.name}")

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Открыть проект", "", "Проект Gaussian Spectra (*.json);;Все файлы (*)")
        if not path:
            return
        try:
            self.load_project(path)
        except ProjectFormatError as error:
            self._show_error("Открытие проекта", str(error))

    def _export_graph_dialog(self) -> None:
        path, selected_filter = QFileDialog.getSaveFileName(self, "Экспортировать график", "spectrum.png", "PNG (*.png);;JPEG (*.jpg *.jpeg)")
        if not path:
            return
        target = Path(path)
        if not target.suffix:
            target = target.with_suffix(".jpg" if "JPEG" in selected_filter else ".png")
        try:
            self.export_graph(target)
        except OSError as error:
            self._show_error("Экспорт графика", str(error))
            return
        self.statusBar().showMessage(f"График экспортирован: {target.name}")

    def _clear_experimental(self) -> None:
        self._set_experimental(None)
        self._update_status("Экспериментальный спектр удалён")

    def _reset_spectrum_settings(self) -> None:
        self.raman_radio.setChecked(True)
        self.profile_combo.setCurrentText("Гауссовский")
        self.width_spin.setValue(18.0)
        self.scale_spin.setValue(1.0)
        self.normalization_combo.setCurrentText("Максимум = 1")
        self.x_min_spin.setValue(0.0)
        self.x_max_spin.setValue(4000.0)
        self.reverse_axis.setChecked(False)
        self.laser_spin.setValue(514.5)
        self.temperature_spin.setValue(300.0)
        self.polarization_combo.setCurrentText("Неполяризованный")
        self._refresh_preview()
        self.statusBar().showMessage("Параметры спектра сброшены")

    def _show_quick_help(self) -> None:
        QMessageBox.information(self, "Краткое руководство", "1. Откройте Gaussian log или импортируйте XY.\n2. Настройте контур и диапазон спектра.\n3. Для XY выберите baseline и нажмите «Применить baseline».\n4. Экспортируйте график или сохраните проект в JSON.")

    def _show_about(self) -> None:
        QMessageBox.information(self, "О программе", "Gaussian Spectra\nВизуализация и сопоставление ИК/Raman-спектров Gaussian с экспериментальными данными.")

    def _apply_baseline(self) -> None:
        if self.experimental is None:
            return
        method = {"ALS": "als", "airPLS": "airpls", "Линейный": "linear", "Полином 2-й степени": "polynomial"}[self.baseline_method.currentText()]
        try:
            self.baseline = estimate_baseline(
                self.experimental.x, self.experimental.y, method,
                smoothing=10.0**self.lambda_spin.value(),
                asymmetry=self.asymmetry_spin.value(),
                iterations=self.iterations_spin.value(),
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            self._show_error("Baseline-коррекция", str(error))
            return
        self.preview.set_baseline(self.baseline)
        self.statusBar().showMessage(f"Baseline: {self.baseline.method}")

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self.modes))
        self.table.clearContents()
        for row, mode in enumerate(self.modes):
            values = (
                str(mode.number), f"{mode.frequency:.4f}", f"{mode.ir_intensity:.4f}", f"{mode.raman_activity:.4f}",
                "—" if mode.depolarization_p is None else f"{mode.depolarization_p:.4f}",
                "—" if mode.depolarization_u is None else f"{mode.depolarization_u:.4f}",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        for column in range(5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

    def _select_mode(self, row: int) -> None:
        if self.modes:
            row = max(0, min(row, len(self.modes) - 1))
            self.table.selectRow(row)
            self.preview.select_peak(row)

    def _table_selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.preview.select_peak(rows[0].row())

    def _update_status(self, latest: str) -> None:
        items = []
        if self.calculation is not None:
            items.append(f"Gaussian: {self.calculation.source.name}\n{len(self.modes)} мод")
        if self.experimental is not None:
            items.append(f"Эксперимент: {self.experimental.source.name}\n{len(self.experimental.x)} точек")
        self.status_label.setText("\n\n".join(items) if items else latest)
        self.statusBar().showMessage(latest.replace("\n", "; "))

    def _show_error(self, title: str, message: str) -> None:
        self.statusBar().showMessage(f"Ошибка: {message}")
        QMessageBox.critical(self, title, message)


STYLE = """
* { font-family: Arial, Helvetica, sans-serif; color: #202020; font-size: 12px; }
QMainWindow, #root { background: #e8e8e8; }
QMenuBar { background: #f3f3f3; border-bottom: 1px solid #a6a6a6; padding: 1px 3px; }
QMenuBar::item { padding: 4px 10px; background: transparent; }
QMenuBar::item:selected, QMenu::item:selected { background: #d9e7f3; }
QMenu { background: #f7f7f7; border: 1px solid #8f8f8f; }
QGroupBox { background: #f6f6f6; border: 1px solid #a9a9a9; margin-top: 8px; padding-top: 7px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #1f1f1f; }
#control_scroll { border: none; background: transparent; min-width: 270px; }
#control_scroll > QWidget > QWidget { background: #e8e8e8; }
#workspace { background: transparent; }
#field_label { color: #505050; font-size: 10px; font-weight: bold; margin-top: 3px; }
QRadioButton { spacing: 6px; }
QRadioButton::indicator { width: 12px; height: 12px; border: 1px solid #717171; border-radius: 6px; background: white; }
QRadioButton::indicator:checked { background: #1f5b8c; border: 3px solid #ffffff; outline: 1px solid #1f5b8c; }
QComboBox, QSpinBox, QDoubleSpinBox { background: #ffffff; border: 1px solid #949494; border-radius: 1px; padding: 3px 5px; min-height: 18px; }
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #3d6f99; }
#mode_table { background: #ffffff; border: 1px solid #9c9c9c; gridline-color: #dddddd; selection-background-color: #cfe1f0; selection-color: #161616; }
#mode_table::item { padding: 5px; }
QHeaderView::section { background: #e6e6e6; color: #202020; border: none; border-right: 1px solid #b8b8b8; border-bottom: 1px solid #969696; padding: 5px; font-size: 11px; font-weight: bold; }
QPushButton { text-align: left; background: #f9f9f9; border: 1px solid #8f8f8f; border-radius: 1px; padding: 6px 8px; min-height: 18px; }
QPushButton:hover:enabled { background: #dceaf5; border-color: #3f6f98; }
QPushButton:disabled { color: #8b8b8b; background: #eeeeee; border-color: #c7c7c7; }
QPushButton#primary { text-align: center; background: #e4edf5; border-color: #557d9f; color: #172d41; font-weight: bold; }
#status { color: #323232; background: #ffffff; border: 1px solid #adadad; padding: 6px; }
QStatusBar { background: #f2f2f2; border-top: 1px solid #b3b3b3; color: #3b3b3b; }
QToolTip { background: #ffffdc; color: #202020; border: 1px solid #707070; padding: 4px; }
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Gaussian Spectra")
    parser.add_argument("--smoke-test", action="store_true", help="create the window then exit")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    if args.smoke_test:
        window.show()
        app.processEvents()
        return 0
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
