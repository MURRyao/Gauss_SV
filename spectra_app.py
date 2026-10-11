"""Gaussian Spectra desktop workspace."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFrame,
    QFileDialog,
    QGraphicsOpacityEffect,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gaussian_parser import GaussianCalculation, GaussianImportError, VibrationalMode, parse_gaussian_log

class SpectrumPreview(QWidget):
    """Lightweight interactive preview of a calculated spectrum."""

    peak_selected = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(330)
        self.setMouseTracking(True)
        self._mode = "raman"
        self._profile = "Гауссовский"
        self._width = 18.0
        self._selected = 0
        self._modes: list[VibrationalMode] = []
        self._x_min, self._x_max = 0.0, 4000.0

    def set_modes(self, modes: list[VibrationalMode]) -> None:
        self._modes = modes
        self._selected = min(self._selected, len(modes) - 1) if modes else 0
        self.update()

    def set_spectrum_kind(self, kind: str) -> None:
        self._mode = kind
        self.update()

    def set_profile(self, profile: str) -> None:
        self._profile = profile
        self.update()

    def set_width_value(self, width: int) -> None:
        self._width = float(width)
        self.update()

    def select_peak(self, row: int) -> None:
        self._selected = row
        self.update()

    def _intensity(self, mode: VibrationalMode) -> float:
        return mode.raman_activity if self._mode == "raman" else mode.ir_intensity

    def _value_at(self, x: float, maximum: float) -> float:
        total = 0.0
        for mode in self._modes:
            amplitude = self._intensity(mode) / maximum
            dx = (x - mode.frequency) / self._width
            if self._profile == "Лоренцевский":
                total += amplitude / (1.0 + dx * dx)
            elif self._profile == "Voigt":
                total += amplitude * (0.55 * math.exp(-0.5 * dx * dx) + 0.45 / (1.0 + dx * dx))
            else:
                total += amplitude * math.exp(-0.5 * dx * dx)
        return total

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        plot = QRectF(72, 28, self.width() - 104, self.height() - 86)
        if plot.width() <= 0 or plot.height() <= 0:
            return

        grid_pen = QPen(QColor("#d9d9d9"), 1)
        painter.setPen(grid_pen)
        for index in range(5):
            y = plot.top() + index * plot.height() / 4
            painter.drawLine(plot.left(), y, plot.right(), y)
        for index in range(5):
            x = plot.left() + index * plot.width() / 4
            painter.drawLine(x, plot.top(), x, plot.bottom())

        painter.setPen(QPen(QColor("#727272"), 1.0))
        painter.drawLine(plot.left(), plot.bottom(), plot.right(), plot.bottom())
        painter.drawLine(plot.left(), plot.top(), plot.left(), plot.bottom())

        tick_font = QFont("Inter", 9)
        painter.setFont(tick_font)
        painter.setPen(QColor("#4d4d4d"))
        for value in range(0, 4001, 1000):
            x = plot.left() + plot.width() * value / 4000
            painter.drawText(QRectF(x - 24, plot.bottom() + 11, 48, 18), Qt.AlignmentFlag.AlignCenter, str(value))
        painter.drawText(QRectF(plot.left(), plot.bottom() + 36, plot.width(), 18), Qt.AlignmentFlag.AlignCenter, "Волновое число, см⁻¹")
        painter.save()
        painter.translate(18, plot.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-plot.height() / 2, -10, plot.height(), 18), Qt.AlignmentFlag.AlignCenter, "Относительная интенсивность")
        painter.restore()

        if not self._modes:
            return
        maximum = max(self._intensity(mode) for mode in self._modes)
        path = QPainterPath()
        steps = max(300, int(plot.width() * 1.6))
        for index in range(steps):
            x_value = self._x_min + (self._x_max - self._x_min) * index / (steps - 1)
            y_value = min(1.06, self._value_at(x_value, maximum))
            x = plot.left() + (x_value - self._x_min) * plot.width() / (self._x_max - self._x_min)
            y = plot.bottom() - y_value * (plot.height() * 0.91)
            if index == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)

        fill_path = QPainterPath(path)
        fill_path.lineTo(plot.right(), plot.bottom())
        fill_path.lineTo(plot.left(), plot.bottom())
        fill_path.closeSubpath()
        gradient = QLinearGradient(plot.left(), plot.top(), plot.right(), plot.bottom())
        gradient.setColorAt(0, QColor(31, 91, 140, 22))
        gradient.setColorAt(1, QColor(31, 91, 140, 2))
        painter.fillPath(fill_path, gradient)
        painter.setPen(QPen(QColor("#1f5b8c"), 1.8))
        painter.drawPath(path)

        selected_mode = self._modes[self._selected]
        x = plot.left() + selected_mode.frequency / 4000 * plot.width()
        y = plot.bottom() - min(1.06, self._value_at(selected_mode.frequency, maximum)) * (plot.height() * 0.91)
        painter.setPen(QPen(QColor("#a53a2a"), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(x, plot.top() + 4, x, plot.bottom())
        painter.setBrush(QColor("#ffffff"))
        painter.setPen(QPen(QColor("#a53a2a"), 1.5))
        painter.drawEllipse(x - 4.5, y - 4.5, 9, 9)
        painter.setPen(QColor("#8b3023"))
        painter.setFont(QFont("Inter", 9, QFont.Weight.DemiBold))
        painter.drawText(QRectF(x + 8, max(plot.top() + 4, y - 30), 130, 22), f"{selected_mode.frequency:.1f} см⁻¹")

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() != Qt.MouseButton.LeftButton:
            return
        plot_left, plot_width = 72, self.width() - 104
        if plot_width <= 0:
            return
        frequency = (event.position().x() - plot_left) / plot_width * 4000
        if not self._modes:
            return
        nearest = min(range(len(self._modes)), key=lambda i: abs(self._modes[i].frequency - frequency))
        self.peak_selected.emit(nearest)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gaussian Spectra")
        self.resize(1460, 900)
        self.setMinimumSize(1120, 700)
        self.modes: list[VibrationalMode] = []
        self.calculation: GaussianCalculation | None = None
        self.preview = SpectrumPreview()
        self.table = QTableWidget(0, 4)
        self._build_ui()
        self._populate_table()
        self._animate_workspace()

    def _build_ui(self) -> None:
        self._build_menu()
        outer = QWidget()
        outer.setObjectName("root")
        self.setCentralWidget(outer)
        layout = QVBoxLayout(outer)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("main_splitter")
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._controls_panel())
        splitter.addWidget(self._work_area())
        splitter.addWidget(self._actions_panel())
        splitter.setSizes([255, 930, 235])
        layout.addWidget(splitter, 1)

        self.statusBar().showMessage("Готово")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("Файл")
        file_menu.addAction("Открыть Gaussian log…", self._open_gaussian_log)
        save_action = file_menu.addAction("Сохранить проект…")
        save_action.setEnabled(False)
        file_menu.addSeparator()
        file_menu.addAction("Выход", self.close)

        self.menuBar().addMenu("Спектр")
        view_menu = self.menuBar().addMenu("Вид")
        view_menu.addAction("Сбросить выделение", lambda: self._select_mode(0))
        self.menuBar().addMenu("Справка")

    def _controls_panel(self) -> QWidget:
        panel = QGroupBox("Построение спектра")
        panel.setObjectName("panel")
        panel.setMinimumWidth(220)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 18, 12, 12)
        layout.setSpacing(9)

        spectrum_label = QLabel("ТИП СПЕКТРА")
        spectrum_label.setObjectName("field_label")
        layout.addWidget(spectrum_label)
        raman = QRadioButton("Raman")
        ir = QRadioButton("ИК")
        raman.setChecked(True)
        group = QButtonGroup(self)
        group.addButton(raman)
        group.addButton(ir)
        raman.toggled.connect(lambda checked: checked and self.preview.set_spectrum_kind("raman"))
        ir.toggled.connect(lambda checked: checked and self.preview.set_spectrum_kind("ir"))
        layout.addWidget(raman)
        layout.addWidget(ir)

        layout.addWidget(self._divider())
        layout.addWidget(self._field_label("Контур"))
        profile = QComboBox()
        profile.addItems(["Гауссовский", "Лоренцевский", "Voigt"])
        profile.currentTextChanged.connect(self.preview.set_profile)
        layout.addWidget(profile)
        layout.addWidget(self._field_label("Ширина пика · см⁻¹"))
        self.width_value = QLabel("18")
        self.width_value.setObjectName("control_value")
        width_row = QHBoxLayout()
        width = QSlider(Qt.Orientation.Horizontal)
        width.setRange(5, 60)
        width.setValue(18)
        width.valueChanged.connect(lambda value: (self.width_value.setText(str(value)), self.preview.set_width_value(value)))
        width_row.addWidget(width)
        width_row.addWidget(self.width_value)
        layout.addLayout(width_row)

        layout.addWidget(self._divider())
        layout.addWidget(self._field_label("Диапазон · см⁻¹"))
        range_row = QHBoxLayout()
        lower = QSpinBox()
        lower.setRange(0, 3999)
        lower.setValue(0)
        upper = QSpinBox()
        upper.setRange(1, 4000)
        upper.setValue(4000)
        range_row.addWidget(lower)
        range_row.addWidget(QLabel("—"))
        range_row.addWidget(upper)
        layout.addLayout(range_row)
        layout.addStretch(1)
        hint = QLabel("Нажмите на пик на графике, чтобы выбрать соответствующую моду.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return panel

    def _work_area(self) -> QWidget:
        workspace = QWidget()
        workspace.setObjectName("workspace")
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.table.setObjectName("mode_table")
        self.table.setHorizontalHeaderLabels(["Мода", "Частота, см⁻¹", "ИК, KM/mol", "Raman, Å⁴/amu"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setMinimumHeight(205)
        self.table.itemSelectionChanged.connect(self._table_selection_changed)
        table_box = QGroupBox("Колебательные моды")
        table_box.setObjectName("data_group")
        table_layout = QVBoxLayout(table_box)
        table_layout.setContentsMargins(7, 18, 7, 7)
        table_layout.addWidget(self.table)
        layout.addWidget(table_box)

        graph_frame = QGroupBox("Спектр")
        graph_frame.setObjectName("graph_frame")
        graph_layout = QVBoxLayout(graph_frame)
        graph_layout.setContentsMargins(7, 18, 7, 7)
        graph_layout.addWidget(self.preview)
        layout.addWidget(graph_frame, 1)
        self.preview.peak_selected.connect(self._select_mode)
        return workspace

    def _actions_panel(self) -> QWidget:
        panel = QGroupBox("Действия")
        panel.setObjectName("panel")
        panel.setMinimumWidth(205)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 18, 12, 12)
        layout.setSpacing(8)
        self.open_log_button = QPushButton("Открыть Gaussian log")
        self.open_log_button.setObjectName("primary")
        self.open_log_button.clicked.connect(self._open_gaussian_log)
        layout.addWidget(self.open_log_button)
        button = QPushButton("Обновить график")
        button.clicked.connect(self.preview.update)
        layout.addWidget(button)
        for text in ("Импортировать XY", "Экспорт графика", "Сохранить проект"):
            action = QPushButton(text)
            action.setEnabled(False)
            layout.addWidget(action)
        layout.addWidget(self._divider())
        layout.addWidget(self._field_label("Состояние"))
        self.status_label = QLabel("Файл не выбран")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch()
        return panel

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setObjectName("divider")
        return line

    @staticmethod
    def _field_label(text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("field_label")
        return label

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self.modes))
        self.table.clearContents()
        for row, mode in enumerate(self.modes):
            values = (str(mode.number), f"{mode.frequency:.4f}", f"{mode.ir_intensity:.4f}", f"{mode.raman_activity:.4f}")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        for column in range(3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        if self.modes:
            self.table.selectRow(min(4, len(self.modes) - 1))

    def _select_mode(self, row: int) -> None:
        if not self.modes:
            return
        row = max(0, min(row, len(self.modes) - 1))
        self.table.selectRow(row)
        self.preview.select_peak(row)

    def _table_selection_changed(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.preview.select_peak(rows[0].row())

    def _open_gaussian_log(self) -> None:
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Открыть Gaussian log",
            "",
            "Gaussian output (*.log *.out);;Все файлы (*)",
        )
        if not selected_path:
            return
        try:
            calculation = parse_gaussian_log(selected_path)
        except GaussianImportError as error:
            self.status_label.setText("Импорт не выполнен")
            self.statusBar().showMessage("Ошибка импорта Gaussian")
            QMessageBox.critical(self, "Gaussian log", str(error))
            return

        self.calculation = calculation
        self.modes = list(calculation.modes)
        self.preview.set_modes(self.modes)
        self._populate_table()
        self._select_mode(0)
        summary = f"{Path(selected_path).name}\n{len(self.modes)} мод"
        if calculation.route:
            summary += f"\n{calculation.route.removeprefix('# ').split()[0]}"
        self.status_label.setText(summary)
        self.statusBar().showMessage(f"Загружено: {Path(selected_path).name}; мод: {len(self.modes)}")

    def _animate_workspace(self) -> None:
        effect = QGraphicsOpacityEffect(self.preview)
        self.preview.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(480)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.start()
        self._intro_animation = animation


STYLE = """
* { font-family: Arial, Helvetica, sans-serif; color: #202020; font-size: 12px; }
QMainWindow, #root { background: #e8e8e8; }
QMenuBar { background: #f3f3f3; border-bottom: 1px solid #a6a6a6; padding: 1px 3px; }
QMenuBar::item { padding: 4px 10px; background: transparent; }
QMenuBar::item:selected, QMenu::item:selected { background: #d9e7f3; }
QMenu { background: #f7f7f7; border: 1px solid #8f8f8f; }
QGroupBox { background: #f6f6f6; border: 1px solid #a9a9a9; margin-top: 8px; padding-top: 7px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #1f1f1f; }
#panel { min-width: 205px; }
#workspace { background: transparent; }
#field_label { color: #505050; font-size: 10px; font-weight: bold; margin-top: 3px; }
#divider { color: #c5c5c5; margin: 3px 0; }
QRadioButton { spacing: 6px; }
QRadioButton::indicator { width: 12px; height: 12px; border: 1px solid #717171; border-radius: 6px; background: white; }
QRadioButton::indicator:checked { background: #1f5b8c; border: 3px solid #ffffff; outline: 1px solid #1f5b8c; }
QComboBox, QSpinBox, QDoubleSpinBox { background: #ffffff; border: 1px solid #949494; border-radius: 1px; padding: 3px 5px; min-height: 18px; }
QComboBox:hover, QSpinBox:hover { border-color: #3d6f99; }
QSlider::groove:horizontal { background: #c9c9c9; height: 4px; }
QSlider::handle:horizontal { background: #1f5b8c; border: 1px solid #174666; width: 10px; margin: -4px 0; }
#control_value { min-width: 23px; color: #1f5b8c; font-weight: bold; }
#hint { color: #3f3f3f; background: #eeeeee; border: 1px solid #c5c5c5; padding: 7px; }
#mode_table { background: #ffffff; border: 1px solid #9c9c9c; gridline-color: #dddddd; selection-background-color: #cfe1f0; selection-color: #161616; }
#mode_table::item { padding: 5px; }
QHeaderView::section { background: #e6e6e6; color: #202020; border: none; border-right: 1px solid #b8b8b8; border-bottom: 1px solid #969696; padding: 5px; font-size: 11px; font-weight: bold; }
#graph_frame { background: #f6f6f6; }
QPushButton { text-align: left; background: #f9f9f9; border: 1px solid #8f8f8f; border-radius: 1px; padding: 6px 8px; min-height: 18px; }
QPushButton:hover:enabled { background: #dceaf5; border-color: #3f6f98; }
QPushButton:disabled { color: #8b8b8b; background: #eeeeee; border-color: #c7c7c7; }
QPushButton#primary { text-align: center; background: #e4edf5; border-color: #557d9f; color: #172d41; font-weight: bold; }
QPushButton#primary:hover { background: #cfe1f0; }
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
