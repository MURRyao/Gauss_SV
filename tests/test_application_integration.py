"""Offscreen integration tests for the import, UI, export, and project workflows."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from experimental_parser import parse_experimental_file
from gaussian_parser import parse_gaussian_log
from spectra_app import MainWindow


FIXTURES = Path(__file__).parent / "fixtures"


class ApplicationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = MainWindow()
        self.window.resize(1200, 800)
        self.window.show()
        self.app.processEvents()

    def tearDown(self) -> None:
        self.window.close()
        self.app.processEvents()

    def test_import_render_export_and_restore_project(self) -> None:
        calculation = parse_gaussian_log(FIXTURES / "water.log")
        experimental = parse_experimental_file(FIXTURES / "experimental.csv")

        self.window._set_calculation(calculation)
        self.window._set_experimental(experimental)
        self.window._apply_baseline()
        self.window.width_spin.setValue(24.0)
        self.window.reverse_axis.setChecked(True)
        self.app.processEvents()

        self.assertEqual(self.window.table.rowCount(), 3)
        self.assertTrue(self.window.baseline_box.isEnabled())
        self.assertTrue(self.window.export_button.isEnabled())
        self.assertTrue(self.window.save_project_button.isEnabled())

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            image_path = directory_path / "spectrum.png"
            project_path = directory_path / "water-spectrum.json"
            self.window.export_graph(image_path)
            self.window.save_project(project_path)

            self.assertGreater(image_path.stat().st_size, 100)
            self.assertTrue(project_path.is_file())

            restored = MainWindow()
            restored.resize(1200, 800)
            restored.show()
            self.app.processEvents()
            try:
                restored.load_project(project_path)
                self.assertEqual(restored.table.rowCount(), 3)
                self.assertEqual(len(restored.experimental.x), 7)
                self.assertIsNotNone(restored.baseline)
                self.assertAlmostEqual(restored.width_spin.value(), 24.0)
                self.assertTrue(restored.reverse_axis.isChecked())
            finally:
                restored.close()


if __name__ == "__main__":
    unittest.main()
