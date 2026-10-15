import unittest

from experimental_parser import ExperimentalImportError, parse_experimental_text


class ExperimentalParserTests(unittest.TestCase):
    def test_reads_header_and_semicolon_decimal_comma(self):
        spectrum = parse_experimental_text("Wavenumber;Intensity\n100,0;2,5\n200,0;4,0\n300,0;1,0\n")

        self.assertEqual(spectrum.x.tolist(), [100.0, 200.0, 300.0])
        self.assertEqual(spectrum.y.tolist(), [2.5, 4.0, 1.0])

    def test_rejects_too_short_input(self):
        with self.assertRaises(ExperimentalImportError):
            parse_experimental_text("100 1\n200 2\n")


if __name__ == "__main__":
    unittest.main()
