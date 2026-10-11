import unittest

from gaussian_parser import GaussianImportError, parse_gaussian_text


COMPLETED_LOG = """
 # freq=raman b3lyp/6-31g(d)

water

 Frequencies --   100.0000               200.0000               300.0000
 IR Inten    --     1.0000                 2.0000                 3.0000
 Raman Activ --     4.0000                 5.0000                 6.0000
 Normal termination of Gaussian 09
"""


class GaussianParserTests(unittest.TestCase):
    def test_extracts_complete_vibrational_group(self):
        calculation = parse_gaussian_text(COMPLETED_LOG)

        self.assertEqual(calculation.title, "water")
        self.assertEqual(len(calculation.modes), 3)
        self.assertEqual(calculation.modes[1].frequency, 200.0)
        self.assertEqual(calculation.modes[2].ir_intensity, 3.0)
        self.assertEqual(calculation.modes[0].raman_activity, 4.0)

    def test_rejects_unfinished_log(self):
        with self.assertRaisesRegex(GaussianImportError, "Normal termination"):
            parse_gaussian_text(COMPLETED_LOG.replace(" Normal termination of Gaussian 09", ""))

    def test_rejects_error_as_last_termination(self):
        failed = COMPLETED_LOG.replace("Normal termination", "Error termination")
        with self.assertRaisesRegex(GaussianImportError, "Error termination"):
            parse_gaussian_text(failed)


if __name__ == "__main__":
    unittest.main()
