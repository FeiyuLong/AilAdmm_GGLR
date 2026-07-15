from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np
from scipy import sparse

from main import RESULTS_DIR, _clear_results_dir, _spectral_squared_norm, main


class CliTests(unittest.TestCase):
    def test_dry_run_does_not_touch_data(self) -> None:
        status = main(["--dry-run", "--data-dir", "does-not-exist"])
        self.assertEqual(status, 0)

    def test_removed_output_override_options_are_rejected(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--dry-run", "--results-dir", "somewhere"])
            with self.assertRaises(SystemExit):
                main(["--dry-run", "--plot-formats", "png"])

    def test_renamed_ail_svrg_cli_name_is_required(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--dry-run", "--algorithms", "AIL-SVRG-ADMM"]), 0)
            with self.assertRaises(SystemExit):
                main(["--dry-run", "--algorithms", "Ail-ADMM"])

    def test_results_cleanup_removes_all_prior_artifacts(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / RESULTS_DIR
            (output / "nested").mkdir(parents=True)
            (output / "old.pdf").write_text("old", encoding="utf-8")
            (output / "nested" / "raw.csv").write_text("old", encoding="utf-8")
            _clear_results_dir(output)
            self.assertEqual(list(output.iterdir()), [])

    def test_incidence_spectral_squared_norm_does_not_use_nullspace_start(self) -> None:
        incidence = sparse.csr_matrix(
            [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]]
        )
        value = _spectral_squared_norm(incidence)
        self.assertGreater(value, 3.0)
        self.assertAlmostEqual(value, 3.0, places=5)

    def test_zero_and_nonfinite_spectral_inputs_are_handled(self) -> None:
        self.assertEqual(_spectral_squared_norm(sparse.csr_matrix((2, 3))), 0.0)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _spectral_squared_norm(sparse.csr_matrix([[np.nan, 0.0]]))


if __name__ == "__main__":
    unittest.main()
