from __future__ import annotations

import contextlib
import io
import json
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

    def _dry_run_algorithms(self, *names: str) -> list[str]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["--dry-run", "--algorithms", *names]), 0)
        return json.loads(output.getvalue())["algorithms"]

    def test_full_names_and_exact_short_names_are_equivalent(self) -> None:
        canonical = ["SVRG-ADMM", "ASVRG-ADMM", "SPIDER-ADMM"]
        self.assertEqual(self._dry_run_algorithms(*canonical), canonical)
        self.assertEqual(self._dry_run_algorithms("SVRG", "ASVRG", "SPIDER"), canonical)

    def test_ailsvrg_short_names_resolve_to_canonical_names(self) -> None:
        short_names = ["AILSVRG", "AILSVRG-NoMom", "AILSVRG-Fixed-p", "AILSVRG-WithCorr"]
        expected = [
            "AILSVRG-ADMM", "AILSVRG-ADMM-NoMom",
            "AILSVRG-ADMM-Fixed-p", "AILSVRG-ADMM-WithCorr",
        ]
        self.assertEqual(self._dry_run_algorithms(*short_names), expected)

    def test_partial_lowercase_unknown_and_legacy_names_are_rejected(self) -> None:
        invalid = ("ASVR", "AIL", "SPI", "svrg", "ailsvrg", "UNKNOWN", "AIL" + "-SVRG-ADMM")
        for name in invalid:
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["--dry-run", "--algorithms", name])

    def test_aliases_are_deduplicated_in_first_seen_order(self) -> None:
        self.assertEqual(
            self._dry_run_algorithms("SPIDER", "SVRG", "SPIDER-ADMM", "SVRG-ADMM"),
            ["SPIDER-ADMM", "SVRG-ADMM"],
        )

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
