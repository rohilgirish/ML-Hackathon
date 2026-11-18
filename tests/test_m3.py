"""Unit and integration test suite for Member 3 (M3: Random Forest & MLP)."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.features import FEATURES
from pathsnap.models import MODEL_ZOO


class TestMember3(unittest.TestCase):
    def test_01_model_zoo_structure(self):
        """Verify M3 models are defined with correct hyperparameters."""
        self.assertIn("RandomForest", MODEL_ZOO)
        self.assertIn("MLP", MODEL_ZOO)
        self.assertEqual(len(MODEL_ZOO), 2)

        rf_fam, rf_fac, rf_desc, _ = MODEL_ZOO["RandomForest"]
        self.assertEqual(rf_fam, "Ensemble (bagging)")
        rf = rf_fac()
        self.assertEqual(rf.n_estimators, 150)
        self.assertEqual(rf.min_samples_leaf, 5)
        self.assertEqual(rf.max_features, "sqrt")

        mlp_fam, mlp_fac, mlp_desc, _ = MODEL_ZOO["MLP"]
        self.assertEqual(mlp_fam, "Neural network")
        mlp_pipe = mlp_fac()
        self.assertEqual(len(mlp_pipe.steps), 2)
        self.assertEqual(mlp_pipe.steps[0][0], "standardscaler")
        self.assertEqual(mlp_pipe.steps[1][0], "mlpclassifier")

    def test_02_synthetic_training_inference(self):
        """Verify both models train and output valid probabilities on 24 features."""
        rng = np.random.default_rng(42)
        N = 200
        X = rng.normal(size=(N, len(FEATURES))).astype(np.float32)
        y = rng.integers(0, 2, size=N)

        for name in ["RandomForest", "MLP"]:
            m = MODEL_ZOO[name][1]()
            m.fit(X, y)
            probs = m.predict_proba(X)
            self.assertEqual(probs.shape, (N, 2))
            self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

    def test_03_benchmark_results_exist(self):
        """Verify student_test and evaluator_test benchmark tables exist and are valid."""
        cmp_path = config.RESULTS_DIR / "model_comparison.csv"
        self.assertTrue(cmp_path.exists(), "results/model_comparison.csv must exist")
        df_cmp = pd.read_csv(cmp_path)
        self.assertIn("RandomForest", df_cmp.model.values)
        self.assertIn("MLP", df_cmp.model.values)
        self.assertTrue(all(df_cmp.test_top1 > 0.85))

        eval_path = config.RESULTS_DIR / "evaluator_test_comparison.csv"
        self.assertTrue(eval_path.exists(), "results/evaluator_test_comparison.csv must exist")
        df_eval = pd.read_csv(eval_path)
        self.assertIn("RandomForest", df_eval.model.values)
        self.assertIn("MLP", df_eval.model.values)
        self.assertTrue(all(df_eval.top1_acc > 0.85))

    def test_04_prediction_parquets(self):
        """Verify parquet predictions exist for student_test and evaluator_test."""
        for pdir in ["student_test_scores", "evaluator_test_scores"]:
            for mname in ["RandomForest", "MLP"]:
                pq_path = config.RESULTS_DIR / pdir / f"{mname}.parquet"
                self.assertTrue(pq_path.exists(), f"{pq_path} must exist")
                df = pd.read_parquet(pq_path)
                self.assertListEqual(list(df.columns), ["trip", "obs", "arc", "p"])
                self.assertTrue(np.all((df.p >= 0.0) & (df.p <= 1.0)))

    def test_05_figures_generated(self):
        """Verify all M3 diagnostic and route figures are generated."""
        expected_figs = [
            "cmp_cv_macro_f1.png",
            "cmp_cv_folds_box.png",
            "cmp_test_top1_f1.png",
            "cmp_logloss_inference.png",
            "cmp_train_vs_test.png",
            "cmp_confusion_matrices.png",
            "cmp_training_histories.png",
            "cmp_actual_vs_predicted_path.png",
            "evaluator_test_actual_vs_predicted.png",
        ]
        for fig_name in expected_figs:
            fig_path = config.FIG_DIR / fig_name
            self.assertTrue(fig_path.exists(), f"Figure {fig_name} must exist in {config.FIG_DIR}")
            self.assertGreater(fig_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
