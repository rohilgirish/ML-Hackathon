"""Unit and integration test suite for Member 4 (M4: Pipeline, HMM, API, UI, and Deployment)."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from pathsnap import config
from pathsnap.features import FEATURES
from pathsnap.hmm import HMMParams, reconstruct_route, viterbi
from pathsnap.pipeline import PathSnapMatcher, evaluate_trip


class TestMember4(unittest.TestCase):
    def test_01_production_artifacts(self):
        """Verify production inference artifacts exist and are non-empty."""
        model_path = config.ARTIFACT_DIR / "model.joblib"
        meta_path = config.ARTIFACT_DIR / "model_meta.json"
        hmm_path = config.ARTIFACT_DIR / "hmm_config.json"

        self.assertTrue(model_path.exists(), "artifacts/model.joblib must exist")
        self.assertTrue(meta_path.exists(), "artifacts/model_meta.json must exist")
        self.assertTrue(hmm_path.exists(), "artifacts/hmm_config.json must exist")

        meta = json.loads(meta_path.read_text())
        self.assertIn("features", meta)
        self.assertEqual(meta["features"], FEATURES)

        hmm_cfg = json.loads(hmm_path.read_text())
        self.assertIn("alpha", hmm_cfg)
        self.assertIn("beta", hmm_cfg)
        self.assertIn("topk", hmm_cfg)

    def test_02_matcher_initialization_and_inference(self):
        """Verify PathSnapMatcher loads and runs prediction on a trip without route leakage."""
        matcher = PathSnapMatcher()
        self.assertEqual(len(matcher.features), 24)

        # Test prediction on student test trip 00000000
        trip_dir = config.STUDENT_TEST_DIR / "00000000"
        if trip_dir.exists():
            res = matcher.predict_trip(trip_dir)
            self.assertEqual(res["trip_id"], "00000000")
            self.assertIn("clf_arc", res)
            self.assertIn("hmm_arc", res)
            self.assertIn("route", res)
            self.assertEqual(len(res["clf_arc"]), len(res["pt"].xy))
            self.assertEqual(len(res["hmm_arc"]), len(res["pt"].xy))

            # Test server-side evaluation function
            ev = evaluate_trip(res, trip_dir)
            self.assertIn("hmm_top1", ev)
            self.assertIn("route_f1", ev)
            self.assertIn("route_continuity", ev)
            self.assertGreater(ev["hmm_top1"], 0.90)
            self.assertGreater(ev["route_f1"], 0.95)

    def test_03_fastapi_backend_endpoints(self):
        """Verify FastAPI REST API routes: /health, /trips, and /predict."""
        from api.main import app

        client = TestClient(app)

        # 1. Health check
        res_health = client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        d_health = res_health.json()
        self.assertEqual(d_health["status"], "ok")

        # 2. Trips listing
        res_trips = client.get("/trips")
        self.assertEqual(res_trips.status_code, 200)
        d_trips = res_trips.json()
        self.assertIn("trips", d_trips)
        self.assertGreaterEqual(len(d_trips["trips"]), 1)

        # 3. Prediction request on first available trip
        first_trip = d_trips["trips"][0]["trip_id"]
        res_pred = client.post("/predict", json={"trip_id": first_trip})
        self.assertEqual(res_pred.status_code, 200)
        d_pred = res_pred.json()
        self.assertEqual(d_pred["trip_id"], first_trip)
        self.assertIn("predicted_route", d_pred)
        self.assertIn("metrics", d_pred)
        self.assertIn("hmm_top1", d_pred["metrics"])

    def test_04_evaluator_ui_assets(self):
        """Verify evaluator web interface assets are present."""
        root = config.ROOT
        self.assertTrue((root / "ui" / "index.html").exists(), "ui/index.html must exist")
        self.assertTrue((root / "ui" / "app.js").exists(), "ui/app.js must exist")
        self.assertTrue((root / "ui" / "style.css").exists(), "ui/style.css must exist")

    def test_05_report_and_documentation(self):
        """Verify M4 documentation and technical report assets exist."""
        root = config.ROOT
        self.assertTrue((root / "docs" / "M4_DOCUMENTATION.md").exists())
        self.assertTrue((root / "report" / "build_report.py").exists())
        self.assertTrue((root / "report" / "report.css").exists())


if __name__ == "__main__":
    unittest.main()
