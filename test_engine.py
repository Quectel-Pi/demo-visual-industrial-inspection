import io
import json
import re
import select
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image

from engine import InspectionEngine, enhance_text, refine_region
from decisions import CalibratedDecisions
import server
from server import decode_image


class InspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = InspectionEngine()

    def test_text_enhancement_expands_small_low_contrast_print_without_mutating_source(self):
        image = np.full((32, 96, 3), 80, np.uint8)
        cv2.putText(image, "M2", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (95, 95, 95), 1)
        original = image.copy()
        enhanced = enhance_text(image)
        self.assertEqual(enhanced.shape, (64, 192, 3))
        self.assertGreater(float(enhanced.max()) - float(enhanced.min()), 15)
        np.testing.assert_array_equal(image, original)
        self.assertEqual(float(enhance_text(np.full_like(image, 80)).std()), 0)

    def test_local_region_refines_small_parallax_without_moving_entire_board(self):
        reference = np.random.default_rng(103).integers(20, 230, (100, 140), dtype=np.uint8)
        points = np.float32([[40, 30], [90, 30], [90, 65], [40, 65]])
        observed = reference.copy()
        observed[27:74, 34:99] = 90
        observed[34:69, 45:95] = reference[30:65, 40:90]
        corrected, detail = refine_region(reference, observed, points)
        np.testing.assert_allclose(corrected, points + [5, 4])
        self.assertEqual(detail["offset"], [5, 4])

    def test_local_region_does_not_snap_to_blank_or_unrelated_structure(self):
        reference = np.random.default_rng(104).integers(20, 230, (100, 140), dtype=np.uint8)
        points = np.float32([[40, 30], [90, 30], [90, 65], [40, 65]])
        for observed in [np.full_like(reference, 90), np.random.default_rng(105).integers(20, 230, reference.shape, dtype=np.uint8)]:
            corrected, detail = refine_region(reference, observed, points)
            np.testing.assert_allclose(corrected, points)
            self.assertIsNone(detail)

    def test_bare_reference_and_missing_posts(self):
        report = self.engine.inspect(self.engine.bare)
        self.assertTrue(report["located"])
        self.assertEqual(len(report["items"]), 16)
        self.assertIn("A10", {item["id"] for item in report["items"]})
        self.assertTrue(all(item["status"] == "match" for item in report["items"][:11]))
        self.assertTrue(all(item["status"] in {"suspect", "review"} for item in report["items"][-4:]))

    def test_dark_reference_retains_board_localization(self):
        image = np.clip(self.engine.bare.astype(np.float32) * 0.18, 0, 255).astype(np.uint8)
        report = self.engine.inspect(image)
        self.assertTrue(report["located"])

    def test_changed_version_print_does_not_prevent_localization(self):
        image = self.engine.bare.copy()
        image[98:190, 449:744] = np.random.default_rng(128).integers(0, 256, (92, 295, 3), dtype=np.uint8)
        self.assertTrue(self.engine.inspect(image)["located"])

    def test_device_camera_frame_locates_without_ocr_or_calibration(self):
        image = cv2.imread(str(Path(__file__).parent / "testdata" / "m2-camera-localization.jpg"))
        self.assertIsNotNone(image)
        matrix, quality = self.engine.align(image)
        self.assertIsNotNone(matrix)
        self.assertGreaterEqual(quality["inliers"], 12)
        self.assertGreaterEqual(quality["ratio"], 0.35)

    def test_approved_complete_board_survives_reload_and_small_movement(self):
        image = cv2.imread(str(Path(__file__).parent / "testdata" / "m2-camera-localization.jpg"))
        engine = InspectionEngine()
        engine.calibrated = True
        report = engine.inspect(image)
        with tempfile.TemporaryDirectory() as directory:
            decisions = CalibratedDecisions(directory)
            decisions.calibrate(image, report)
            reloaded = InspectionEngine()
            reloaded.calibrated = True
            self.assertTrue(reloaded.load_station_reference(directory))
            reloaded.ocr = Mock(return_value=([[[[0, 0]], "Quectel Pi M2", 0.99]], 0))
            decisions = CalibratedDecisions(directory)
            self.assertEqual(len(decisions.state()), 16)
            moved = cv2.warpAffine(image, np.float32([[1, 0, 6], [0, 1, 4]]), (image.shape[1], image.shape[0]))
            report = reloaded.inspect(moved)
            self.assertEqual(report["alignment"]["reference"], "approved_complete_board")
            for _ in range(3):
                result = decisions.apply(moved, report)
            self.assertEqual(result["verdict"], "检测通过")
            self.assertEqual(sum(item["status"] == "pass" for item in result["items"]), 16)
            blank = np.zeros_like(image)
            self.assertFalse(reloaded.inspect(blank)["located"])

    def test_calibrated_rotation_preserves_component_ids_and_passes(self):
        image = cv2.imread(str(Path(__file__).parent / "testdata" / "m2-camera-localization.jpg"))
        engine = InspectionEngine()
        engine.calibrated = True
        engine.ocr = Mock(return_value=([[[[0, 0]], "Quectel Pi M2", 0.99]], 0))
        with tempfile.TemporaryDirectory() as directory:
            decisions = CalibratedDecisions(directory)
            decisions.calibrate(image, engine.inspect(image))
            engine.load_station_reference(directory)
            for rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                rotated = cv2.rotate(image, rotation)
                report = engine.inspect(rotated)
                self.assertTrue(report["located"], (rotation, report["alignment"]))
                decisions.reset()
                for _ in range(3):
                    result = decisions.apply(rotated, report)
                self.assertEqual(sum(item["status"] == "pass" for item in result["items"]), 16, (rotation, [(item["id"], item["scores"]) for item in result["items"]]))

    def test_station_post_coordinates_are_not_replaced_by_old_photo_projection(self):
        image = cv2.imread(str(Path(__file__).parent / "testdata" / "m2-camera-localization.jpg"))
        engine = InspectionEngine()
        engine.calibrated = True
        report = engine.inspect(image)
        post = next(item for item in report["items"] if item["id"] == "B04")
        post["polygon"] = [[365, 342], [432, 342], [432, 414], [365, 414]]
        with tempfile.TemporaryDirectory() as directory:
            CalibratedDecisions(directory).calibrate(image, report)
            self.assertTrue(engine.load_station_reference(directory))
            actual = next(item for item in engine.inspect(image)["items"] if item["id"] == "B04")
            np.testing.assert_allclose(actual["polygon"], post["polygon"], atol=0.5)

    def test_station_interface_coordinates_are_not_replaced_by_old_photo_projection(self):
        image = cv2.imread(str(Path(__file__).parent / "testdata" / "m2-camera-localization.jpg"))
        engine = InspectionEngine()
        engine.calibrated = True
        report = engine.inspect(image)
        corrected = {"A01": [[140, 108], [395, 96], [397, 125], [139, 137]], "A04": [[300, 290], [337, 292], [335, 377], [297, 375]]}
        for item in report["items"]:
            if item["id"] in corrected:
                item["polygon"] = corrected[item["id"]]
        with tempfile.TemporaryDirectory() as directory:
            CalibratedDecisions(directory).calibrate(image, report)
            self.assertTrue(engine.load_station_reference(directory))
            actual = {item["id"]: item for item in engine.inspect(image)["items"]}
            for identity, points in corrected.items():
                np.testing.assert_allclose(actual[identity]["polygon"], points, atol=0.5)

    def test_installed_reference_and_transformed_image(self):
        for image in [self.engine.posts, cv2.resize(self.engine.posts, (1152, 864))]:
            report = self.engine.inspect(image)
            self.assertTrue(report["located"])
            self.assertTrue(all(item["status"] == "match" for item in report["items"][-4:]))
            self.assertEqual(report["verdict"], "待复核")

    def test_blank_and_random_images_never_pass(self):
        rng = np.random.default_rng(42)
        for image in [np.zeros((720, 1280, 3), np.uint8), rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)]:
            report = self.engine.inspect(image)
            self.assertFalse(report["located"])
            self.assertTrue(all(item["status"] == "review" for item in report["items"]))

    def test_changed_region_requires_review(self):
        image = self.engine.bare.copy()
        image[310:530, 1013:1278] = 30
        report = self.engine.inspect(image)
        usb = next(item for item in report["items"] if item["id"] == "A06")
        self.assertEqual(usb["status"], "review")

    def test_post_region_change_is_not_confirmed_installed(self):
        image = self.engine.posts.copy()
        image[263:419, 94:286] = 30
        report = self.engine.inspect(image)
        post = next(item for item in report["items"] if item["id"] == "B01")
        self.assertNotEqual(post["status"], "match")

    def test_version_never_fills_in_reference_text(self):
        report = self.engine.inspect(self.engine.posts)
        self.assertEqual(report["version"]["text"], "未识别")
        self.assertEqual(report["version"]["status"], "review")

    def test_uploaded_image_resized_and_invalid_rejected(self):
        buffer = io.BytesIO()
        Image.new("RGB", (3000, 1800)).save(buffer, format="JPEG")
        image = decode_image(buffer.getvalue())
        self.assertEqual(image.shape, (1200, 2000, 3))
        with self.assertRaises(Exception):
            decode_image(b"not an image")

    def test_identity_comes_from_ocr_not_fixed_model(self):
        engine = InspectionEngine()
        engine.ocr = Mock(return_value=([[[], "Acme Board X7", 0.98], [[], "ABC123_V2.3", 0.96]], None))
        result = engine.read_version(engine.bare)
        self.assertEqual(result["device_name"], "Acme Board X7")
        self.assertEqual(result["board_revision"], "")
        self.assertEqual(result["model"], "X7")
        self.assertEqual(result["status"], "match")

    def test_ocr_throttled_and_confirmed_model_survives_location_change(self):
        engine = InspectionEngine()
        engine.ocr_interval = 3
        engine.ocr = Mock(return_value=([[[], "Quectel Pi M2", 0.98]], None))
        engine.current_version(engine.bare, np.eye(3))
        engine.current_version(engine.bare, np.eye(3))
        self.assertEqual(engine.ocr.call_count, 1)
        result = engine.current_version(engine.bare, None)
        self.assertEqual(engine.ocr.call_count, 1)
        self.assertEqual(result["model"], "M2")
        self.assertEqual(result["status"], "match")
        engine.reset_identity()
        engine.current_version(engine.bare, None)
        self.assertEqual(engine.ocr.call_count, 2)

    def test_unlocated_ocr_reads_rotated_model_without_inventing_revision(self):
        engine = InspectionEngine()
        engine.ocr = Mock(side_effect=[(None, None), (None, None), ([[[], "M2", 0.96]], None)])
        first = engine.read_version(np.zeros((480, 640, 3), np.uint8), full_frame=True)
        self.assertEqual(first["model"], "")
        self.assertEqual(engine.ocr.call_count, 2)
        result = engine.read_version(np.zeros((480, 640, 3), np.uint8), full_frame=True)
        self.assertEqual(result["device_name"], "M2")
        self.assertEqual(result["board_revision"], "")
        self.assertEqual(result["ocr_rotation"], 90)
        self.assertEqual(engine.ocr.call_count, 3)
        self.assertEqual(result["status"], "match")

    def test_model_only_ocr_accepts_m2_and_distinguishes_h1_l1(self):
        engine = InspectionEngine()
        for text, model in [("M2", "M2"), ("H1", "H1"), ("L1", "L1"), ("Quectel Pi M2", "M2"), ("Quectel Pi H1", "H1"), ("Quectel Pi L1", "L1"), ("QuectelPiM2", "M2"), ("QuectelPiH1", "H1"), ("QuectelPiL1", "L1")]:
            engine.ocr = Mock(return_value=([[text, 0.98], ["QSM610ZAP V1.2 USB GPIO", 0.97]], None))
            result = engine.read_version(engine.bare)
            self.assertEqual(result["model"], model)
            self.assertEqual(result["text"], text)
            self.assertEqual(result["status"], "match")
            self.assertEqual(result["board_revision"], "")
            self.assertEqual(engine.ocr.call_count, 1)
            self.assertFalse(engine.ocr.call_args.kwargs["use_det"])

    def test_model_line_does_not_expand_or_fall_back_to_full_detector(self):
        engine = InspectionEngine()
        engine.ocr = Mock(return_value=(None, None))
        engine.read_version(engine.bare)
        self.assertEqual(engine.ocr.call_count, 2)
        left, top, right, bottom = engine.model_crop(engine.bare)
        np.testing.assert_array_equal(engine.ocr.call_args_list[0].args[0], engine.bare[top:bottom, left:right])
        np.testing.assert_array_equal(engine.ocr.call_args.args[0], enhance_text(engine.bare[top:bottom, left:right]))
        self.assertFalse(engine.ocr.call_args.kwargs["use_det"])

    def test_enhanced_dual_model_m2_needs_two_tasks_and_rejects_disagreement(self):
        engine = InspectionEngine()
        engine.ocr = Mock(side_effect=[([["02", 0.45]], None), ([["M2", 0.59]], None)] * 2)
        engine.latin_ocr = Mock(return_value=([["M2", 0.52]], None))
        first = engine.current_version(engine.bare, np.eye(3))
        self.assertEqual(first["model"], "")
        self.assertEqual(first["ocr_preprocessing"], "dual_model_enhanced")
        self.assertEqual(engine.current_version(engine.bare, np.eye(3))["model"], "M2")
        engine.reset_identity()
        engine.ocr = Mock(side_effect=[([["02", 0.45]], None), ([["M2", 0.59]], None)] * 3)
        engine.latin_ocr = Mock(return_value=([["MZ", 0.98]], None))
        for _ in range(3):
            self.assertEqual(engine.current_version(engine.bare, np.eye(3))["model"], "")
        for token in [("M2", 0.49), ("H1", 0.98), ("2", 0.99)]:
            engine.reset_identity()
            engine.ocr = Mock(side_effect=[([["02", 0.45]], None), ([["M2", 0.59]], None)] * 2)
            engine.latin_ocr = Mock(return_value=([[*token]], None))
            for _ in range(2):
                self.assertEqual(engine.current_version(engine.bare, np.eye(3))["model"], "")

    def test_full_frame_enhancement_checks_english_only_inside_detected_word(self):
        engine = InspectionEngine()
        box = [[10, 8], [30, 8], [30, 20], [10, 20]]
        engine.ocr = Mock(side_effect=[(None, None), ([[box, "M2", 0.59]], None)])
        engine.latin_ocr = Mock(return_value=([["M2", 0.52]], None))
        result = engine.read_version(np.zeros((480, 640, 3), np.uint8), full_frame=True)
        self.assertEqual(result["candidate_model"], "M2")
        self.assertEqual(result["model"], "")
        self.assertEqual(engine.latin_ocr.call_args.args[0].shape, (12, 20, 3))
        self.assertFalse(engine.latin_ocr.call_args.kwargs["use_det"])

    def test_model_crop_follows_print_offset_without_full_text_detection(self):
        engine = InspectionEngine()
        image = engine.bare.copy()
        image[80:205, 510:745] = 50
        image[128:170, 620:678] = engine.bare[103:145, 590:648]
        engine.ocr = Mock(return_value=([["M2", 0.99]], None))
        result = engine.read_version(image)
        self.assertEqual(result["model"], "M2")
        self.assertEqual(result["bounds"], (626, 120, 688, 178))
        self.assertFalse(engine.ocr.call_args.kwargs["use_det"])

    def test_medium_confidence_m2_needs_two_complete_reads_and_reset_discards_vote(self):
        engine = InspectionEngine()
        engine.ocr = Mock(return_value=([["M2", 0.75]], None))
        self.assertEqual(engine.current_version(engine.bare, np.eye(3))["model"], "")
        self.assertEqual(engine.current_version(engine.bare, np.eye(3))["model"], "M2")
        engine.reset_identity()
        self.assertEqual(engine.current_version(engine.bare, np.eye(3))["model"], "")
        for text, score in [("2", 0.99), ("MZ", 0.98), ("M2", 0.60)]:
            engine.reset_identity()
            engine.ocr = Mock(return_value=([[text, score]], None))
            for _ in range(3):
                self.assertEqual(engine.current_version(engine.bare, np.eye(3))["model"], "")

    def test_three_seconds_without_target_clears_identity_on_any_background(self):
        engine = InspectionEngine()
        image = np.random.default_rng(92).integers(20, 240, (120, 160, 3), dtype=np.uint8)
        blurred = cv2.GaussianBlur(image, (21, 21), 0)
        for frame in [np.zeros_like(image), image, blurred, cv2.GaussianBlur(image, (61, 61), 0), np.full_like(image, round(float(image.mean())))]:
            self.assertEqual(engine.board_presence(image, True, 0), "present")
            engine.ocr_cached = {"model": "M2"}
            for now in [0.1, 1, 2.9]:
                self.assertEqual(engine.board_presence(frame, False, now), "unknown")
                self.assertEqual(engine.ocr_cached["model"], "M2")
            self.assertEqual(engine.board_presence(frame, False, 3.2), "absent")
            self.assertIsNone(engine.ocr_cached)

    def test_target_reappearing_before_timeout_cancels_reset(self):
        engine = InspectionEngine()
        image = np.full((120, 160, 3), 100, np.uint8)
        engine.board_presence(image, True, 0)
        engine.ocr_cached = {"model": "M2"}
        self.assertEqual(engine.board_presence(image, False, 1), "unknown")
        self.assertEqual(engine.board_presence(image, True, 3), "present")
        self.assertEqual(engine.board_presence(image, False, 4), "unknown")
        self.assertEqual(engine.ocr_cached["model"], "M2")
        self.assertEqual(engine.board_presence(image, False, 7.1), "absent")

    def test_tracking_restores_geometry_without_repeating_model_ocr(self):
        engine = InspectionEngine()
        image = cv2.imread(str(Path(__file__).parent / "testdata" / "m2-camera-localization.jpg"))
        matrix, _ = engine.align(image)
        self.assertIsNotNone(matrix)
        engine.remember_target(image, matrix)
        tracked = engine.tracked_reference
        self.assertIsNotNone(tracked)
        engine.ocr_cached = {"model": "M2", "device_name": "M2"}
        engine.ocr = Mock()
        original_match = engine._match_features

        def only_current_target(keypoints, descriptors, features, image_shape, reference_shape=None, reference_outline=None):
            if features is tracked[1]:
                return original_match(keypoints, descriptors, features, image_shape, reference_shape, reference_outline)
            return None, {"matches": 0, "inliers": 0, "ratio": 0}

        moved = cv2.warpAffine(image, np.float32([[1, 0, 15], [0, 1, 8]]), (image.shape[1], image.shape[0]))
        with patch.object(engine, "_match_features", side_effect=only_current_target):
            recovered, quality = engine.align(moved)
        self.assertIsNotNone(recovered)
        self.assertEqual(quality["reference"], "tracked_board")
        self.assertGreaterEqual(quality["inliers"], 12)
        engine.current_version(moved, recovered)
        engine.ocr.assert_not_called()
        engine.reset_identity()
        self.assertIsNone(engine.tracked_reference)

    def test_close_board_geometry_excludes_reference_photo_background(self):
        engine = InspectionEngine()
        reference_points = [cv2.KeyPoint(float(horizontal), float(vertical), 8) for vertical in [150, 190, 230, 270] for horizontal in [200, 250, 300, 350, 400]]
        observed_points = [cv2.KeyPoint((point.pt[0] - 200) * 2 + 80, (point.pt[1] - 150) * 2 + 50, 8) for point in reference_points]
        descriptors = np.random.default_rng(815).random((20, 128)).astype(np.float32)
        features = (reference_points, descriptors)
        matrix, quality = engine._match_features(observed_points, descriptors, features, (480, 640, 3), (480, 640, 3))
        self.assertIsNone(matrix)
        self.assertEqual(quality["inliers"], 20)
        outline = np.float32([[190, 140], [410, 140], [410, 280], [190, 280]])
        matrix, quality = engine._match_features(observed_points, descriptors, features, (480, 640, 3), (480, 640, 3), outline)
        self.assertIsNotNone(matrix)
        self.assertEqual(quality["inliers"], 20)
        self.assertEqual(quality["ratio"], 1)

    def test_six_consistent_features_keep_identity_but_cannot_locate_components(self):
        engine = InspectionEngine()
        points = [cv2.KeyPoint(float(horizontal), float(vertical), 8) for vertical in [40, 350] for horizontal in [40, 300, 580]]
        descriptors = np.random.default_rng(814).random((6, 128)).astype(np.float32)
        matrix, quality = engine._match_features(points, descriptors, (points, descriptors), (480, 640, 3), (480, 640, 3))
        self.assertIsNone(matrix)
        self.assertEqual(quality["inliers"], 6)
        self.assertTrue(engine.target_visible)
        image = np.full((120, 160, 3), 100, np.uint8)
        engine.ocr_cached = {"model": "M2"}
        engine.board_presence(image, True, 0)
        for now in [1, 4, 8]:
            self.assertEqual(engine.board_presence(image, False, now, target_visible=True), "present")
            self.assertEqual(engine.ocr_cached["model"], "M2")
        self.assertEqual(engine.board_presence(image, False, 9), "unknown")
        self.assertEqual(engine.board_presence(image, False, 12.1), "absent")
        self.assertIsNone(engine.ocr_cached)

    def test_medium_confidence_brand_m2_retries_same_orientation_and_confirms(self):
        engine = InspectionEngine()
        engine.ocr = Mock(return_value=([[[], "QuectelPiM2", 0.78]], None))
        image = np.zeros((480, 640, 3), np.uint8)
        first = engine.confirm_version(engine.read_version(image, full_frame=True))
        self.assertEqual(first["model"], "")
        self.assertEqual(first["candidate_model"], "M2")
        second = engine.confirm_version(engine.read_version(image, full_frame=True))
        self.assertEqual(second["model"], "M2")
        self.assertEqual(second["ocr_rotation"], first["ocr_rotation"])

    def test_slow_ocr_does_not_block_live_detection(self):
        engine = InspectionEngine()
        engine.ocr_interval = 3
        started = threading.Event()
        release = threading.Event()

        def delayed_ocr(*args, **kwargs):
            started.set()
            release.wait(2)
            return {"status": "match", "text": "Acme Board X7", "device_name": "Acme Board X7", "board_revision": "V2.3", "raw": []}

        engine.read_version = delayed_ocr
        engine.start_ocr_worker()
        try:
            before = time.monotonic()
            result = engine.current_version(engine.bare, np.eye(3))
            self.assertLess(time.monotonic() - before, 0.5)
            self.assertEqual(result["device_name"], "")
            self.assertTrue(started.wait(1))
            release.set()
            with engine.ocr_condition:
                self.assertTrue(engine.ocr_condition.wait_for(lambda: engine.ocr_cached is not None, timeout=1))
            self.assertEqual(engine.current_version(engine.bare, np.eye(3))["device_name"], "Acme Board X7")
            self.assertEqual(engine.current_version(engine.bare, None)["device_name"], "Acme Board X7")
        finally:
            release.set()
            engine.close_ocr_worker()


class HttpApiTests(unittest.TestCase):
    def test_command_line_starts_from_unrelated_directory_and_reuses_calibration(self):
        data_directory = Path(self.temporary.name) / "fresh-data"
        for iteration in range(2):
            process = subprocess.Popen([sys.executable, str(server.ROOT / "server.py"), "--image-only", "--port", "0", "--data-dir", str(data_directory)], cwd=self.temporary.name, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
            try:
                deadline = time.monotonic() + 60
                base = None
                while time.monotonic() < deadline:
                    ready, _, _ = select.select([process.stdout], [], [], max(0, deadline - time.monotonic()))
                    if not ready:
                        break
                    line = process.stdout.readline().decode("utf-8", errors="replace")
                    if not line:
                        break
                    match = re.search(r"Board inspection: (http://127\.0\.0\.1:\d+)", line)
                    if match:
                        base = match.group(1)
                        break
                self.assertIsNotNone(base, "Command-line service did not become ready")
                with urlopen(base + "/api/config", timeout=10) as response:
                    config = json.load(response)
                self.assertTrue(config["ocr_ready"])
                self.assertIsNone(config["device_camera"])
                self.assertEqual(len(config["checkpoints"]), 12)
                with urlopen(base + "/api/production", timeout=10) as response:
                    self.assertEqual(json.load(response)["total"], 0)
                marker = data_directory / "calibration/operator-marker.txt"
                if iteration == 0:
                    marker.write_text("retain operator changes")
                else:
                    self.assertEqual(marker.read_text(), "retain operator changes")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                process.stdout.close()

    def test_audio_announces_only_once_and_switch_is_validated(self):
        announcer = server.ResultAnnouncer(self.temporary.name)
        announcer.available = True
        self.http.announcer = announcer
        with patch("server.subprocess.run") as playback:
            announcer.announce("first", "pass")
            announcer.thread.join(timeout=2)
            announcer.announce("first", "fail")
            self.assertEqual(playback.call_count, 1)
            self.assertTrue(str(playback.call_args.args[0][-1]).endswith("qualified.wav"))
            self.assertIn("--volume=81920", playback.call_args.args[0])
            announcer.announce("second", "fail")
            announcer.thread.join(timeout=2)
            self.assertTrue(str(playback.call_args.args[0][-1]).endswith("rejected.wav"))
            self.assertIn("--volume=65536", playback.call_args.args[0])
            self.assertEqual(self.post("/api/audio/control", b'{"enabled":"yes"}')[0], 400)
            self.assertEqual(self.post("/api/audio/control", b'{"enabled":false}')[0], 200)
            announcer.announce("third", "pass")
            self.assertEqual(playback.call_count, 2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.output_patch = patch.object(server, "OUTPUT", Path(self.temporary.name))
        self.output_patch.start()
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        self.http.engine = InspectionEngine()
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.http.server_port}"

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()
        self.output_patch.stop()
        self.temporary.cleanup()

    def post(self, route, payload, headers=None):
        request = Request(self.base + route, data=payload, headers=headers or {}, method="POST")
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def test_inspection_review_history_and_attachment(self):
        _, encoded = cv2.imencode(".png", self.http.engine.posts)
        status, report = self.post("/api/inspect", encoded.tobytes())
        self.assertEqual(status, 200)
        self.assertEqual(len(report["items"]), 16)
        payload = {"id": report["id"], "checkpoint": "B01", "verdict": "异常"}
        status, updated = self.post("/api/review", json.dumps(payload).encode())
        self.assertEqual(status, 200)
        self.assertEqual(updated["verdict"], "人工标记异常")
        with urlopen(self.base + report["report_url"] + "?download=1") as response:
            self.assertIn("attachment", response.headers["Content-Disposition"])
            saved = json.loads(response.read())
            self.assertEqual(saved["manual_reviews"]["B01"]["verdict"], "异常")
        with urlopen(self.base + "/api/history") as response:
            self.assertEqual(json.loads(response.read())[0]["verdict"], "人工标记异常")
        payload["verdict"] = None
        _, cleared = self.post("/api/review", json.dumps(payload).encode())
        self.assertEqual(cleared["verdict"], "待复核")

    def test_invalid_images_wired_mode_and_invalid_review_rejected(self):
        self.assertEqual(self.post("/api/inspect", b"not an image")[0], 400)
        self.assertEqual(self.post("/api/inspect", b"image", {"X-Inspection-Mode": "wired"})[0], 409)
        self.assertEqual(self.post("/api/review", b'{"id":"../../secret"}')[0], 400)

    def test_table_export_is_downloadable_escaped_and_checkpoint_scoped(self):
        report = {"id": "table-test", "inspection_scope": "checkpoint", "version": {"model": "M2"}, "verdict": "无法检测", "items": [{"id": "A01", "name": "40pin<script>bad()</script>", "status": "unavailable", "observed_status": "unavailable", "scores": {"pass": 0.72}, "reason": "未确认"}]}
        (Path(self.temporary.name) / "table-test.json").write_text(json.dumps(report), encoding="utf-8")
        with urlopen(self.base + "/output/table-test.html?download=1") as response:
            text = response.read().decode("utf-8")
            self.assertIn("attachment", response.headers["Content-Disposition"])
            self.assertIn("text/html", response.headers["Content-Type"])
            self.assertIn("单模块复检", text)
            self.assertIn("72.0%", text)
            self.assertIn("&lt;script&gt;", text)
            self.assertNotIn("<script>", text)
            self.assertNotIn("A06", text)
        with self.assertRaises(HTTPError):
            urlopen(self.base + "/output/not-found.html")

    def test_production_ledger_counts_cycles_once_and_preserves_first_pass_rate(self):
        path = Path(self.temporary.name) / "production.sqlite3"
        ledger = server.ProductionLedger(path)
        self.http.ledger = ledger
        try:
            self.assertIsNone(ledger.summary()["pass_rate"])
            self.assertIsNone(ledger.last_completed())
            for identity, outcome, status in [("one", "pass", "pass"), ("two", "fail", "unavailable")]:
                report = {"id": identity, "cycle_id": identity, "inspection_scope": "board_cycle", "cycle_complete": True, "outcome": outcome, "items": [{"id": "A01", "name": "40pin<script>", "status": status}]}
                first = ledger.record(report)
                self.assertEqual(first["board_number"], ledger.record(report)["board_number"])
            ledger.record_recheck({"id": "retry", "parent_report_id": "two", "inspection_scope": "checkpoint", "items": [{"id": "A01", "name": "40pin<script>", "status": "pass"}]})
            summary = ledger.summary()
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["first_pass_rate"], 0.5)
            self.assertEqual(summary["pass_rate"], 1)
            self.assertEqual(summary["boards"][1]["initial_outcome"], "fail")
            self.assertEqual(ledger.last_completed()["id"], "two")
            self.assertEqual(ledger.last_completed()["outcome"], "fail")
            with self.assertRaises(ValueError):
                ledger.record({"inspection_scope": "checkpoint"})
            with urlopen(self.base + "/api/production/export.html") as response:
                text = response.read().decode()
                self.assertIn("attachment", response.headers["Content-Disposition"])
                self.assertIn("50.0%", text)
                self.assertIn("100.0%", text)
                self.assertIn("逐板检测结果", text)
            ledger.db.close()
            reopened = server.ProductionLedger(path)
            self.assertEqual(reopened.summary()["total"], 2)
            reopened.db.close()
        finally:
            if getattr(self.http, "ledger", None):
                del self.http.ledger

    def test_live_operations_require_device_camera(self):
        for route in ["/api/live/control", "/api/live/reset", "/api/live/recheck", "/api/calibrate", "/api/live/save"]:
            self.assertEqual(self.post(route, b'{}')[0], 409)

    def test_checkpoint_recheck_persists_only_selected_module_and_bound_frame(self):
        image = np.full((120, 160, 3), 90, np.uint8)
        report = {"inspection_scope": "checkpoint", "sequence": 13, "items": [{"id": "A01", "name": "GPIO", "polygon": [[0, 0], [159, 0], [159, 119], [0, 119]], "status": "unavailable"}], "version": {"model": "M2"}, "verdict": "无法检测", "live": True, "recheck": {"checkpoint": "A01", "requested_after_sequence": 10, "sequences": [11, 12, 13]}}
        live = Mock()
        live.recheck.return_value = (image, report)
        self.http.live = live
        self.assertEqual(self.post("/api/live/recheck", b'{"checkpoint":[]}')[0], 400)
        self.assertEqual(self.post("/api/live/recheck", b'{"checkpoint":"A01","sequence":1}')[0], 400)
        status, saved = self.post("/api/live/recheck", b'{"checkpoint":"A01"}')
        self.assertEqual(status, 200)
        live.recheck.assert_called_once_with("A01")
        self.assertEqual(saved["inspection_scope"], "checkpoint")
        self.assertEqual(saved["recheck"]["sequences"], [11, 12, 13])
        self.assertEqual([item["id"] for item in saved["items"]], ["A01"])
        self.assertEqual(saved["verdict"], "GPIO复验：无法检测")
        self.assertTrue((Path(self.temporary.name) / f"{saved['id']}.jpg").is_file())
        self.assertFalse(saved["live"])

    def test_live_frame_control_save_and_calibration_validation(self):
        image = np.zeros((120, 160, 3), np.uint8)
        _, encoded = cv2.imencode(".jpg", image)
        report = {"sequence": 7, "live": True, "located": False, "items": [], "version": {"text": "未识别"}, "verdict": "无法检测"}
        live = Mock()
        live.camera.source = "/dev/test"
        live.decisions = server.CalibratedDecisions(Path(self.temporary.name) / "calibration")
        live.snapshot.side_effect = lambda sequence: (7, encoded.tobytes(), image, report) if sequence == 7 else None
        live.status.return_value = {"enabled": True, "error": "", "report": report}
        self.http.live = live
        with urlopen(self.base + "/api/live/frame.jpg?sequence=7") as response:
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            self.assertEqual(response.read(), encoded.tobytes())
        with self.assertRaises(HTTPError) as expired:
            urlopen(self.base + "/api/live/frame.jpg?sequence=6")
        self.assertEqual(expired.exception.code, 410)
        self.assertEqual(self.post("/api/live/control", b'{"enabled":"false"}')[0], 400)
        self.assertEqual(self.post("/api/live/control", b'{"enabled":false}')[0], 200)
        self.assertFalse(live.enabled)
        live.reset_session.return_value = {"reset": True, "required_model": "M2"}
        self.assertEqual(self.post("/api/live/reset", b'{"reset":false}')[0], 400)
        self.assertEqual(self.post("/api/live/reset", b'{"reset":true}')[0], 200)
        live.reset_session.assert_called_once()
        self.assertEqual(self.post("/api/live/save", b'{"sequence":6}')[0], 400)
        status, saved = self.post("/api/live/save", b'{"sequence":7}')
        self.assertEqual(status, 200)
        self.assertFalse(saved["live"])
        self.assertEqual(saved["sequence"], 7)
        self.assertTrue((Path(self.temporary.name) / f"{saved['id']}.json").is_file())
        self.assertTrue(report["live"])
        status, rejected = self.post("/api/calibrate", b'{"sequence":7,"label":"pass"}')
        self.assertEqual(status, 400)
        self.assertIn("不能标定", rejected["error"])


if __name__ == "__main__":
    unittest.main()