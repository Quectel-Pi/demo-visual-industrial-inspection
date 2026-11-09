import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import cv2

from decisions import CalibratedDecisions, align_component, crop_item, copper_post_evidence, gpio_evidence


class DecisionTests(unittest.TestCase):
    def test_usb_and_vrtc_two_fresh_positives_survive_brief_localization_loss(self):
        for checkpoint in ("A06", "A12"):
            with self.subTest(checkpoint=checkpoint):
                report = {**self.report, "items": [{**self.report["items"][0], "id": checkpoint}]}
                self.decisions.calibrate(self.normal, report, checkpoint=checkpoint)
                self.decisions.clock = Mock(return_value=10.0)
                first = self.decisions.apply(self.normal, report)
                self.assertEqual(first["items"][0]["status"], "unavailable")
                self.decisions.clock.return_value = 10.5
                self.decisions.apply(self.normal, {**report, "located": False})
                self.decisions.clock.return_value = 11.0
                result = self.decisions.apply(self.normal, report)
                self.assertEqual(result["items"][0]["status"], "pass")
                self.assertEqual(result["items"][0]["observed_status"], "pass")

    def test_usb_and_vrtc_expired_reset_or_defect_evidence_cannot_reuse_old_confirmation(self):
        for checkpoint in ("A06", "A12"):
            with self.subTest(checkpoint=checkpoint):
                report = {**self.report, "items": [{**self.report["items"][0], "id": checkpoint}]}
                self.decisions.calibrate(self.normal, report, checkpoint=checkpoint)
                self.decisions.calibrate(self.absent, report, "missing", checkpoint)
                self.decisions.clock = Mock(return_value=10.0)
                self.decisions.apply(self.normal, report)
                self.decisions.clock.return_value = 13.1
                result = self.decisions.apply(self.normal, report)
                self.assertEqual(result["items"][0]["status"], "unavailable")
                self.decisions.reset_checkpoint(checkpoint)
                result = self.decisions.apply(self.normal, report)
                self.assertEqual(result["items"][0]["status"], "unavailable")
                self.decisions.apply(self.absent, report)
                self.assertFalse(self.decisions.window_confirmations[checkpoint])
                result = self.decisions.apply(self.normal, report)
                self.assertEqual(result["items"][0]["status"], "unavailable")

    def test_usb_and_vrtc_confirmation_evidence_is_independent(self):
        reports = {checkpoint: {**self.report, "items": [{**self.report["items"][0], "id": checkpoint}]} for checkpoint in ("A06", "A12")}
        for checkpoint, report in reports.items():
            self.decisions.calibrate(self.normal, report, checkpoint=checkpoint)
        self.decisions.clock = Mock(return_value=10.0)
        for report in reports.values():
            result = self.decisions.apply(self.normal, report)
            self.assertEqual(result["items"][0]["status"], "unavailable")
        self.decisions.reset_checkpoint("A06")
        self.assertFalse(self.decisions.window_confirmations["A06"])
        self.assertEqual(len(self.decisions.window_confirmations["A12"]), 1)
        result = self.decisions.apply(self.normal, reports["A06"])
        self.assertEqual(result["items"][0]["status"], "unavailable")
        result = self.decisions.apply(self.normal, reports["A12"])
        self.assertEqual(result["items"][0]["status"], "pass")
        self.decisions.reset()
        self.assertTrue(all(not evidence for evidence in self.decisions.window_confirmations.values()))

    def test_lower_presence_threshold_accepts_68_percent_but_not_62_percent(self):
        self.decisions.calibrate(self.normal, self.report, checkpoint="A01")
        with patch("decisions.gpio_evidence", return_value={"present": False}):
            with patch("decisions.local_similarity", return_value=0.68):
                for _ in range(3):
                    result = self.decisions.apply(self.normal, self.report)
                self.assertEqual(result["items"][0]["status"], "pass")
                self.assertEqual(result["items"][0]["pass_threshold"], 0.65)
            self.decisions.reset()
            with patch("decisions.local_similarity", return_value=0.62):
                for _ in range(3):
                    result = self.decisions.apply(self.normal, self.report)
                self.assertEqual(result["items"][0]["status"], "unavailable")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.decisions = CalibratedDecisions(self.temporary.name, confirm_frames=3, pass_hold_seconds=2.0)
        self.report = {"located": True, "version": {"device_name": "Quectel Pi M2"}, "items": [{"id": "A01", "name": "接口", "polygon": [[0, 0], [159, 0], [159, 119], [0, 119]]}]}
        rng = np.random.default_rng(42)
        self.normal = rng.integers(30, 240, (120, 160, 3), dtype=np.uint8)
        self.absent = rng.integers(30, 240, (120, 160, 3), dtype=np.uint8)

    def tearDown(self):
        self.temporary.cleanup()

    def test_no_calibration_no_fake_pass(self):
        result = self.decisions.apply(self.normal, self.report)
        self.assertEqual(result["items"][0]["status"], "unavailable")
        self.assertEqual(result["blocking_reason"], "缺少完整板标准")

    def test_poe_sample_is_loaded_and_offered_for_recheck(self):
        cv2.imwrite(str(self.decisions.directory / "A10_pass_00.png"), self.normal)
        self.assertIn("A10", CalibratedDecisions(self.temporary.name).state())

    def test_checkpoint_reset_clears_only_selected_evidence(self):
        self.decisions.votes["A01"].append("pass")
        self.decisions.votes["A02"].append("pass")
        self.decisions.last_pass.update(A01=10.0, A02=11.0)
        self.decisions.model_confirmed = True
        self.decisions.confirmed_identity = {"model": "M2"}
        self.decisions.reset_checkpoint("A01")
        self.assertNotIn("A01", self.decisions.votes)
        self.assertNotIn("A01", self.decisions.last_pass)
        self.assertEqual(list(self.decisions.votes["A02"]), ["pass"])
        self.assertEqual(self.decisions.last_pass["A02"], 11.0)
        self.assertTrue(self.decisions.model_confirmed)
        self.assertEqual(self.decisions.confirmed_identity, {"model": "M2"})

    def test_component_alignment_recovers_small_parallax_but_not_unrelated_patch(self):
        random = np.random.default_rng(108)
        reference = random.integers(30, 230, (180, 240, 3), dtype=np.uint8)
        item = {"polygon": [[80, 60], [160, 60], [160, 120], [80, 120]]}
        sample, _ = crop_item(reference, item)
        observed = np.full_like(reference, 90)
        observed[54:115, 84:165] = reference[60:121, 80:161]
        aligned = align_component(observed, item, [sample], 0.3)
        self.assertIsNotNone(aligned)
        np.testing.assert_allclose(aligned[0], np.float32(item["polygon"]) + [4, -6], atol=1)
        self.assertEqual(aligned[1]["method"], "bounded_component_template")
        for negative in [np.full_like(reference, 90), random.integers(30, 230, reference.shape, dtype=np.uint8)]:
            self.assertIsNone(align_component(negative, item, [sample], 0.3))

    def test_component_alignment_cannot_override_calibrated_defect(self):
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        self.decisions.calibrate(self.absent, self.report, "missing", "A01")
        with patch("decisions.align_component") as align, patch("decisions.gpio_evidence") as gpio:
            for _ in range(3):
                result = self.decisions.apply(self.absent, self.report)
        align.assert_not_called()
        gpio.assert_not_called()
        self.assertEqual(result["items"][0]["observed_status"], "missing")
        self.assertEqual(result["items"][0]["status"], "missing")

    def test_gpio_gradient_handles_lighting_but_rejects_occlusion_and_unrelated(self):
        normal = np.full((120, 160, 3), 35, np.uint8)
        for column in range(20):
            horizontal = 5 + column * 8
            cv2.rectangle(normal, (horizontal, 15), (horizontal + 2, 105), (125, 125, 125), -1)
            for vertical in [30, 75]:
                cv2.circle(normal, (horizontal + 1, vertical), 2, (240, 240, 240), -1)
        changed = np.clip(normal.astype(np.float32) * np.linspace(0.5, 1.5, 160)[None, :, None] + 15, 0, 255).astype(np.uint8)
        self.assertTrue(gpio_evidence(changed, [normal])["present"])
        negatives = [np.zeros_like(normal), self.normal, self.absent]
        for start, end in [(0, 80), (80, 160), (60, 100)]:
            masked = changed.copy()
            masked[:, start:end] = 35
            negatives.append(masked)
        for negative in negatives:
            self.assertFalse(gpio_evidence(negative, [normal])["present"])

    def test_copper_head_requires_metal_ring_and_dark_hole(self):
        item = {**self.report["items"][0], "id": "B01"}
        positive = np.full((120, 160, 3), (60, 35, 20), np.uint8)
        cv2.rectangle(positive, (67, 45), (91, 100), (55, 100, 145), -1)
        cv2.ellipse(positive, (80, 45), (23, 18), 0, 0, 360, (70, 155, 210), -1)
        cv2.ellipse(positive, (80, 45), (10, 8), 0, 0, 360, (15, 15, 15), -1)
        self.assertTrue(copper_post_evidence(positive, item)["present"])
        solid = positive.copy()
        cv2.ellipse(solid, (80, 45), (12, 10), 0, 0, 360, (70, 155, 210), -1)
        for negative in [solid, np.full_like(positive, (70, 155, 210)), np.zeros_like(positive), self.normal, self.absent]:
            self.assertFalse(copper_post_evidence(negative, item)["present"])

    def test_copper_evidence_needs_standard_and_three_frames_but_never_overrides_defect(self):
        report = {**self.report, "items": [{**self.report["items"][0], "id": "B01"}]}
        with patch("decisions.copper_post_evidence", return_value={"present": True}) as evidence:
            result = self.decisions.apply(self.absent, report)
            evidence.assert_not_called()
            self.assertEqual(result["items"][0]["status"], "unavailable")
        self.decisions.calibrate(self.normal, report, "pass", "B01")
        with patch("decisions.copper_post_evidence", return_value={"present": True}):
            for index in range(3):
                result = self.decisions.apply(self.absent, report)
                self.assertEqual(result["items"][0]["status"], "pass" if index == 2 else "unavailable")
                self.assertLess(result["items"][0]["scores"]["pass"], 0.82)
        self.decisions.calibrate(self.absent, report, "missing", "B01")
        with patch("decisions.copper_post_evidence", return_value={"present": True}) as evidence:
            for _ in range(3):
                result = self.decisions.apply(self.absent, report)
            evidence.assert_not_called()
            self.assertEqual(result["items"][0]["status"], "missing")

    def test_m2_model_is_required_but_revision_does_not_gate_inspection(self):
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        self.report["version"] = {"text": "QSM610ZAP_V1.1", "status": "match"}
        with patch("decisions.crop_item") as crop:
            result = self.decisions.apply(self.normal, self.report)
            crop.assert_not_called()
        self.assertEqual(result["model_gate"]["status"], "pending")
        for version in [{"device_name": "Quectel Pi M2", "board_revision": "QSM610ZAP_V1.1"}, {"device_name": "Quectel Pi M2", "board_revision": "QSM610ZAP_V2.9"}, {"text": "未识别", "status": "review"}, {}]:
            self.report["version"] = version
            for _ in range(3):
                result = self.decisions.apply(self.normal, self.report)
            self.assertEqual(result["verdict"], "检测通过")
            self.assertEqual(result["version_policy"], "require_m2_model")
            self.assertEqual(result["inspection_profile"], "m2-common")

    def test_recognized_m2_is_not_blocked_by_unrelated_ocr_numbers(self):
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        for version in [
            {"model": "M2", "raw": [{"text": "H1", "score": 0.99}, {"text": "L1", "score": 0.98}]},
            {"device_name": "Quectel Pi M2", "model": "X7", "raw": [{"text": "MODULE X7", "score": 0.99}, {"text": "H1", "score": 0.98}]},
        ]:
            self.decisions.reset()
            for _ in range(3):
                result = self.decisions.apply(self.normal, {**self.report, "version": version})
            self.assertEqual(result["model_gate"]["status"], "confirmed")
            self.assertEqual(result["items"][0]["status"], "pass")
        result = self.decisions.apply(self.normal, {**self.report, "version": {"model": "H1", "raw": [{"text": "M2", "score": 0.99}]}})
        self.assertEqual(result["model_gate"]["status"], "rejected")

    def test_default_session_locks_pass_across_motion_loss_and_long_gaps(self):
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        clock = Mock(return_value=10.0)
        decisions = CalibratedDecisions(self.temporary.name, clock=clock)
        for _ in range(3):
            decisions.apply(self.normal, self.report)
        clock.return_value = 120.0
        unlocated = {**self.report, "located": False, "version": {}, "items": [{**self.report["items"][0], "polygon": []}]}
        result = decisions.apply(np.zeros_like(self.normal), unlocated)
        self.assertEqual(result["items"][0]["status"], "pass")
        self.assertEqual(result["items"][0]["observed_status"], "unavailable")
        self.assertTrue(result["items"][0]["session_latched"])
        self.assertEqual(result["verdict"], "检测通过")
        self.assertEqual(result["model_gate"]["identity"]["device_name"], "Quectel Pi M2")
        self.assertEqual(result["items"][0]["polygon"], [])
        decisions.reset()
        result = decisions.apply(self.normal, {**self.report, "version": {}})
        self.assertEqual(result["model_gate"]["status"], "pending")
        self.assertEqual(result["items"][0]["status"], "unavailable")
        self.assertIsNone(result["model_gate"]["identity"])

    def test_non_m2_revokes_session_and_low_confidence_ocr_cannot_unlock(self):
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        decisions = CalibratedDecisions(self.temporary.name)
        for version in [{}, {"raw": [{"text": "M2", "score": 0.6}]}, {"device_name": "M20"}, {"device_name": "Quectel Pi M3"}, {"device_name": "Raspberry Pi"}]:
            result = decisions.apply(self.normal, {**self.report, "version": version})
            self.assertEqual(result["items"][0]["status"], "unavailable")
        for version in [{"raw": [{"text": "Quectel Pi M3", "score": 0.96}]}, {"device_name": "Quectel Pi H1"}, {"device_name": "Quectel Pi L1"}, {"device_name": "Acme Board X7"}, {"device_name": "Acme Board ZX"}, {"device_name": "NVIDIA Jetson Nano"}]:
            for _ in range(3):
                decisions.apply(self.normal, self.report)
            result = decisions.apply(self.normal, {**self.report, "version": version})
            self.assertEqual(result["model_gate"]["status"], "rejected")
            self.assertFalse(decisions.last_pass)
            self.assertIsNone(result["model_gate"]["identity"])

    def test_session_never_promotes_unseen_item_and_defect_revokes_latch(self):
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        self.decisions.calibrate(self.absent, self.report, "missing", "A01")
        decisions = CalibratedDecisions(self.temporary.name)
        result = decisions.apply(self.normal, self.report)
        self.assertEqual(result["items"][0]["status"], "unavailable")
        for _ in range(2):
            decisions.apply(self.normal, self.report)
        result = decisions.apply(self.absent, self.report)
        self.assertFalse(result["items"][0]["session_latched"])
        self.assertEqual(result["items"][0]["observed_status"], "missing")
        for _ in range(2):
            result = decisions.apply(self.absent, self.report)
        self.assertEqual(result["items"][0]["status"], "missing")

    def test_confirmed_board_removal_clears_identity_and_pass_but_tracking_loss_does_not(self):
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        decisions = CalibratedDecisions(self.temporary.name)
        for _ in range(3):
            decisions.apply(self.normal, self.report)
        lost = {**self.report, "located": False, "board_presence": "unknown"}
        result = decisions.apply(self.normal, lost)
        self.assertEqual(result["verdict"], "检测通过")
        removed = {**lost, "board_presence": "absent"}
        result = decisions.apply(self.normal, removed)
        self.assertEqual(result["items"][0]["status"], "unavailable")
        self.assertFalse(decisions.last_pass)
        self.assertFalse(decisions.model_confirmed)
        self.assertIsNone(result["model_gate"]["identity"])

    def test_confirmed_pass_survives_brief_ambiguous_frames_but_expires(self):
        clock = Mock(return_value=10.0)
        self.decisions.clock = clock
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        for _ in range(3):
            result = self.decisions.apply(self.normal, self.report)
        self.assertEqual(result["verdict"], "检测通过")
        with patch("decisions.local_similarity", return_value=0.62), patch("decisions.gpio_evidence", return_value={"present": False}):
            clock.return_value = 10.5
            result = self.decisions.apply(self.normal, self.report)
            self.assertEqual(result["items"][0]["status"], "pass")
            self.assertTrue(result["items"][0]["temporal_hold"])
            self.assertEqual(result["items"][0]["observed_status"], "unavailable")
            self.assertEqual(result["items"][0]["hold_remaining_ms"], 1500)
            clock.return_value = 11.9
            self.assertEqual(self.decisions.apply(self.normal, self.report)["held_count"], 1)
            clock.return_value = 12.0
            result = self.decisions.apply(self.normal, self.report)
            self.assertEqual(result["items"][0]["status"], "unavailable")
            self.assertFalse(result["items"][0]["temporal_hold"])

    def test_new_positive_evidence_refreshes_hold_without_reconfirm_flicker(self):
        clock = Mock(return_value=10.0)
        self.decisions.clock = clock
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        for _ in range(3):
            self.decisions.apply(self.normal, self.report)
        with patch("decisions.local_similarity", return_value=0.62), patch("decisions.gpio_evidence", return_value={"present": False}):
            clock.return_value = 10.5
            self.decisions.apply(self.normal, self.report)
        clock.return_value = 11.0
        self.assertEqual(self.decisions.apply(self.normal, self.report)["items"][0]["status"], "pass")
        with patch("decisions.local_similarity", return_value=0.62), patch("decisions.gpio_evidence", return_value={"present": False}):
            clock.return_value = 12.5
            self.assertEqual(self.decisions.apply(self.normal, self.report)["held_count"], 1)
            clock.return_value = 13.0
            self.assertEqual(self.decisions.apply(self.normal, self.report)["items"][0]["status"], "unavailable")

    def test_long_gap_requires_fresh_confirmation_and_recalibration_clears_hold(self):
        clock = Mock(return_value=10.0)
        self.decisions.clock = clock
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        for _ in range(3):
            self.decisions.apply(self.normal, self.report)
        clock.return_value = 13.0
        result = self.decisions.apply(self.normal, self.report)
        self.assertEqual(result["items"][0]["status"], "unavailable")
        self.assertFalse(result["items"][0]["temporal_hold"])
        for _ in range(2):
            result = self.decisions.apply(self.normal, self.report)
        self.assertEqual(result["items"][0]["status"], "pass")
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        self.assertFalse(self.decisions.last_pass)

    def test_first_positive_observation_is_not_promoted_by_hold(self):
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        result = self.decisions.apply(self.normal, self.report)
        self.assertEqual(result["items"][0]["status"], "unavailable")
        self.assertFalse(result["items"][0]["temporal_hold"])

    def test_no_hold_without_confirmed_pass_or_after_invalid_evidence(self):
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        with patch("decisions.local_similarity", return_value=0.75):
            self.assertEqual(self.decisions.apply(self.normal, self.report)["items"][0]["status"], "unavailable")
        unlocated = {**self.report, "located": False}
        out_of_frame = {**self.report, "items": [{**self.report["items"][0], "polygon": [[-80, 0], [79, 0], [79, 119], [-80, 119]]}]}
        for frame, report in [(np.zeros_like(self.normal), self.report), (self.normal, unlocated), (self.normal, out_of_frame), (self.absent, self.report)]:
            for _ in range(3):
                self.decisions.apply(self.normal, self.report)
            result = self.decisions.apply(frame, report)
            self.assertEqual(result["items"][0]["status"], "unavailable")
            self.assertFalse(result["items"][0]["temporal_hold"])

    def test_defect_evidence_cancels_pass_hold_before_defect_confirmation(self):
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        self.decisions.calibrate(self.absent, self.report, "missing", "A01")
        for _ in range(3):
            self.decisions.apply(self.normal, self.report)
        result = self.decisions.apply(self.absent, self.report)
        self.assertEqual(result["items"][0]["status"], "unavailable")
        self.assertFalse(result["items"][0]["temporal_hold"])
        for _ in range(2):
            result = self.decisions.apply(self.absent, self.report)
        self.assertEqual(result["items"][0]["status"], "missing")

    def test_calibrated_classes_need_three_frames_and_survive_reload(self):
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        self.decisions.calibrate(self.absent, self.report, "missing", "A01")
        result = self.decisions.apply(self.normal, self.report)
        self.assertEqual(result["items"][0]["status"], "unavailable")
        self.decisions.apply(self.normal, self.report)
        self.assertEqual(self.decisions.apply(self.normal, self.report)["verdict"], "检测通过")
        for _ in range(3):
            result = self.decisions.apply(self.absent, self.report)
        self.assertEqual(result["items"][0]["status"], "missing")
        reloaded = CalibratedDecisions(self.temporary.name)
        self.assertEqual(reloaded.state()["A01"]["missing"], 1)

    def test_missing_and_damage_are_not_interchangeable(self):
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        self.decisions.calibrate(self.absent, self.report, "damaged", "A01")
        for _ in range(3):
            result = self.decisions.apply(self.absent, self.report)
        self.assertEqual(result["items"][0]["status"], "damaged")

    def test_small_local_shift_is_not_rejected_as_an_unknown_component(self):
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        moved = cv2.warpAffine(self.normal, np.float32([[1, 0, 3], [0, 1, 2]]), (160, 120), borderMode=cv2.BORDER_REFLECT)
        for _ in range(3):
            result = self.decisions.apply(moved, self.report)
        self.assertEqual(result["items"][0]["status"], "pass")

    def test_local_shift_search_does_not_accept_unrelated_or_removed_structure(self):
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        removed = self.normal.copy()
        removed[20:100, 25:135] = 90
        for frame in [self.absent, removed]:
            for _ in range(3):
                result = self.decisions.apply(frame, self.report)
            self.assertEqual(result["items"][0]["status"], "unavailable")

    def test_dark_or_unlocated_never_gets_defect_label(self):
        self.decisions.calibrate(self.normal, self.report, "pass", "A01")
        result = self.decisions.apply(np.zeros_like(self.normal), self.report)
        self.assertEqual(result["items"][0]["status"], "unavailable")
        self.report["located"] = False
        with self.assertRaises(ValueError):
            self.decisions.calibrate(self.normal, self.report, "pass", "A01")

    def test_dark_textured_component_is_not_rejected_by_mean_brightness(self):
        rows, columns = np.indices((120, 160))
        dark = np.where((rows // 8 + columns // 8) % 2, 32, 8).astype(np.uint8)
        dark = np.repeat(dark[:, :, None], 3, axis=2)
        bright = dark * 2 + 24
        self.decisions.calibrate(bright, self.report, "pass", "A01")
        for _ in range(3):
            result = self.decisions.apply(dark, self.report)
        self.assertEqual(result["items"][0]["status"], "pass")

    def test_near_black_noise_is_not_accepted_as_structure(self):
        noise = np.random.default_rng(16).integers(0, 5, self.normal.shape, dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.decisions.calibrate(noise, self.report, "pass", "A01")

    def test_previous_raw_samples_still_match_after_normalization(self):
        cv2.imwrite(str(self.decisions.directory / "A01_pass_00.png"), self.normal)
        reloaded = CalibratedDecisions(self.temporary.name)
        for _ in range(3):
            result = reloaded.apply(self.normal, self.report)
        self.assertEqual(result["items"][0]["status"], "pass")


if __name__ == "__main__":
    unittest.main()