import time
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from live import DeviceCamera, LiveInspection, resolve_camera
from engine import POST_BOXES, REGIONS


class FakeCapture:
    def __init__(self):
        self.released = False
        self.count = 0

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def read(self):
        self.count += 1
        return True, np.full((120, 160, 3), self.count % 200, np.uint8)

    def release(self):
        self.released = True


class FakeEngine:
    def reset_identity(self):
        pass

    def inspect(self, frame):
        return {"items": [], "width": frame.shape[1], "height": frame.shape[0]}


class DeviceLiveTests(unittest.TestCase):
    def test_auto_camera_requires_one_usb_capture_node_and_deduplicates_links(self):
        first = Mock()
        first.exists.return_value = True
        first.resolve.return_value = "/dev/video7"
        first.__str__ = Mock(return_value="/dev/v4l/by-id/usb-first-video-index0")
        second = Mock()
        second.exists.return_value = True
        second.resolve.return_value = "/dev/video7"
        second.__str__ = Mock(return_value="/dev/v4l/by-id/usb-second-video-index0")
        with patch("live.Path.glob", return_value=[]):
            with self.assertRaisesRegex(ValueError, "found 0"):
                resolve_camera("auto")
        with patch("live.Path.glob", return_value=[first]):
            self.assertEqual(resolve_camera("auto"), str(first))
        with patch("live.Path.glob", return_value=[first, second]):
            with patch("builtins.sorted", side_effect=lambda values: values):
                self.assertEqual(resolve_camera("auto"), str(first))
                second.resolve.return_value = "/dev/video9"
                with self.assertRaisesRegex(ValueError, "found 2"):
                    resolve_camera("auto")
        self.assertEqual(resolve_camera("/dev/explicit"), "/dev/explicit")

    def test_all_components_pass_finishes_early_once_and_requires_removal_before_next_board(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        clock = Mock(return_value=100)
        completed = Mock(side_effect=lambda image, report: report)
        camera = Mock()
        camera.latest.return_value = (1, frame, "")
        live = LiveInspection(camera, Mock(), Mock(), clock=clock, on_complete=completed)
        report = {"model_gate": {"status": "confirmed"}, "items": [{"id": checkpoint, "status": "unavailable" if checkpoint == "A10" else "pass"} for checkpoint, _, _ in REGIONS + POST_BOXES]}
        live._update_cycle(frame, report)
        self.assertEqual(live.cycle_state()["remaining_seconds"], 20)
        with self.assertRaisesRegex(ValueError, "尚未结束"):
            live.reset_session()
        next(item for item in report["items"] if item["id"] == "A10")["status"] = "pass"
        clock.return_value = 103
        live._update_cycle(frame, report)
        completed.assert_called_once()
        self.assertEqual(live.last_result["outcome"], "pass")
        self.assertEqual(live.last_result["cycle_duration_seconds"], 3)
        self.assertEqual(live.last_result["cycle_limit_seconds"], 20)
        self.assertEqual(live.last_result["completion_reason"], "all_passed")
        self.assertEqual(live.cycle_phase, "wait_removal")
        live._update_cycle(frame, report)
        live._update_cycle()
        completed.assert_called_once()
        live._update_cycle(frame, {"board_presence": "absent"})
        next(item for item in report["items"] if item["id"] == "A10")["status"] = "unavailable"
        live._update_cycle(frame, report)
        self.assertEqual(live.cycle_phase, "running")

    def test_missing_component_waits_until_timeout_and_late_frame_cannot_turn_it_into_pass(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        clock = Mock(return_value=100)
        completed = Mock(side_effect=lambda image, report: report)
        camera = Mock()
        camera.latest.return_value = (1, frame, "")
        live = LiveInspection(camera, Mock(), Mock(), clock=clock, on_complete=completed)
        report = {"model_gate": {"status": "confirmed"}, "items": [{"id": checkpoint, "status": "unavailable" if checkpoint == "A06" else "pass"} for checkpoint, _, _ in REGIONS + POST_BOXES]}
        live._update_cycle(frame, report)
        clock.return_value = 119
        live._update_cycle()
        completed.assert_not_called()
        clock.return_value = 120
        next(item for item in report["items"] if item["id"] == "A06")["status"] = "pass"
        live._update_cycle(frame, report)
        self.assertEqual(live.last_result["outcome"], "fail")
        self.assertEqual(live.last_result["completion_reason"], "timeout")
        completed.assert_called_once()

    def test_partial_report_or_unhealthy_camera_cannot_trigger_early_pass(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        for complete in (False, True):
            with self.subTest(complete=complete):
                clock = Mock(return_value=100)
                completed = Mock(side_effect=lambda image, report: report)
                camera = Mock()
                camera.latest.return_value = (1, None, "相机断开") if complete else (1, frame, "")
                live = LiveInspection(camera, Mock(), Mock(), clock=clock, on_complete=completed)
                items = [{"id": checkpoint, "status": "pass"} for checkpoint, _, _ in REGIONS + POST_BOXES] if complete else [{"id": "A06", "status": "pass"}]
                live._update_cycle(frame, {"model_gate": {"status": "confirmed"}, "items": items})
                clock.return_value = 103
                live._update_cycle()
                completed.assert_not_called()
                self.assertEqual(live.cycle_phase, "running")
                clock.return_value = 120
                live._update_cycle()
                self.assertEqual(live.last_result["outcome"], "fail")

    def test_removal_after_identity_reset_rearms_only_after_three_seconds(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        clock = Mock(return_value=100)
        live = LiveInspection(Mock(), Mock(), Mock(), clock=clock)
        live.cycle_phase = "wait_removal"
        live._update_cycle(frame, {"located": False, "board_presence": "unknown"})
        clock.return_value = 102
        live._update_cycle(frame, {"located": False, "board_presence": "unknown"})
        self.assertEqual(live.cycle_phase, "wait_removal")
        clock.return_value = 103
        live._update_cycle(frame, {"located": False, "board_presence": "unknown"})
        self.assertEqual(live.cycle_phase, "waiting")

    def test_cycle_camera_failure_or_interruption_cannot_produce_a_pass(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        for interrupted in [False, True]:
            clock = Mock(return_value=0)
            camera = Mock()
            camera.latest.return_value = (1, None, "相机断开")
            live = LiveInspection(camera, Mock(), Mock(), clock=clock)
            live._update_cycle(frame, {"model_gate": {"status": "confirmed"}, "items": [{"id": "A01", "status": "unavailable"}]})
            if interrupted:
                live._update_cycle(frame, {"board_presence": "absent", "items": []})
            clock.return_value = 21
            live._update_cycle()
            self.assertEqual(live.last_result["outcome"], "fail")
            self.assertEqual(live.last_result["items"][0]["status"], "unavailable")

    def test_recheck_uses_three_new_frames_and_only_selected_item(self):
        frame = np.full((120, 160, 3), 100, np.uint8)
        camera = Mock(source="/dev/test")
        camera.latest.side_effect = [(10, frame, ""), (10, frame, ""), (11, frame, ""), (12, frame, ""), (13, frame, "")]
        engine = Mock()
        engine.inspect.side_effect = lambda image: {"items": [{"id": "A01"}, {"id": "A02"}], "width": 160, "height": 120}
        decisions = Mock(model_confirmed=True, confirm_frames=3)
        decisions.state.return_value = {"A01": {"pass": 2}, "A02": {"pass": 2}}
        decisions.apply.side_effect = lambda image, report: report
        live = LiveInspection(camera, engine, decisions)
        image, report = live.recheck("A01")
        self.assertIs(image, frame)
        self.assertEqual(report["recheck"]["sequences"], [11, 12, 13])
        self.assertEqual(report["recheck"]["requested_after_sequence"], 10)
        self.assertEqual(report["inspection_scope"], "checkpoint")
        self.assertEqual([item["id"] for item in report["items"]], ["A01"])
        decisions.reset_checkpoint.assert_called_once_with("A01")
        self.assertEqual(decisions.apply.call_count, 3)
        self.assertTrue(all([item["id"] for item in call.args[1]["items"]] == ["A01"] for call in decisions.apply.call_args_list))
        self.assertFalse(live.inspection_lock.locked())
        self.assertEqual(live.last_inspected_sequence, 13)
        engine.inspect.reset_mock()
        camera.latest.side_effect = None
        camera.latest.return_value = (13, frame, "")
        worker = threading.Thread(target=live._run)
        worker.start()
        try:
            live.stop_event.wait(0.06)
            engine.inspect.assert_not_called()
        finally:
            live.stop_event.set()
            worker.join(timeout=2)

    def test_recheck_cannot_reuse_frozen_frame_and_allows_unconfirmed_retry(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        camera = Mock(source="/dev/test")
        camera.latest.return_value = (10, frame, "")
        engine = Mock()
        decisions = Mock(model_confirmed=True, confirm_frames=3)
        decisions.state.return_value = {"A01": {"pass": 2}}
        live = LiveInspection(camera, engine, decisions)
        with self.assertRaisesRegex(ValueError, "新画面"):
            live.recheck("A01", timeout=0.01)
        engine.inspect.assert_not_called()
        self.assertFalse(live.inspection_lock.locked())
        decisions.model_confirmed = False
        camera.latest.side_effect = [(20, frame, ""), (21, frame, ""), (22, frame, ""), (23, frame, "")]
        engine.inspect.side_effect = lambda image: {"items": [{"id": "A01", "status": "unavailable"}], "model_gate": {"status": "pending"}}
        decisions.apply.side_effect = lambda image, report: report
        _, report = live.recheck("A01")
        self.assertEqual(report["recheck"]["sequences"], [21, 22, 23])
        self.assertEqual(report["items"][0]["status"], "unavailable")
        with self.assertRaises(ValueError):
            live.recheck("A99")

    def test_recheck_reset_discards_inflight_result(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        camera = Mock(source="/dev/test")
        camera.latest.side_effect = [(10, frame, ""), (11, frame, "")]
        decisions = Mock(model_confirmed=True, confirm_frames=3)
        decisions.state.return_value = {"A01": {"pass": 2}}
        engine = Mock()
        live = LiveInspection(camera, engine, decisions)
        def resetting(image):
            live.reset_session()
            return {"items": [{"id": "A01"}]}
        engine.inspect.side_effect = resetting
        with self.assertRaisesRegex(ValueError, "轮次"):
            live.recheck("A01")
        decisions.apply.assert_not_called()
        self.assertFalse(live.inspection_lock.locked())

    def test_reset_discards_inflight_report_and_previous_snapshots(self):
        capture = FakeCapture()
        camera = DeviceCamera("/dev/test", opener=lambda _: capture)
        entered = threading.Event()
        release = threading.Event()
        engine = FakeEngine()
        original = engine.inspect
        def delayed(frame):
            entered.set()
            release.wait(2)
            return original(frame)
        engine.inspect = delayed
        engine.reset_identity = Mock()
        decisions = Mock()
        decisions.apply.side_effect = lambda frame, report: report
        live = LiveInspection(camera, engine, decisions, interval=0.02)
        try:
            live.start()
            self.assertTrue(entered.wait(2))
            stream_id = live.stream_id
            live.enabled = False
            live.reset_session()
            release.set()
            live.stop_event.wait(0.1)
            self.assertIsNone(live.report)
            self.assertFalse(live.snapshots)
            self.assertNotEqual(stream_id, live.stream_id)
            decisions.reset.assert_called_once()
            decisions.apply.assert_not_called()
            engine.reset_identity.assert_called_once()
        finally:
            release.set()
            live.close()

    def test_device_capture_and_frame_report_pairing(self):
        capture = FakeCapture()
        camera = DeviceCamera("/dev/test", opener=lambda _: capture)
        live = LiveInspection(camera, FakeEngine(), interval=0.02)
        try:
            live.start()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and live.status()["report"] is None:
                live.stop_event.wait(0.02)
            report = live.status()["report"]
            self.assertIsNotNone(report)
            snapshot = live.snapshot(report["sequence"])
            self.assertEqual(snapshot[0], snapshot[3]["sequence"])
            self.assertTrue(snapshot[1].startswith(b"\xff\xd8"))
            self.assertEqual(report["source"], "/dev/test")
            self.assertTrue(live.status()["stream_id"])
            self.assertNotEqual(live.stream_id, LiveInspection(camera, FakeEngine()).stream_id)
        finally:
            live.close()
        self.assertTrue(capture.released)

    def test_stale_frame_never_returned(self):
        camera = DeviceCamera("/dev/test")
        camera.frame = np.zeros((100, 100, 3), np.uint8)
        camera.captured_at = time.monotonic() - 3
        camera.error = ""
        self.assertIsNone(camera.latest()[1])

    def test_fresh_camera_never_makes_stale_report_look_current(self):
        camera = DeviceCamera("/dev/test")
        camera.frame = np.zeros((100, 100, 3), np.uint8)
        camera.captured_at = time.monotonic()
        camera.error = ""
        live = LiveInspection(camera, FakeEngine())
        live.report = {"sequence": 1}
        live.report_created_at = time.monotonic() - 6
        self.assertIsNone(live.status()["report"])
        self.assertIn("过期", live.status()["error"])

    def test_preview_continues_when_inspection_paused(self):
        capture = FakeCapture()
        camera = DeviceCamera("/dev/test", opener=lambda _: capture)
        live = LiveInspection(camera, FakeEngine(), interval=0.5)
        live.enabled = False
        try:
            live.start()
            first = live.preview()
            self.assertIsNotNone(first)
            second = live.preview(first[0])
            self.assertGreater(second[0], first[0])
            self.assertTrue(second[1].startswith(b"\xff\xd8"))
            self.assertEqual(live.processed, 0)
            self.assertTrue(camera.low_latency)
            self.assertFalse(camera.adaptive_exposure)
            self.assertFalse(camera.set_low_latency(False)["low_latency"])
            self.assertTrue(camera.adaptive_exposure)
            self.assertTrue(camera.set_low_latency(True)["low_latency"])
        finally:
            live.close()

    def test_adaptive_exposure_recovers_shadows_with_bounded_shutter(self):
        camera = DeviceCamera("/dev/test")
        camera.capture = FakeCapture()
        camera.adaptive_exposure = True
        camera._adjust_exposure(np.full((120, 160, 3), 12, np.uint8), 1)
        self.assertGreater(camera.exposure, 313)
        for timestamp in range(2, 8):
            camera._adjust_exposure(np.full((120, 160, 3), 12, np.uint8), timestamp)
        self.assertEqual(camera.exposure, 500)
        camera._adjust_exposure(np.full((120, 160, 3), 255, np.uint8), 9)
        self.assertLess(camera.exposure, 500)

    def test_reflection_does_not_force_dark_board_to_underexpose(self):
        camera = DeviceCamera("/dev/test")
        camera.capture = FakeCapture()
        camera.adaptive_exposure = True
        frame = np.full((120, 160, 3), 12, np.uint8)
        frame[30:70, 50:110] = 255
        camera._adjust_exposure(frame, 1)
        self.assertGreater(camera.exposure, 313)


if __name__ == "__main__":
    unittest.main()