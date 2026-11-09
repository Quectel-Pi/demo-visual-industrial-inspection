import copy
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from engine import POST_BOXES, REGIONS


def resolve_camera(source):
    if source != "auto":
        return source
    candidates = {}
    for path in sorted(Path("/dev/v4l/by-id").glob("usb-*-video-index0")):
        if path.exists():
            candidates.setdefault(str(path.resolve()), str(path))
    if len(candidates) != 1:
        detail = ", ".join(candidates.values()) or "none"
        raise ValueError(f"Expected one USB camera, found {len(candidates)} ({detail}). Use --camera /dev/v4l/by-id/… or --image-only.")
    return next(iter(candidates.values()))


class DeviceCamera:
    def __init__(self, source, width=640, height=480, opener=None):
        self.source = resolve_camera(source)
        self.width = width
        self.height = height
        self.opener = opener or (lambda path: cv2.VideoCapture(path, cv2.CAP_V4L2))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.frame = None
        self.sequence = 0
        self.captured_at = 0
        self.error = "相机正在启动"
        self.capture = None
        self.thread = None
        self.timestamps = deque(maxlen=90)
        self.low_latency = True
        self.exposure_request = None
        self.adaptive_exposure = False
        self.exposure = 313
        self.gamma = None
        self.last_exposure_adjustment = 0

    def start(self):
        self.capture = self.opener(self.source)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"设备相机无法打开：{self.source}")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        if self.capture.set(cv2.CAP_PROP_GAMMA, 120):
            self.gamma = 120
        self._apply_exposure(self.low_latency)
        self.thread = threading.Thread(target=self._capture, name="device-camera", daemon=True)
        self.thread.start()

    def _capture(self):
        failures = 0
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    request = self.exposure_request
                    self.exposure_request = None
                if request is not None:
                    request["ok"] = self._apply_exposure(request["enabled"])
                    request["done"].set()
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    failures += 1
                    with self.lock:
                        self.error = "相机读取失败，请检查连接"
                    if failures >= 10:
                        break
                    self.stop_event.wait(0.1)
                    continue
                failures = 0
                self._adjust_exposure(frame, time.monotonic())
                with self.lock:
                    self.frame = frame
                    self.sequence += 1
                    self.captured_at = time.monotonic()
                    self.timestamps.append(self.captured_at)
                    self.error = ""
        finally:
            self.capture.release()

    def latest(self):
        with self.lock:
            fresh = time.monotonic() - self.captured_at < 2
            frame = self.frame.copy() if self.frame is not None and fresh and not self.error else None
            return self.sequence, frame, self.error or ("相机画面已过期" if not fresh else "")

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)

    def fps(self):
        with self.lock:
            stamps = list(self.timestamps)
        return round((len(stamps) - 1) / max(0.01, stamps[-1] - stamps[0]), 1) if len(stamps) > 1 else 0

    def _apply_exposure(self, enabled):
        ok = self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.exposure = 313
        if ok:
            ok = self.capture.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        if not ok:
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        self.low_latency = bool(enabled and ok)
        self.adaptive_exposure = bool(not enabled and ok)
        self.last_exposure_adjustment = time.monotonic()
        return bool(ok)

    def _adjust_exposure(self, frame, now):
        if not self.adaptive_exposure or now - self.last_exposure_adjustment < 0.7:
            return
        self.last_exposure_adjustment = now
        gray = cv2.cvtColor(cv2.resize(frame, (80, 60)), cv2.COLOR_BGR2GRAY)[6:54, 8:72]
        clipped = float(np.mean(gray >= 248))
        usable = gray[gray < 240]
        level = float(np.percentile(usable if usable.size > gray.size * 0.2 else gray, 65))
        if 48 <= level <= 88 and clipped < 0.6:
            return
        ratio = 0.8 if clipped >= 0.6 else float(np.clip(64 / max(1, level), 0.8, 1.25))
        exposure = int(np.clip(round(self.exposure * ratio), 120, 500))
        if exposure != self.exposure and self.capture.set(cv2.CAP_PROP_EXPOSURE, exposure):
            self.exposure = exposure

    def set_low_latency(self, enabled):
        request = {"enabled": enabled, "done": threading.Event(), "ok": False}
        with self.lock:
            if self.exposure_request is not None:
                raise ValueError("相机设置正在应用，请稍后重试")
            self.exposure_request = request
        if not request["done"].wait(2) or not request["ok"]:
            raise ValueError("相机不支持该曝光设置或未响应")
        return {"low_latency": self.low_latency}


class LiveInspection:
    def __init__(self, camera, engine, decisions=None, interval=0.5, cycle_seconds=20, on_complete=None, clock=None):
        self.camera = camera
        self.engine = engine
        self.decisions = decisions
        self.interval = interval
        self.lock = threading.Lock()
        self.inspection_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.enabled = True
        self.snapshots = deque(maxlen=8)
        self.report = None
        self.error = "等待相机画面"
        self.thread = None
        self.processed = 0
        self.started_at = time.monotonic()
        self.stream_id = uuid4().hex
        self.report_created_at = 0
        self.session_generation = 0
        self.last_inspected_sequence = -1
        self.preview_condition = threading.Condition()
        self.preview_packet = None
        self.preview_thread = None
        self.preview_timestamps = deque(maxlen=90)
        self.clock = clock or time.monotonic
        self.cycle_seconds = cycle_seconds
        self.on_complete = on_complete or (lambda frame, report: report)
        self.cycle_phase = "waiting"
        self.cycle_report = None
        self.cycle_frame = None
        self.cycle_started = None
        self.cycle_interrupted = False
        self.last_result = None
        self.removal_since = None

    def cycle_state(self):
        with self.lock:
            remaining = max(0, self.cycle_seconds - (self.clock() - self.cycle_started)) if self.cycle_phase == "running" else 0
            return {"phase": self.cycle_phase, "remaining_seconds": round(remaining, 1), "duration_seconds": self.cycle_seconds, "cycle_id": (self.cycle_report or {}).get("cycle_id"), "last_result": copy.deepcopy(self.last_result)}

    def _all_components_pass(self):
        items = (self.cycle_report or {}).get("items", [])
        expected = {checkpoint for checkpoint, _, _ in REGIONS + POST_BOXES}
        return not self.cycle_interrupted and len(items) == len(expected) and {item["id"] for item in items} == expected and all(item["status"] == "pass" for item in items)

    def _complete_cycle(self, now):
        _, current_frame, camera_error = self.camera.latest()
        if current_frame is None or camera_error:
            if now - self.cycle_started < self.cycle_seconds:
                return
            self.cycle_interrupted = True
        completed = copy.deepcopy(self.cycle_report)
        passed = self._all_components_pass()
        completed.update(inspection_scope="board_cycle", cycle_complete=True, cycle_duration_seconds=round(now - self.cycle_started, 3), cycle_limit_seconds=self.cycle_seconds, completion_reason="all_passed" if passed else "timeout", cycle_interrupted=self.cycle_interrupted, completed_at=datetime.now(timezone.utc).isoformat(), outcome="pass" if passed else "fail", verdict="合格" if passed else "未通过", blocking_reason="", live=False)
        self.last_result = self.on_complete(self.cycle_frame, completed)
        self.cycle_phase = "wait_removal"
        self.removal_since = None
        self.session_generation += 1
        self.decisions.reset()
        self.engine.reset_identity()
        self.report = None
        self.snapshots.clear()

    def _update_cycle(self, frame=None, report=None):
        now = self.clock()
        if self.cycle_phase == "running" and (now - self.cycle_started >= self.cycle_seconds or self._all_components_pass()):
            self._complete_cycle(now)
            return
        if report is None:
            return
        if self.cycle_phase == "wait_removal":
            visible = report.get("located") or report.get("board_presence") == "present"
            self.removal_since = None if visible else now if self.removal_since is None else self.removal_since
            if report.get("board_presence") == "absent" or (self.removal_since is not None and now - self.removal_since >= 3):
                self.cycle_phase = "waiting"
                self.cycle_report = None
                self.decisions.reset()
                self.engine.reset_identity()
            return
        confirmed = report.get("model_gate", {}).get("status") == "confirmed"
        if self.cycle_phase == "waiting":
            if not confirmed or not report.get("items"):
                return
            self.cycle_phase = "running"
            self.cycle_started = now
            self.cycle_interrupted = False
            self.cycle_report = copy.deepcopy(report)
            self.cycle_report.update(cycle_id=uuid4().hex, inspection_started_at=datetime.now(timezone.utc).isoformat())
            self.cycle_frame = frame.copy()
        if report.get("board_presence") == "absent" or report.get("model_gate", {}).get("status") == "rejected":
            self.cycle_interrupted = True
        if self.cycle_interrupted:
            return
        previous = {item["id"]: item for item in self.cycle_report["items"]}
        for item in report["items"]:
            earlier = previous.get(item["id"])
            if earlier is None:
                continue
            if earlier["status"] in {"missing", "damaged"} or (earlier["status"] == "pass" and item["status"] == "unavailable"):
                continue
            previous[item["id"]] = copy.deepcopy(item)
        self.cycle_report.update(sequence=report.get("sequence"), version=copy.deepcopy(report.get("version", {})))
        self.cycle_report["items"] = list(previous.values())
        self.cycle_frame = frame.copy()
        if self._all_components_pass():
            self._complete_cycle(now)

    def start(self):
        self.camera.start()
        self.preview_thread = threading.Thread(target=self._preview, name="camera-preview", daemon=True)
        self.preview_thread.start()
        self.thread = threading.Thread(target=self._run, name="device-inspection", daemon=True)
        self.thread.start()

    def reset_session(self):
        with self.lock:
            if self.cycle_phase == "running":
                raise ValueError("本轮检测尚未结束，请等待20秒结算后移板")
            self.cycle_report = None
            self.session_generation += 1
            if self.decisions is not None:
                self.decisions.reset()
            self.engine.reset_identity()
            self.report = None
            self.report_created_at = 0
            self.snapshots.clear()
            self.stream_id = uuid4().hex
        return {"reset": True, "required_model": "M2"}

    def recheck(self, checkpoint, timeout=8):
        if self.decisions is None or checkpoint not in self.decisions.state():
            raise ValueError("请选择已有正常标准的检查项")
        if not self.inspection_lock.acquire(timeout=2):
            raise ValueError("检测正在处理，请稍后复验")
        try:
            with self.lock:
                generation = self.session_generation
                stream_id = self.stream_id
                previous, frame, error = self.camera.latest()
                if frame is None or error:
                    raise ValueError(error or "相机画面尚未就绪")
                requested_after = previous
                self.decisions.reset_checkpoint(checkpoint)
                cycle_id = (self.cycle_report or {}).get("cycle_id")
                parent_report_id = (self.last_result or {}).get("id") if self.cycle_phase == "wait_removal" else None
                if self.cycle_phase == "running":
                    for item in self.cycle_report["items"]:
                        if item["id"] == checkpoint:
                            item.update(status="unavailable", reason="等待单模块复检")
            deadline = time.monotonic() + timeout
            sequences = []
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                if self.cycle_phase == "running" and self.clock() - self.cycle_started >= self.cycle_seconds:
                    raise ValueError("本轮20秒已结束，结算后可继续单模块复检")
                sequence, frame, error = self.camera.latest()
                if error:
                    raise ValueError(error)
                if frame is None or sequence <= previous:
                    self.stop_event.wait(0.03)
                    continue
                previous = sequence
                report = self.engine.inspect(frame)
                report["items"] = [item for item in report["items"] if item["id"] == checkpoint]
                if len(report["items"]) != 1:
                    raise ValueError("选中检查项未定位")
                with self.lock:
                    if generation != self.session_generation:
                        raise ValueError("检测轮次已更换，请重新复验")
                    report = self.decisions.apply(frame, report)
                    self.last_inspected_sequence = sequence
                sequences.append(sequence)
                if len(sequences) >= self.decisions.confirm_frames:
                    with self.lock:
                        if self.cycle_phase == "running":
                            for index, item in enumerate(self.cycle_report["items"]):
                                if item["id"] == checkpoint:
                                    self.cycle_report["items"][index] = copy.deepcopy(report["items"][0])
                    report.update(sequence=sequence, stream_id=stream_id, source=self.camera.source, live=True, inspection_scope="checkpoint", cycle_id=cycle_id, parent_report_id=parent_report_id, recheck={"checkpoint": checkpoint, "requested_after_sequence": requested_after, "sequences": sequences})
                    return frame, report
            raise ValueError("复验未取得足够的新画面，请重试")
        finally:
            self.inspection_lock.release()

    def _preview(self):
        previous = -1
        while not self.stop_event.is_set():
            started = time.monotonic()
            sequence, frame, _ = self.camera.latest()
            if frame is not None and sequence != previous:
                previous = sequence
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    with self.preview_condition:
                        self.preview_packet = (sequence, encoded.tobytes())
                        self.preview_timestamps.append(time.monotonic())
                        self.preview_condition.notify_all()
            self.stop_event.wait(max(0, 0.05 - (time.monotonic() - started)))

    def preview(self, previous=-1):
        with self.preview_condition:
            self.preview_condition.wait_for(lambda: self.stop_event.is_set() or (self.preview_packet is not None and self.preview_packet[0] != previous), timeout=2)
            return self.preview_packet if not self.stop_event.is_set() and self.preview_packet is not None and self.preview_packet[0] != previous else None

    def _run(self):
        previous = -1
        while not self.stop_event.is_set():
            try:
                with self.inspection_lock:
                    with self.lock:
                        self._update_cycle()
            except Exception as error:
                with self.lock:
                    self.error = f"结果保存失败：{type(error).__name__}，正在重试"
                self.stop_event.wait(0.2)
                continue
            if not self.enabled:
                self.stop_event.wait(0.1)
                continue
            started = time.monotonic()
            sequence, frame, error = self.camera.latest()
            if frame is None or sequence == previous:
                with self.lock:
                    self.error = error or "等待新画面"
                self.stop_event.wait(0.1)
                continue
            previous = sequence
            try:
                with self.inspection_lock:
                    sequence, frame, error = self.camera.latest()
                    if frame is None or error or sequence <= self.last_inspected_sequence:
                        continue
                    previous = sequence
                    with self.lock:
                        generation = self.session_generation
                    report = self.engine.inspect(frame)
                    with self.lock:
                        if generation != self.session_generation:
                            continue
                        if self.decisions is not None:
                            report = self.decisions.apply(frame, report)
                        self.last_inspected_sequence = sequence
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    raise RuntimeError("相机图像编码失败")
                self.processed += 1
                report.update(sequence=sequence, source=self.camera.source, live=True, id=f"live-{sequence}", image_url=f"/api/live/frame.jpg?sequence={sequence}", processed=self.processed, fps=round(self.processed / max(0.01, time.monotonic() - self.started_at), 2))
                with self.lock:
                    if generation != self.session_generation:
                        continue
                    self._update_cycle(frame, report)
                    if generation != self.session_generation:
                        continue
                    self.snapshots.append((sequence, encoded.tobytes(), frame, report))
                    self.report = report
                    self.report_created_at = time.monotonic()
                    self.error = ""
            except Exception as error:
                with self.lock:
                    self.error = f"实时检测失败：{type(error).__name__}"
                print(f"[live-error] {error}", flush=True)
            self.stop_event.wait(max(0, self.interval - (time.monotonic() - started)))

    def status(self):
        camera_sequence, frame, camera_error = self.camera.latest()
        with self.lock:
            report = dict(self.report) if self.report is not None else None
            error = camera_error or self.error
            report_age_ms = round((time.monotonic() - self.report_created_at) * 1000) if self.report is not None else None
        if self.enabled and report_age_ms is not None and report_age_ms > 5000:
            report = None
            error = camera_error or "检测结果已过期，正在重新检测"
        if frame is None or error:
            report = None
        with self.preview_condition:
            stamps = list(self.preview_timestamps)
        preview_fps = round((len(stamps) - 1) / max(0.01, stamps[-1] - stamps[0]), 1) if len(stamps) > 1 else 0
        return {"enabled": self.enabled, "error": error, "report": report, "cycle": self.cycle_state(), "stream_id": self.stream_id, "camera_sequence": camera_sequence, "report_age_ms": report_age_ms, "capture_fps": self.camera.fps(), "preview_fps": preview_fps, "low_latency": self.camera.low_latency, "adaptive_exposure": self.camera.adaptive_exposure, "exposure_ms": self.camera.exposure / 10, "camera_gamma": self.camera.gamma}

    def snapshot(self, sequence=None):
        with self.lock:
            for entry in reversed(self.snapshots):
                if sequence is None or entry[0] == sequence:
                    return entry
        return None

    def close(self):
        self.stop_event.set()
        with self.preview_condition:
            self.preview_condition.notify_all()
        if self.preview_thread is not None:
            self.preview_thread.join(timeout=3)
        if self.thread is not None:
            self.thread.join(timeout=8)
        self.camera.close()