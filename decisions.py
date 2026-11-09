import copy
import json
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from engine import POST_BOXES, REGIONS, similarity


LABELS = {"pass": "通过", "missing": "缺失", "damaged": "损坏", "unavailable": "无法检测"}
PASS_THRESHOLD = 0.65
GPIO_THRESHOLD = 0.82


def normalize_patch(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(gray, [2, 98])
    normalized = np.clip((gray.astype(np.float32) - low) * 255 / max(1, high - low), 0, 255).astype(np.uint8)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)


def local_similarity(reference, observed):
    direct = similarity(reference, observed)
    reference = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (128, 96)), (3, 3), 0)
    observed = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(observed, cv2.COLOR_BGR2GRAY), (128, 96)), (3, 3), 0)
    interior = reference[4:-4, 4:-4]
    if float(interior.std()) < 4 or float(observed.std()) < 4:
        return direct
    shifted = float(cv2.matchTemplate(observed, interior, cv2.TM_CCOEFF_NORMED).max())
    return max(direct, min(1.0, shifted))


def gpio_evidence(patch, samples):
    descriptor = cv2.HOGDescriptor((160, 64), (16, 16), (8, 8), (8, 8), 9)

    def features(image):
        luminance = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (160, 64))
        vector = descriptor.compute(luminance).reshape(-1).astype(np.float32)
        vector -= vector.mean()
        length = float(np.linalg.norm(vector))
        return vector / length if length > 1e-6 else None

    observed = features(patch)
    best = 0.0
    if observed is not None:
        for sample in samples:
            reference = features(sample)
            if reference is not None:
                best = max(best, float(np.dot(reference, observed)))
    score = float(np.clip(best, 0, 1))
    return {"present": score >= GPIO_THRESHOLD, "method": "calibrated_gpio_hog", "score": round(score, 3)}


def align_component(frame, item, samples, original_score):
    points = np.float32(item.get("polygon", []))
    if points.shape != (4, 2):
        return None
    width = float(np.linalg.norm(points[1] - points[0]))
    height = float(np.linalg.norm(points[3] - points[0]))
    if width < 8 or height < 8:
        return None
    horizontal = max(1, min(12, int(np.ceil(min(8, width * 0.08) * 127 / width))))
    vertical = max(1, min(24, int(np.ceil(min(8, height * 0.30) * 95 / height))))
    destination = np.float32([[horizontal, vertical], [horizontal + 127, vertical], [horizontal + 127, vertical + 95], [horizontal, vertical + 95]])
    matrix = cv2.getPerspectiveTransform(points, destination)
    search = cv2.warpPerspective(frame, matrix, (128 + horizontal * 2, 96 + vertical * 2))
    search = cv2.GaussianBlur(cv2.cvtColor(search, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    best = None
    for sample in samples:
        template = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY), (128, 96)), (3, 3), 0)
        if float(template.std()) < 4:
            continue
        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(scores)
        alternatives = scores.copy()
        candidate_x, candidate_y = location
        alternatives[max(0, candidate_y - 3):candidate_y + 4, max(0, candidate_x - 3):candidate_x + 4] = -1
        if score < max(0.88, original_score + 0.03) or score - float(alternatives.max()) < 0.02:
            continue
        if best is None or score > best[0]:
            best = (score, location)
    if best is None:
        return None
    score, (horizontal, vertical) = best
    corrected = cv2.perspectiveTransform(np.float32([[[horizontal, vertical], [horizontal + 127, vertical], [horizontal + 127, vertical + 95], [horizontal, vertical + 95]]]), np.linalg.inv(matrix))[0]
    return corrected, {"method": "bounded_component_template", "offset": np.round(corrected.mean(axis=0) - points.mean(axis=0), 2).tolist(), "score": round(score, 3)}


def crop_item(frame, item):
    points = np.float32(item.get("polygon", []))
    if points.shape != (4, 2):
        return None, "检查区域未定位"
    destination = np.float32([[0, 0], [159, 0], [159, 119], [0, 119]])
    matrix = cv2.getPerspectiveTransform(points, destination)
    visible = cv2.warpPerspective(np.full(frame.shape[:2], 255, np.uint8), matrix, (160, 120))
    if float(np.mean(visible > 250)) < 0.97:
        return None, "检查区域超出画面，请扩大取景"
    patch = cv2.warpPerspective(frame, matrix, (160, 120))
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    low, high = np.percentile(smooth, [2, 98])
    if high < 8 or high - low < 6 or float(np.mean(gray <= 2)) > 0.98:
        return None, "区域有效细节不足，正在自动调整成像"
    normalized = normalize_patch(patch)
    if float(smooth.std()) < 1.5 or float(cv2.Laplacian(cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var()) < 12:
        return None, "检查区域模糊或被遮挡，请调整对焦"
    return normalized, ""


def copper_post_evidence(frame, item):
    points = np.float32(item.get("polygon", []))
    if points.shape != (4, 2):
        return {"present": False}
    center = points.mean(axis=0)
    expanded = center + (points - center) * 1.8
    matrix = cv2.getPerspectiveTransform(expanded, np.float32([[0, 0], [159, 0], [159, 119], [0, 119]]))
    patch = cv2.warpPerspective(frame, matrix, (160, 120))
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    metal = cv2.inRange(hsv, np.array([2, 35, 35]), np.array([42, 255, 255]))
    smooth = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(smooth, 20, 55)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    nearby_edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    best = None
    for contour in contours:
        if len(contour) < 12 or not 35 < cv2.contourArea(contour) < 1800:
            continue
        ellipse = cv2.fitEllipse(contour)
        (horizontal, vertical), (width, height), angle = ellipse
        if not (8 <= min(width, height) and max(width, height) <= 65 and min(width, height) / max(width, height) >= 0.42):
            continue
        boundary = np.zeros((120, 160), np.uint8)
        cv2.ellipse(boundary, ellipse, 255, 3)
        edge_fit = float(np.mean(nearby_edges[boundary > 0] > 0))
        if edge_fit < 0.65:
            continue
        outer = np.zeros_like(boundary)
        inner = np.zeros_like(boundary)
        cv2.ellipse(outer, (ellipse[0], (width * 1.15, height * 1.15), angle), 255, -1)
        cv2.ellipse(inner, (ellipse[0], (width * 0.4, height * 0.4), angle), 255, -1)
        ring = (outer > 0) & (inner == 0)
        core = inner > 0
        if ring.sum() < 30 or core.sum() < 10:
            continue
        gold = float(np.mean(metal[ring] > 0))
        hole = np.zeros_like(boundary)
        cv2.ellipse(hole, (ellipse[0], (width * 0.85, height * 0.85), angle), 255, -1)
        rim = (outer > 0) & (hole == 0)
        rim_gold = float(np.mean(metal[rim] > 0)) if rim.sum() >= 20 else 0.0
        contrast = float(np.percentile(smooth[ring], 70) - np.percentile(smooth[core], 20))
        if (gold >= 0.55 or rim_gold >= 0.90) and contrast >= 18:
            candidate = {"present": True, "search_scale": 1.8, "head": [round(horizontal, 1), round(vertical, 1)], "axes": [round(width, 1), round(height, 1)], "metal_fraction": round(gold, 3), "hole_contrast": round(contrast, 1), "edge_fit": round(edge_fit, 3)}
            if best is None or gold > best["metal_fraction"]:
                best = candidate
    return best or {"present": False}


class CalibratedDecisions:
    def __init__(self, directory, confirm_frames=3, pass_hold_seconds=None, clock=None):
        self.directory = Path(directory)
        self.lock = threading.Lock()
        self.samples = defaultdict(lambda: defaultdict(list))
        self.votes = defaultdict(lambda: deque(maxlen=confirm_frames))
        self.confirm_frames = confirm_frames
        self.window_confirmations = {checkpoint: deque(maxlen=confirm_frames) for checkpoint in ("A06", "A12")}
        self.pass_hold_seconds = None if pass_hold_seconds is None else max(0.0, float(pass_hold_seconds))
        self.clock = clock or time.monotonic
        self.last_pass = {}
        self.model_confirmed = False
        self.confirmed_identity = None
        self.load()

    def reset(self):
        with self.lock:
            self.votes.clear()
            for confirmations in self.window_confirmations.values():
                confirmations.clear()
            self.last_pass.clear()
            self.model_confirmed = False
            self.confirmed_identity = None
        return {"reset": True, "required_model": "M2"}

    def reset_checkpoint(self, checkpoint):
        with self.lock:
            self.votes.pop(checkpoint, None)
            self.last_pass.pop(checkpoint, None)
            if checkpoint in self.window_confirmations:
                self.window_confirmations[checkpoint].clear()

    def load(self):
        if not self.directory.exists():
            return
        for path in self.directory.glob("*.png"):
            try:
                identity, label, _ = path.stem.split("_")
            except ValueError:
                continue
            if identity not in {code for code, _, _ in REGIONS + POST_BOXES}:
                continue
            sample = cv2.imread(str(path))
            if label in {"pass", "missing", "damaged"} and sample is not None:
                self.samples[identity][label].append(normalize_patch(sample))

    def state(self):
        with self.lock:
            return {identity: {label: len(samples) for label, samples in classes.items()} for identity, classes in self.samples.items()}

    def calibrate(self, frame, report, label="pass", checkpoint=None):
        if label not in {"pass", "missing", "damaged"}:
            raise ValueError("标定类别无效")
        if not report.get("located"):
            raise ValueError("未定位到完整开发板，不能标定")
        if label != "pass" and checkpoint is None:
            raise ValueError("请先选择缺失或损坏的检查位置")
        selected = [item for item in report["items"] if checkpoint is None or item["id"] == checkpoint]
        expected = {code for code, _, _ in REGIONS + POST_BOXES}
        if not selected or (checkpoint is None and (len(selected) != len(expected) or {item["id"] for item in selected} != expected)):
            raise ValueError("检查位置不完整，不能标定")
        patches = []
        for item in selected:
            patch, reason = crop_item(frame, item)
            if patch is None:
                raise ValueError(f"{item['name']}：{reason}")
            patches.append((item["id"], patch))
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.lock:
            for identity, patch in patches:
                index = len(self.samples[identity][label])
                if index >= 12:
                    raise ValueError("该位置的此类样本已达到 12 张上限")
            for identity, patch in patches:
                index = len(self.samples[identity][label])
                path = self.directory / f"{identity}_{label}_{index:02}.png"
                if not cv2.imwrite(str(path), patch):
                    raise OSError("标定样本无法保存")
                self.samples[identity][label].append(patch)
            self.votes.clear()
            self.last_pass.clear()
            self.model_confirmed = False
            self.confirmed_identity = None
            for confirmations in self.window_confirmations.values():
                confirmations.clear()
        metadata = {"method": "operator_labeled_local_templates", "counts": self.state()}
        (self.directory / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if checkpoint is None and label == "pass":
            if not cv2.imwrite(str(self.directory / "approved-complete-board.jpg"), frame):
                raise OSError("完整板标准原图无法保存")
            reference = {"confirmed_by": "operator", "profile": "m2-common", "version_policy": "require_m2_model", "source": "approved-complete-board.jpg", "items": [{"id": item["id"], "polygon": item["polygon"]} for item in selected]}
            (self.directory / "approved-standard.json").write_text(json.dumps(reference, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def apply(self, frame, report):
        result = copy.deepcopy(report)
        result["method"] = "M2 型号确认 / SIFT定位 / 工位标定局部样本 / 本轮结果锁定"
        result["notice"] = "外观检查，不代表电气功能正常；仅对已标定的缺失和损坏外观输出结论。"
        now = self.clock()
        with self.lock:
            version = result.get("version") or {}
            texts = [version.get("device_name", ""), version.get("model", "")] + [token.get("text", "") for token in version.get("raw", []) if token.get("score", 0) >= 0.85]
            models = {re.sub(r"\s+", "", model) for model in re.findall(r"(?<![A-Z0-9])(?:[MHL]\s*\d+|RK\d{4}|RASPBERRY(?:\s+PI)?|JETSON|ARDUINO|BEAGLEBONE)(?![A-Z0-9])", " ".join(texts).upper())}
            named_model = re.search(r"\b(?:BOARD|PI|MODULE)\s+([A-Z][A-Z0-9-]+)\b", version.get("device_name", "").upper())
            if named_model:
                models.add(named_model.group(1))
            branded_model = re.search(r"QUECTEL\s*PI\s*([MHL]\s*\d+)\b", " ".join(texts).upper())
            declared_model = re.sub(r"\s+", "", version.get("model", "").upper())
            if branded_model:
                models = {re.sub(r"\s+", "", branded_model.group(1))}
            elif declared_model:
                models = {declared_model}
            rejected = bool(models and models != {"M2"})
            absent = result.get("board_presence") == "absent"
            if rejected or absent:
                self.votes.clear()
                for confirmations in self.window_confirmations.values():
                    confirmations.clear()
                self.last_pass.clear()
                self.model_confirmed = False
                self.confirmed_identity = None
            elif models == {"M2"}:
                self.model_confirmed = True
                self.confirmed_identity = copy.deepcopy(version)
            gate_reason = "板卡已移出，等待放入设备" if absent else "非 M2 板卡，停止检测" if rejected else "等待识别 Quectel Pi M2"
            result["model_gate"] = {"required": "M2", "status": "rejected" if rejected else "confirmed" if self.model_confirmed else "pending", "observed": sorted(models), "identity": copy.deepcopy(self.confirmed_identity)}
            for item in result["items"]:
                status = "unavailable"
                reason = "未定位到板卡正面，无法确定检查区域" if self.model_confirmed else gate_reason
                patch, quality_reason = crop_item(frame, item) if self.model_confirmed and result.get("located") else (None, reason)
                classes = self.samples.get(item["id"], {})
                scores = {}
                copper = {"present": False}
                gpio = {"present": False}
                if patch is None:
                    reason = quality_reason
                elif not classes.get("pass"):
                    reason = "尚未采集此工位的完整板标准"
                else:
                    scores = {label: max(local_similarity(sample, patch) for sample in samples) for label, samples in classes.items() if samples}
                    if item["id"] in {"A01", "A06", "A07", "A08"} and scores.get("pass", 0) < PASS_THRESHOLD and max((value for label, value in scores.items() if label != "pass"), default=0) < 0.86:
                        alignment = align_component(frame, item, classes["pass"], scores["pass"])
                        if alignment is not None:
                            corrected, detail = alignment
                            candidate = {**item, "polygon": corrected.tolist()}
                            aligned_patch, _ = crop_item(frame, candidate)
                            if aligned_patch is not None:
                                aligned_scores = {label: max(local_similarity(sample, aligned_patch) for sample in samples) for label, samples in classes.items() if samples}
                                if aligned_scores.get("pass", 0) >= PASS_THRESHOLD:
                                    item["polygon"] = np.round(corrected, 1).tolist()
                                    item["local_alignment"] = detail
                                    scores = {label: value if label == "pass" else max(value, scores.get(label, 0)) for label, value in aligned_scores.items()}
                    if item["id"] == "A01" and scores.get("pass", 0) < PASS_THRESHOLD and max((value for label, value in scores.items() if label != "pass"), default=0) < 0.86:
                        gpio = gpio_evidence(patch, classes["pass"])
                        if gpio["present"]:
                            scores["pass"] = max(scores["pass"], gpio["score"])
                    ranked = sorted(scores.items(), key=lambda entry: entry[1], reverse=True)
                    label, score = ranked[0]
                    margin = score - ranked[1][1] if len(ranked) > 1 else score
                    threshold = PASS_THRESHOLD if label == "pass" else 0.86
                    if score >= threshold and margin >= 0.12:
                        status = label
                        reason = "排针梯度结构匹配已标定正常样本" if label == "pass" and gpio["present"] else f"匹配已标定的{LABELS[label]}样本"
                    else:
                        reason = f"局部尚未确认：匹配 {score:.0%}，要求 {threshold:.0%}；不代表器件缺失或型号不符"
                    if item["id"].startswith("B") and status not in {"missing", "damaged"}:
                        copper = copper_post_evidence(frame, item)
                        if copper["present"]:
                            status = "pass"
                            reason = "铜色金属与带孔柱头轮廓确认"
                item["pass_threshold"] = GPIO_THRESHOLD if gpio["present"] else PASS_THRESHOLD
                item["copper_evidence"] = copper if item["id"].startswith("B") else None
                item["gpio_evidence"] = gpio if item["id"] == "A01" else None
                votes = self.votes[item["id"]]
                previous = self.last_pass.get(item["id"])
                if self.pass_hold_seconds is not None and previous is not None and now - previous >= self.pass_hold_seconds:
                    self.last_pass.pop(item["id"], None)
                    votes.clear()
                observed_status = status
                votes.append(status)
                confirmed = len(votes) >= self.confirm_frames and all(vote == status for vote in votes)
                confirmation_reason = f"连续 {self.confirm_frames} 帧确认中"
                if item["id"] in self.window_confirmations:
                    confirmations = self.window_confirmations[item["id"]]
                    while confirmations and now - confirmations[0] > 3.0:
                        confirmations.popleft()
                    if observed_status in {"missing", "damaged"}:
                        confirmations.clear()
                    if observed_status == "pass":
                        confirmations.append(now)
                        required = min(2, self.confirm_frames)
                        confirmed = len(confirmations) >= required
                        confirmation_reason = f"最近3秒内 {required} 次正常确认中"
                if status != "unavailable" and not confirmed:
                    status = "unavailable"
                    reason = confirmation_reason
                previous = self.last_pass.get(item["id"])
                held = False
                remaining = 0
                if status == "pass":
                    self.last_pass[item["id"]] = now
                elif previous is not None:
                    eligible = patch is not None and scores.get("pass", 0) >= 0.60 and scores["pass"] > max((score for label, score in scores.items() if label != "pass"), default=0)
                    persistent = self.pass_hold_seconds is None and self.model_confirmed
                    if observed_status in {"pass", "unavailable"} and (persistent or (self.pass_hold_seconds is not None and eligible and now - previous < self.pass_hold_seconds)):
                        if observed_status == "pass":
                            self.last_pass[item["id"]] = now
                        remaining = 0 if persistent else max(0, self.pass_hold_seconds - (now - self.last_pass[item["id"]]))
                        status = "pass"
                        held = True
                        reason = "已检测通过" if persistent else f"短时保持最近已确认通过，剩余 {remaining:.1f} 秒"
                    else:
                        self.last_pass.pop(item["id"], None)
                item.update(status=status, reason=reason, scores={key: round(value, 3) for key, value in scores.items()}, decision_label=LABELS[status], observed_status=observed_status, temporal_hold=held, session_latched=held and self.pass_hold_seconds is None, evidence_age_ms=round((now - self.last_pass[item["id"]]) * 1000) if item["id"] in self.last_pass else None, hold_remaining_ms=None if held and self.pass_hold_seconds is None else round(remaining * 1000))
            if not result.get("located"):
                self.votes.clear()
                if self.pass_hold_seconds is not None:
                    self.last_pass.clear()
        result["held_count"] = sum(item["temporal_hold"] for item in result["items"])
        states = [item["status"] for item in result["items"]]
        if "missing" in states or "damaged" in states:
            result["verdict"] = f"缺失 {states.count('missing')} / 损坏 {states.count('damaged')}"
        elif states and all(status == "pass" for status in states):
            result["verdict"] = "检测通过（短时保持）" if result["held_count"] and self.pass_hold_seconds is not None else "检测通过"
        else:
            result["verdict"] = "无法检测"
        result["decision_mode"] = "calibrated"
        result["inspection_profile"] = "m2-common"
        result["version_policy"] = "require_m2_model"
        result["revision_policy"] = "informational_only"
        result["blocking_reason"] = ""
        if result["verdict"] == "无法检测":
            if result["model_gate"]["status"] != "confirmed":
                result["blocking_reason"] = gate_reason
            elif not result.get("located"):
                result["blocking_reason"] = "板卡未定位"
            elif any(item["reason"] == "尚未采集此工位的完整板标准" for item in result["items"]):
                result["blocking_reason"] = "缺少完整板标准"
            else:
                result["blocking_reason"] = "检查区域未确认"
        return result