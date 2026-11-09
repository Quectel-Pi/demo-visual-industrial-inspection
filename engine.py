import json
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np


PHOTO_DIR = Path(__file__).resolve().parent / "assets" / "reference"
BARE_PHOTO = PHOTO_DIR / "9ed83342-c1c1-4f99-96c4-e08a240ae80f.jpg"
POST_PHOTO = PHOTO_DIR / "img_v3_0216t_613df002-644b-459e-ae62-cc3339a2bcag.jpg"
REGIONS = [
    ("A01", "40 Pin GPIO 排针", (108, 12, 853, 92)),
    ("A02", "SD / SIM 卡座", (517, 183, 727, 409)),
    ("A03", "HDMI 接口", (307, 687, 576, 851)),
    ("A04", "CAM0 相机排线座", (631, 600, 698, 812)),
    ("A05", "耳机接口", (718, 629, 831, 851)),
    ("A06", "上方 USB 组件", (1013, 300, 1278, 530)),
    ("A07", "下方 USB 组件", (1013, 573, 1278, 792)),
    ("A08", "以太网接口", (965, 6, 1278, 258)),
    ("A09", "Type-C 电源接口", (112, 715, 256, 845)),
    ("A10", "POE 排针", (862, 102, 938, 185)),
    ("A11", "PCIe 排线座", (16, 290, 67, 496)),
    ("A12", "VRTC 连接座", (212, 609, 290, 667)),
]
POST_BOXES = [
    ("B01", "左上铜柱", (94, 263, 286, 419)),
    ("B02", "右上铜柱", (757, 262, 855, 483)),
    ("B03", "左下铜柱", (28, 818, 237, 942)),
    ("B04", "右下铜柱", (770, 832, 878, 942)),
]
MODEL_BOUNDS = (590, 103, 648, 145)


def polygon(bounds):
    left, top, right, bottom = bounds
    return np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])


def transform(points, matrix):
    return cv2.perspectiveTransform(np.float32(points).reshape(1, -1, 2), matrix)[0]


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def feature_gray(image):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray(image))


def enhance_text(image):
    luminance = gray(image)
    contrast = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4)).apply(luminance)
    enlarged = cv2.resize(contrast, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    sharpened = cv2.addWeighted(enlarged, 1.6, cv2.GaussianBlur(enlarged, (0, 0), 0.8), -0.6, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def similarity(reference, observed):
    size = (128, 96)
    reference = cv2.resize(gray(reference), size)
    observed = cv2.resize(gray(observed), size)
    if float(reference.std()) < 4 or float(observed.std()) < 4:
        return 0.0
    reference = cv2.GaussianBlur(reference, (3, 3), 0)
    observed = cv2.GaussianBlur(observed, (3, 3), 0)
    return float(np.clip(cv2.matchTemplate(observed, reference, cv2.TM_CCOEFF_NORMED)[0, 0], 0, 1))


def refine_region(reference, observed, points, radius=6):
    left, top = np.floor(np.min(points, axis=0)).astype(int)
    right, bottom = np.ceil(np.max(points, axis=0)).astype(int)
    height, width = reference.shape[:2]
    if left < 0 or top < 0 or right > width or bottom > height or right - left < 8 or bottom - top < 8:
        return np.float32(points), None
    template = reference[top:bottom, left:right]
    if float(template.std()) < 4:
        return np.float32(points), None
    search_left, search_top = max(0, left - radius), max(0, top - radius)
    search = observed[search_top:min(height, bottom + radius), search_left:min(width, right + radius)]
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(scores)
    origin = (left - search_left, top - search_top)
    offset = np.int32(location) - np.int32(origin)
    alternatives = scores.copy()
    candidate_x, candidate_y = location
    alternatives[max(0, candidate_y - 2):candidate_y + 3, max(0, candidate_x - 2):candidate_x + 3] = -1
    if score < 0.88 or score - float(scores[origin[1], origin[0]]) < 0.03 or score - float(alternatives.max()) < 0.02:
        return np.float32(points), None
    return np.float32(points) + offset, {"offset": offset.tolist(), "score": round(score, 3)}


class InspectionEngine:
    def __init__(self):
        self.bare = cv2.imread(str(BARE_PHOTO))
        self.posts = cv2.imread(str(POST_PHOTO))
        if self.bare is None or self.posts is None:
            raise RuntimeError("参考照片不存在或无法读取")
        self.detector = cv2.SIFT_create(nfeatures=4500)
        self.feature_matchers = {}
        source_landmarks = np.float32([[857, 337], [1095, 501], [916, 544], [1115, 694], [920, 754], [1125, 897], [390, 810], [532, 947]])
        target_landmarks = np.float32([[968, 10], [1272, 250], [1030, 310], [1278, 510], [1030, 580], [1270, 786], [310, 697], [544, 850]])
        self.post_matrix, _ = cv2.findHomography(source_landmarks, target_landmarks, 0)
        reference_mask = np.full(self.bare.shape[:2], 255, np.uint8)
        reference_mask[98:190, 449:744] = 0
        post_mask = cv2.warpPerspective(reference_mask, np.linalg.inv(self.post_matrix), (self.posts.shape[1], self.posts.shape[0]), borderValue=255)
        self.reference_features = self.detector.detectAndCompute(feature_gray(self.bare), reference_mask)
        self.post_features = self.detector.detectAndCompute(feature_gray(self.posts), post_mask)
        self.live_reference_features = self.scaled_features(self.bare, reference_mask)
        self.live_post_features = self.scaled_features(self.posts, post_mask)
        self.lock = threading.Lock()
        self.ocr = None
        self.latin_ocr = None
        self.ocr_error = "OCR 引擎未安装"
        self.ocr_interval = 0.0
        self.ocr_sampled_at = 0
        self.ocr_located = None
        self.ocr_cached = None
        self.calibrated = False
        self.ocr_condition = threading.Condition()
        self.ocr_thread = None
        self.ocr_stopping = False
        self.ocr_job = None
        self.ocr_busy = False
        self.ocr_generation = 0
        self.ocr_rotation_index = 0
        self.ocr_candidate_count = 0
        self.board_seen = False
        self.board_level = None
        self.target_lost_since = None
        self.empty_since = None
        self.empty_frames = 0
        self.empty_gray = None
        self.station_reference = None
        self.station_posts = {}
        self.station_regions = {}
        self.station_source_regions = {}
        self.station_gray = None
        self.tracked_reference = None
        self.frame_features = None
        self.target_visible = False

    def load_station_reference(self, directory):
        directory = Path(directory)
        path = directory / "approved-standard.json"
        if not path.exists():
            return False
        metadata = json.loads(path.read_text(encoding="utf-8"))
        frame = cv2.imread(str(directory / "approved-complete-board.jpg"))
        if frame is None:
            raise ValueError("完整板标准原图无法读取")
        items = {item["id"]: item for item in metadata["items"]}
        source = np.float32([point for code, _, _ in REGIONS for point in items[code]["polygon"]])
        target = np.float32([point for _, _, bounds in REGIONS for point in polygon(bounds)])
        matrix, _ = cv2.findHomography(source, target, 0)
        if matrix is None or not np.isfinite(matrix).all():
            raise ValueError("完整板标准定位坐标无效")
        mask = np.full(self.bare.shape[:2], 255, np.uint8)
        mask[98:190, 449:744] = 0
        mask = cv2.warpPerspective(mask, np.linalg.inv(matrix), (frame.shape[1], frame.shape[0]), flags=cv2.INTER_NEAREST)
        with self.lock:
            features = self.detector.detectAndCompute(feature_gray(frame), mask)
            if features[1] is None or len(features[0]) < 12:
                raise ValueError("完整板标准缺少有效定位特征")
            self.feature_matchers.clear()
            self.station_reference = (frame.shape, features, matrix)
            self.station_posts = {code: transform(items[code]["polygon"], matrix) for code, _, _ in POST_BOXES}
            self.station_regions = {code: transform(items[code]["polygon"], matrix) for code, _, _ in REGIONS + POST_BOXES}
            self.station_source_regions = {code: np.float32(items[code]["polygon"]) for code, _, _ in REGIONS + POST_BOXES}
            self.station_gray = cv2.GaussianBlur(gray(frame), (3, 3), 0)
        return True

    def load_ocr(self, threads=2):
        try:
            from rapidocr_onnxruntime import RapidOCR

            self.ocr = RapidOCR(intra_op_num_threads=threads, inter_op_num_threads=threads)
            latin_model = Path(__file__).resolve().parent / "models" / "en_PP-OCRv3_rec_infer.onnx"
            if latin_model.exists():
                try:
                    self.latin_ocr = RapidOCR(rec_model_path=str(latin_model), intra_op_num_threads=threads, inter_op_num_threads=threads)
                except Exception as error:
                    self.latin_ocr = None
                    print(f"[ocr-warning] Latin fallback unavailable: {type(error).__name__}", flush=True)
            self.ocr_error = ""
        except Exception as error:
            self.ocr_error = f"OCR 不可用：{type(error).__name__}"

    def scaled_features(self, image, mask):
        size = (round(image.shape[1] * 0.4), round(image.shape[0] * 0.4))
        scaled = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        scaled_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        keypoints, descriptors = self.detector.detectAndCompute(feature_gray(scaled), scaled_mask)
        for keypoint in keypoints:
            keypoint.pt = (keypoint.pt[0] * image.shape[1] / size[0], keypoint.pt[1] * image.shape[0] / size[1])
        return keypoints, descriptors

    def align(self, image):
        self.target_visible = False
        keypoints, descriptors = self.detector.detectAndCompute(feature_gray(image), None)
        self.frame_features = (keypoints, descriptors)
        if descriptors is None or len(keypoints) < 12:
            return None, {"matches": 0, "inliers": 0, "ratio": 0}
        if self.station_reference is not None:
            shape, features, reference_matrix = self.station_reference
            reference_outline = transform(polygon((0, 0, self.bare.shape[1], self.bare.shape[0])), np.linalg.inv(reference_matrix))
            matrix, quality = self._match_features(keypoints, descriptors, features, image.shape, shape, reference_outline)
            if matrix is not None:
                quality["reference"] = "approved_complete_board"
                return reference_matrix @ matrix, quality
        matrix, quality = self._match_features(keypoints, descriptors, self.reference_features, image.shape)
        quality["reference"] = "bare"
        if matrix is not None:
            return matrix, quality
        matrix, live_quality = self._match_features(keypoints, descriptors, self.live_reference_features, image.shape)
        live_quality["reference"] = "bare_camera_scale"
        if matrix is not None:
            return matrix, live_quality
        if live_quality["matches"] > quality["matches"]:
            quality = live_quality
        if hasattr(self, "post_features"):
            secondary, secondary_quality = self._match_features(keypoints, descriptors, self.post_features, image.shape, self.posts.shape)
            secondary_quality["reference"] = "installed_manual_calibration"
            if secondary is not None:
                return self.post_matrix @ secondary, secondary_quality
            secondary, secondary_quality = self._match_features(keypoints, descriptors, self.live_post_features, image.shape, self.posts.shape)
            secondary_quality["reference"] = "installed_camera_scale"
            if secondary is not None:
                return self.post_matrix @ secondary, secondary_quality
            if secondary_quality["matches"] > quality["matches"]:
                quality = secondary_quality
        if self.tracked_reference is not None:
            shape, features, reference_matrix = self.tracked_reference
            reference_outline = transform(polygon((0, 0, self.bare.shape[1], self.bare.shape[0])), np.linalg.inv(reference_matrix))
            tracked_matrix, tracked_quality = self._match_features(keypoints, descriptors, features, image.shape, shape, reference_outline)
            if tracked_matrix is not None:
                tracked_quality["reference"] = "tracked_board"
                return reference_matrix @ tracked_matrix, tracked_quality
            if tracked_quality["matches"] > quality["matches"]:
                tracked_quality["reference"] = "tracked_board"
                quality = tracked_quality
        return None, quality

    def remember_target(self, image, matrix):
        if self.frame_features is None or self.frame_features[1] is None:
            return
        keypoints, descriptors = self.frame_features
        outline = transform(polygon((0, 0, self.bare.shape[1], self.bare.shape[0])), np.linalg.inv(matrix))
        selected = [index for index, keypoint in enumerate(keypoints) if cv2.pointPolygonTest(outline, keypoint.pt, False) >= 0]
        if len(selected) < 12:
            return
        if self.tracked_reference is not None:
            self.feature_matchers.pop(id(self.tracked_reference[1][1]), None)
        self.tracked_reference = (image.shape, ([keypoints[index] for index in selected], descriptors[selected].copy()), matrix.copy())

    def _match_features(self, keypoints, descriptors, reference_features, image_shape, reference_shape=None, reference_outline=None):
        reference_points, reference_descriptors = reference_features
        identity = id(reference_descriptors)
        if identity not in self.feature_matchers:
            matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 64})
            matcher.add([reference_descriptors])
            matcher.train()
            self.feature_matchers[identity] = matcher
        matches = self.feature_matchers[identity].knnMatch(descriptors, k=2)
        good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance]
        unique = {}
        for match in sorted(good, key=lambda match: match.distance):
            unique.setdefault(match.trainIdx, match)
        good = list(unique.values())
        quality = {"matches": len(good), "inliers": 0, "ratio": 0}
        if len(good) < 6:
            return None, quality
        source = np.float32([keypoints[match.queryIdx].pt for match in good])
        target = np.float32([reference_points[match.trainIdx].pt for match in good])
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        matrix, mask = cv2.findHomography(source, target, method, 4.0, maxIters=10000, confidence=0.999)
        if matrix is None or mask is None:
            return None, quality
        count = int(mask.sum())
        quality.update(inliers=count, ratio=round(count / len(good), 3))
        height, width = (reference_shape or self.bare.shape)[:2]
        try:
            outline = transform(polygon((0, 0, width, height)) if reference_outline is None else reference_outline, np.linalg.inv(matrix))
        except np.linalg.LinAlgError:
            return None, quality
        area = abs(cv2.contourArea(outline))
        image_area = image_shape[0] * image_shape[1]
        if not cv2.isContourConvex(outline) or area < image_area * 0.08 or area > image_area * 1.8:
            return None, quality
        if count >= 6 and quality["ratio"] >= 0.60:
            self.target_visible = True
        if count < 12 or quality["ratio"] < 0.35:
            return None, quality
        return matrix, quality

    def model_crop(self, image):
        left, top, right, bottom = (510, 80, 745, 205)
        search = gray(image[top:bottom, left:right])
        template = gray(self.bare[MODEL_BOUNDS[1]:MODEL_BOUNDS[3], MODEL_BOUNDS[0]:MODEL_BOUNDS[2]])
        best_score = 0.45
        best_bounds = None
        for height in (30, 36, 42, 50, 58):
            width = round(height * template.shape[1] / template.shape[0])
            if height > search.shape[0] or width > search.shape[1]:
                continue
            sample = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
            _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(search, sample, cv2.TM_CCOEFF_NORMED))
            if score > best_score:
                best_score = score
                horizontal, vertical = location
                padding_x, padding_y = round(width * 0.18), round(height * 0.18)
                best_bounds = (max(0, left + horizontal + round(width * 0.1)), max(0, top + vertical - padding_y), min(image.shape[1], left + horizontal + width + padding_x), min(image.shape[0], top + vertical + height + padding_y))
        return best_bounds

    def extract_model_text(self, image, full_frame):
        options = {"use_cls": False} if full_frame else {"use_det": False, "use_cls": False}
        results, _ = self.ocr(image, **options)
        recognized = lambda entries: any(float(entry[-1]) >= 0.70 and re.search(r"(?:\b|QUECTEL\s*PI\s*)[MHL]\s*\d+\b", entry[-2], re.I) for entry in entries or [])
        if recognized(results):
            return results, "original", []
        enhanced = enhance_text(image)
        improved, _ = self.ocr(enhanced, **options)
        if recognized(improved):
            return improved, "contrast_upscale_sharpen", []
        consensus = []
        weak_m2 = [entry for entry in improved or [] if float(entry[-1]) >= 0.50 and re.fullmatch(r"M\s*2", entry[-2].strip(), re.I)]
        if self.latin_ocr is not None and weak_m2:
            if full_frame:
                points = np.float32(weak_m2[0][0])
                if points.shape != (4, 2):
                    return results, "original", []
                left, top = np.maximum(0, np.floor(points.min(axis=0)).astype(int))
                right, bottom = np.minimum([enhanced.shape[1], enhanced.shape[0]], np.ceil(points.max(axis=0)).astype(int))
                latin_input = enhanced[top:bottom, left:right]
            else:
                latin_input = enhanced
            if latin_input.size:
                latin_results, _ = self.latin_ocr(latin_input, use_det=False, use_cls=False)
                consensus = [{"text": entry[-2], "score": round(float(entry[-1]), 3)} for entry in latin_results or []]
                if any(token["score"] >= 0.50 and re.fullmatch(r"M\s*2", token["text"].strip(), re.I) for token in consensus):
                    return improved, "dual_model_enhanced", consensus
        return results, "original", []

    def read_version(self, aligned, full_frame=False):
        started = time.monotonic()
        bounds = (0, 0, aligned.shape[1], aligned.shape[0]) if full_frame else MODEL_BOUNDS
        if self.ocr is None:
            return {"status": "review", "text": "未识别", "raw": [], "reason": self.ocr_error, "bounds": bounds}
        left, top, right, bottom = bounds
        crop = aligned[top:bottom, left:right]
        prepared = crop
        ocr_rotation = 0
        if full_frame:
            scale = min(1.0, 640 / max(prepared.shape[:2]))
            prepared = cv2.resize(prepared, None, fx=scale, fy=scale)
            ocr_rotation = (self.ocr_rotation_index % 4) * 90
            self.ocr_rotation_index += 1
            prepared = np.ascontiguousarray(np.rot90(prepared, ocr_rotation // 90))
            results, preprocessing, consensus = self.extract_model_text(prepared, True)
        else:
            bounds = self.model_crop(aligned) or MODEL_BOUNDS
            left, top, right, bottom = bounds
            prepared = aligned[top:bottom, left:right]
            results, preprocessing, consensus = self.extract_model_text(prepared, False)
        tokens = [{"text": entry[-2], "score": round(float(entry[-1]), 3)} for entry in results or []]
        text = " / ".join(token["text"] for token in tokens)
        credible = [token["text"] for token in tokens if token["score"] >= 0.85]
        candidate_m2 = any(token["score"] >= 0.70 and re.search(r"(?:\b|QUECTEL\s*PI\s*)M\s*2\b", token["text"], re.I) for token in tokens)
        candidate_m2 = candidate_m2 or preprocessing == "dual_model_enhanced"
        if not full_frame:
            credible = [re.sub(r"^[IP](?=M\s*2$)", "", value.strip(), flags=re.I) for value in credible]
        model_text = " ".join(credible)
        model_match = re.search(r"(?:\b|QUECTEL\s*PI\s*)([MHL]\s*\d+)\b|\b(?:BOARD|PI|MODULE)\s+([A-Z][A-Z0-9-]+)\b", model_text.upper())
        model = re.sub(r"\s+", "", next((part for part in model_match.groups() if part), "")) if model_match else ""
        brand_match = re.search(r"Quectel\s*Pi\s*[MHL]\s*\d+\b", model_text, re.I)
        if brand_match:
            model = re.sub(r"\s+", "", re.search(r"[MHL]\s*\d+\b", brand_match.group(0), re.I).group(0)).upper()
        device_name = brand_match.group(0) if brand_match else next((value for value in credible if re.search(r"\b[MHL]\s*\d+\b|\b(?:pi|board|module|RK\d{4})\b|Quectel|Raspberry|开发板", value, re.I)), "")
        confident = bool(device_name and model)
        if full_frame and candidate_m2 and not model:
            self.ocr_rotation_index -= 1
        return {
            "status": "match" if confident else "review",
            "text": device_name if model and device_name else model or "未识别",
            "device_name": device_name,
            "board_revision": "",
            "model": model,
            "candidate_model": "M2" if candidate_m2 and not model else "",
            "ocr_preprocessing": preprocessing,
            "ocr_consensus": consensus,
            "raw": tokens,
            "ocr_rotation": ocr_rotation,
            "ocr_ms": round((time.monotonic() - started) * 1000),
            "reason": "型号已识别" if confident else "正在识别型号",
            "bounds": bounds,
        }

    def confirm_version(self, version):
        if version.get("model") or version.get("candidate_model") != "M2":
            self.ocr_candidate_count = 0
            return version
        self.ocr_candidate_count += 1
        if self.ocr_candidate_count >= 2:
            version = {**version, "status": "match", "text": "M2", "device_name": "M2", "model": "M2", "reason": "连续两帧读出 M2 字样", "confirmation_frames": 2}
        return version

    def current_version(self, image, matrix):
        now = time.monotonic()
        located = matrix is not None
        if self.ocr_thread is not None:
            with self.ocr_condition:
                location_changed = self.ocr_located != located
                self.ocr_located = located
                model_known = bool(self.ocr_cached and self.ocr_cached.get("model"))
                retry_interval = min(self.ocr_interval, 0.2 if located else 0.5)
                if not model_known and not self.ocr_busy and self.ocr_job is None and (self.ocr_cached is None or location_changed or now - self.ocr_sampled_at >= retry_interval):
                    self.ocr_job = (image.copy(), matrix.copy() if located else None, self.ocr_generation, now)
                    self.ocr_condition.notify_all()
                version = dict(self.ocr_cached or {"status": "review", "text": "未识别", "device_name": "", "board_revision": "", "raw": [], "reason": "正在读取丝印"})
                sampled_at = self.ocr_sampled_at if self.ocr_cached else now
        elif not (self.ocr_cached and self.ocr_cached.get("model")) and (self.ocr_cached is None or self.ocr_located != located or now - self.ocr_sampled_at >= self.ocr_interval):
            aligned = cv2.warpPerspective(image, matrix, (self.bare.shape[1], self.bare.shape[0])) if located else image
            self.ocr_cached = self.confirm_version(self.read_version(aligned, full_frame=not located))
            self.ocr_sampled_at = time.monotonic()
            self.ocr_located = located
            version = dict(self.ocr_cached)
            sampled_at = self.ocr_sampled_at
        else:
            version = dict(self.ocr_cached)
            sampled_at = self.ocr_sampled_at
        bounds = version.pop("bounds", MODEL_BOUNDS)
        if located and (bounds[2] - bounds[0] > 150 or bounds[3] - bounds[1] > 100):
            bounds = MODEL_BOUNDS
        if not located:
            bounds = (0, 0, image.shape[1], image.shape[0])
        version["polygon"] = transform(polygon(bounds), np.linalg.inv(matrix)).round(1).tolist() if located else polygon(bounds).tolist()
        version["age_ms"] = round((time.monotonic() - sampled_at) * 1000)
        if version.get("model") and not self.board_seen:
            self.board_seen = True
            self.board_level = float(cv2.resize(gray(image), (80, 60), interpolation=cv2.INTER_AREA).mean())
        return version

    def reset_identity(self):
        with self.ocr_condition:
            self.ocr_generation += 1
            self.ocr_cached = None
            self.ocr_sampled_at = 0
            self.ocr_job = None
            self.ocr_rotation_index = 0
            self.ocr_candidate_count = 0
            self.board_seen = False
            self.target_lost_since = None
            if self.tracked_reference is not None:
                self.feature_matchers.pop(id(self.tracked_reference[1][1]), None)
                self.tracked_reference = None

    def start_ocr_worker(self):
        self.ocr_thread = threading.Thread(target=self._ocr_loop, name="silkscreen-ocr", daemon=True)
        self.ocr_thread.start()

    def _ocr_loop(self):
        while True:
            with self.ocr_condition:
                self.ocr_condition.wait_for(lambda: self.ocr_stopping or self.ocr_job is not None)
                if self.ocr_stopping:
                    return
                image, matrix, generation, sampled_at = self.ocr_job
                self.ocr_job = None
                self.ocr_busy = True
            try:
                aligned = cv2.warpPerspective(image, matrix, (self.bare.shape[1], self.bare.shape[0])) if matrix is not None else image
                version = self.read_version(aligned, full_frame=matrix is None)
            except Exception as error:
                print(f"[ocr-error] {type(error).__name__}: {error}", flush=True)
                version = {"status": "review", "text": "未识别", "raw": [], "reason": "丝印识别失败，请检查画面或服务日志"}
            with self.ocr_condition:
                if generation == self.ocr_generation:
                    self.ocr_cached = self.confirm_version(version)
                    self.ocr_sampled_at = sampled_at
                self.ocr_busy = False
                self.ocr_condition.notify_all()

    def close_ocr_worker(self):
        with self.ocr_condition:
            self.ocr_stopping = True
            self.ocr_condition.notify_all()
        if self.ocr_thread is not None:
            self.ocr_thread.join(timeout=5)

    def board_presence(self, image, located, now=None, target_visible=False):
        now = time.monotonic() if now is None else now
        if located or target_visible:
            self.board_seen = True
            self.board_level = float(cv2.resize(gray(image), (80, 60), interpolation=cv2.INTER_AREA).mean())
            self.empty_since = None
            self.empty_frames = 0
            self.empty_gray = None
            self.target_lost_since = None
            return "present"
        if self.board_seen:
            if self.target_lost_since is None:
                self.target_lost_since = now
            if now - self.target_lost_since >= 3.0:
                self.board_seen = False
                self.board_level = None
                self.target_lost_since = None
                self.empty_since = None
                self.empty_frames = 0
                self.empty_gray = None
                self.reset_identity()
                return "absent"
        return "unknown"

    def inspect(self, image):
        with self.lock:
            return self._inspect(image)

    def _inspect(self, image):
        started = time.monotonic()
        matrix, quality = self.align(image)
        if matrix is not None:
            self.remember_target(image, matrix)
        result = {
            "schema": 1,
            "method": "SIFT 定位 + 局部参考图比对 + 铜柱样本比对 + OCR（非训练模型）",
            "items": [],
            "alignment": quality,
            "located": matrix is not None,
            "board_presence": self.board_presence(image, matrix is not None, target_visible=self.target_visible),
            "width": image.shape[1],
            "height": image.shape[0],
            "version": {"status": "review", "text": "未识别", "raw": [], "reason": "未定位到开发板"},
            "verdict": "待复核",
            "notice": "参考一致不等于硬件功能正常；损坏与缺失结论需复核，当前未完成现场精度验收。",
        }
        if matrix is None:
            for code, name, _ in REGIONS + POST_BOXES:
                result["items"].append({"id": code, "name": name, "status": "review", "reason": "开发板定位失败，请固定板位、移除遮挡或重新采图", "polygon": [], "score": None})
            if self.ocr_interval > 0:
                result["version"] = self.current_version(image, None)
            result["duration_ms"] = round((time.monotonic() - started) * 1000)
            return result

        if self.calibrated:
            inverse = np.linalg.inv(matrix)
            station_mapping = None
            if self.station_reference is not None and quality.get("reference") == "approved_complete_board":
                shape, _, reference_matrix = self.station_reference
                station_mapping = inverse @ reference_matrix
                aligned_gray = cv2.warpPerspective(cv2.GaussianBlur(gray(image), (3, 3), 0), np.linalg.inv(station_mapping), (shape[1], shape[0]))
            for code, name, bounds in REGIONS + POST_BOXES:
                mapping = inverse if code.startswith("A") else inverse @ self.post_matrix
                points = transform(self.station_regions[code], inverse) if code in self.station_regions else transform(polygon(bounds), mapping)
                refinement = None
                if station_mapping is not None:
                    corrected, refinement = refine_region(self.station_gray, aligned_gray, self.station_source_regions[code])
                    points = transform(corrected, station_mapping)
                result["items"].append({"id": code, "name": name, "status": "review", "reason": "等待工位判定", "score": None, "polygon": points.round(1).tolist(), "local_alignment": refinement})
            result["version"] = self.current_version(image, matrix)
            result["duration_ms"] = round((time.monotonic() - started) * 1000)
            return result

        height, width = self.bare.shape[:2]
        aligned = cv2.warpPerspective(image, matrix, (width, height))
        coverage = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8), matrix, (width, height))
        inverse = np.linalg.inv(matrix)
        for code, name, bounds in REGIONS:
            left, top, right, bottom = bounds
            sample = aligned[top:bottom, left:right]
            reference = self.bare[top:bottom, left:right]
            visible = float(np.mean(coverage[top:bottom, left:right] > 0))
            score = similarity(reference, sample)
            status = "match" if score >= 0.72 and visible > 0.97 else "review"
            reason = "外观与空板参考相近，非功能测试" if status == "match" else "外观变化：可能遮挡、插接、模糊或器件差异，需复核"
            if visible <= 0.97:
                reason = "检查区域不完整，请重新采图"
            result["items"].append({"id": code, "name": name, "status": status, "score": round(score, 3), "reason": reason, "polygon": transform(polygon(bounds), inverse).round(1).tolist()})

        if self.post_matrix is not None:
            post_height, post_width = self.posts.shape[:2]
            to_post = np.linalg.inv(self.post_matrix) @ matrix
            observed_posts = cv2.warpPerspective(image, to_post, (post_width, post_height))
            bare_posts = cv2.warpPerspective(self.bare, np.linalg.inv(self.post_matrix), (post_width, post_height))
            post_coverage = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8), to_post, (post_width, post_height))
            from_post = inverse @ self.post_matrix
            for code, name, bounds in POST_BOXES:
                left, top, right, bottom = bounds
                positive = self.posts[top:bottom, left:right]
                negative = bare_posts[top:bottom, left:right]
                observed = observed_posts[top:bottom, left:right]
                positive_score = similarity(positive, observed)
                negative_score = similarity(negative, observed)
                visible = float(np.mean(post_coverage[top:bottom, left:right] > 0))
                present = visible > 0.97 and positive_score >= 0.70 and positive_score - negative_score > 0.16
                absent = visible > 0.97 and negative_score >= 0.78 and negative_score - positive_score > 0.20
                status = "match" if present else "suspect" if absent else "review"
                reason = "与安装铜柱样本相近，待现场验证" if present else "疑似缺失：更接近未安装参考" if absent else "铜柱证据不足：位置、遮挡或清晰度变化，请复核"
                if visible <= 0.97:
                    reason = "完整铜柱区域超出画面，无法确认有无，请扩大取景"
                result["items"].append({"id": code, "name": name, "status": status, "score": round(positive_score, 3), "absence_score": round(negative_score, 3), "reason": reason, "polygon": transform(polygon(bounds), from_post).round(1).tolist()})
        else:
            for code, name, _ in POST_BOXES:
                result["items"].append({"id": code, "name": name, "status": "review", "score": None, "reason": "铜柱参考图无法定位", "polygon": []})

        result["version"] = self.current_version(image, matrix)
        result["duration_ms"] = round((time.monotonic() - started) * 1000)
        return result