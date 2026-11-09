import argparse
import copy
import json
import mimetypes
import os
import pwd
import shutil
import sqlite3
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from decisions import CalibratedDecisions
from engine import BARE_PHOTO, POST_PHOTO, REGIONS, InspectionEngine
from live import DeviceCamera, LiveInspection


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
MAX_BYTES = 16 * 1024 * 1024
MAX_PIXELS = 24_000_000


class ResultAnnouncer:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.enabled = True
        self.available = bool(shutil.which("paplay") and all((self.directory / name).is_file() for name in ["qualified.wav", "rejected.wav"]))
        self.error = "" if self.available else "离线音频或播放器未就绪"
        self.seen = set()
        self.thread = None

    def state(self):
        return {"enabled": self.enabled, "available": self.available, "error": self.error}

    def announce(self, identity, outcome):
        if not self.enabled or not self.available or identity in self.seen:
            return
        self.seen.add(identity)
        self.thread = threading.Thread(target=self._play, args=(outcome,), daemon=True)
        self.thread.start()

    def _play(self, outcome):
        path = self.directory / ("qualified.wav" if outcome == "pass" else "rejected.wav")
        volume = 81920 if outcome == "pass" else 65536
        command = ["paplay", f"--volume={volume}", str(path)]
        try:
            if os.geteuid() == 0:
                for socket in sorted(Path("/run/user").glob("*/pulse/native")):
                    identity = int(socket.parents[1].name)
                    if identity >= 1000:
                        username = pwd.getpwuid(identity).pw_name
                        command = ["runuser", "-u", username, "--", "env", f"XDG_RUNTIME_DIR=/run/user/{identity}", f"PULSE_SERVER=unix:{socket}"] + command
                        break
            subprocess.run(command, check=True, timeout=8, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            self.error = ""
        except (OSError, subprocess.SubprocessError) as error:
            self.error = f"音频播放失败：{type(error).__name__}"


class ProductionLedger:
    def __init__(self, path):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        with self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS cycles (number INTEGER PRIMARY KEY AUTOINCREMENT, cycle_id TEXT UNIQUE NOT NULL, report TEXT NOT NULL)")
            self.db.execute("CREATE TABLE IF NOT EXISTS rechecks (id TEXT PRIMARY KEY, parent_id TEXT NOT NULL, report TEXT NOT NULL)")

    def record(self, report):
        if report.get("inspection_scope") != "board_cycle" or not report.get("cycle_complete"):
            raise ValueError("仅完整检测轮次可计入板数")
        with self.lock, self.db:
            existing = self.db.execute("SELECT report FROM cycles WHERE cycle_id=?", (report["cycle_id"],)).fetchone()
            if existing:
                return json.loads(existing[0])
            saved = copy.deepcopy(report)
            cursor = self.db.execute("INSERT INTO cycles(cycle_id,report) VALUES (?,?)", (report["cycle_id"], "{}"))
            saved["board_number"] = cursor.lastrowid
            self.db.execute("UPDATE cycles SET report=? WHERE number=?", (json.dumps(saved, ensure_ascii=False), cursor.lastrowid))
            return saved

    def record_recheck(self, report):
        if not report.get("parent_report_id"):
            return
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO rechecks(id,parent_id,report) VALUES (?,?,?)", (report["id"], report["parent_report_id"], json.dumps(report, ensure_ascii=False)))

    def last_completed(self):
        with self.lock:
            row = self.db.execute("SELECT report FROM cycles ORDER BY number DESC LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None

    def summary(self):
        with self.lock:
            boards = [json.loads(row[0]) for row in self.db.execute("SELECT report FROM cycles ORDER BY number")]
            rechecks = [json.loads(row[0]) for row in self.db.execute("SELECT report FROM rechecks ORDER BY rowid")]
        by_id = {board["id"]: board for board in boards}
        for board in boards:
            board["initial_outcome"] = board["outcome"]
            board["rechecks"] = []
        for report in rechecks:
            board = by_id.get(report.get("parent_report_id"))
            if board is None or report.get("inspection_scope") != "checkpoint" or len(report.get("items", [])) != 1:
                continue
            selected = report["items"][0]
            for index, item in enumerate(board["items"]):
                if item["id"] == selected["id"]:
                    board["items"][index] = copy.deepcopy(selected)
                    board["rechecks"].append({"id": report["id"], "checkpoint": selected["id"], "status": selected["status"], "created_at": report.get("created_at")})
                    board["outcome"] = "pass" if all(entry["status"] == "pass" for entry in board["items"]) and not board.get("cycle_interrupted") else "fail"
        total = len(boards)
        passed = sum(board["outcome"] == "pass" for board in boards)
        first_passed = sum(board["initial_outcome"] == "pass" for board in boards)
        return {"total": total, "passed": passed, "failed": total - passed, "first_passed": first_passed, "pass_rate": passed / total if total else None, "first_pass_rate": first_passed / total if total else None, "boards": boards}


def production_table(summary):
    def cells(values, tag="td"):
        return "<tr>" + "".join(f"<{tag}>{escape(str(value))}</{tag}>" for value in values) + "</tr>"

    def rate(value):
        return f"{value:.1%}" if value is not None else "暂无数据"

    totals = cells(["累计检测", "合格", "未通过", "首次通过率", "复检后通过率"], "th") + cells([summary["total"], summary["passed"], summary["failed"], rate(summary["first_pass_rate"]), rate(summary["pass_rate"])])
    rows = []
    for board in summary["boards"]:
        def components(status):
            return "、".join(f"{item['id']} {item['name']}" for item in board["items"] if item["status"] == status) or "无"
        rows.append(cells([board["board_number"], board.get("completed_at", ""), "合格" if board["initial_outcome"] == "pass" else "未通过", "合格" if board["outcome"] == "pass" else "未通过", components("missing"), components("damaged"), components("unavailable"), "是" if board.get("cycle_interrupted") else "否", len(board["rechecks"])]))
    return ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>M2 批次检测统计</title>"
            "<style>body{font-family:sans-serif;color:#17251f;margin:24px}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #87978f;padding:8px;text-align:left;overflow-wrap:anywhere}th{background:#edf3ef}.scroll{overflow:auto}@media print{body{margin:0}thead{display:table-header-group}tr{break-inside:avoid}}</style>"
            f"<h1>M2 批次检测统计</h1><table aria-label='良品率汇总'>{totals}</table><div class='scroll'><table aria-label='逐板检测结果'><thead>"
            + cells(["板号", "完成时间", "首次结论", "复检后结论", "明确缺失", "明确损坏", "未确认模块", "检测中断", "复检次数"], "th")
            + f"</thead><tbody>{''.join(rows)}</tbody></table></div><p>全部16项通过立即结算，最长等待20秒，每轮只计一块；超时未确认也计入未通过。单模块复检不增加板数。低匹配分不直接认定缺失。首次结论保留，复检后结论单独统计。空批次不显示虚假的100%。</p></html>").encode("utf-8")


def persist_report(image, report):
    report = dict(report)
    stamp = datetime.now(timezone.utc)
    identity = stamp.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    if report.get("inspection_scope") == "board_cycle":
        identity = "cycle-" + uuid.UUID(report["cycle_id"]).hex
    report.update(id=identity, created_at=stamp.isoformat(), image_url=f"/output/{identity}.jpg", annotated_url=f"/output/{identity}-annotated.jpg", report_url=f"/output/{identity}.json", table_url=f"/output/{identity}.html", live=False)
    OUTPUT.mkdir(exist_ok=True)
    if not cv2.imwrite(str(OUTPUT / f"{identity}.jpg"), image) or not cv2.imwrite(str(OUTPUT / f"{identity}-annotated.jpg"), annotate(image, report)):
        raise OSError("无法保存图像")
    (OUTPUT / f"{identity}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def decode_image(data):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图片无法解码")
    height, width = image.shape[:2]
    if height * width > MAX_PIXELS:
        raise ValueError("图片超过 2400 万像素，请先缩小")
    if max(height, width) > 2000:
        scale = 2000 / max(height, width)
        image = cv2.resize(image, (round(width * scale), round(height * scale)))
    return image


def annotate(image, report):
    annotated = image.copy()
    colors = {"match": (112, 145, 12), "review": (25, 167, 224), "suspect": (70, 65, 217), "pass": (112, 145, 12), "missing": (70, 65, 217), "damaged": (70, 65, 217), "unavailable": (25, 167, 224)}
    for item in report["items"]:
        if not item["polygon"]:
            continue
        points = np.int32(item["polygon"])
        color = colors[item["status"]]
        cv2.polylines(annotated, [points], True, color, 2, cv2.LINE_AA)
        left, top = points.min(axis=0)
        cv2.putText(annotated, item["id"], (max(0, left), max(20, top)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated


def table_report(report):
    labels = {"pass": "通过", "match": "参考一致", "missing": "缺失", "damaged": "损坏", "unavailable": "无法检测", "review": "待复核", "suspect": "疑似缺失", "pending": "待检测"}
    rows = []
    for item in report.get("items", []):
        score = item.get("scores", {}).get("pass")
        review = report.get("manual_reviews", {}).get(item["id"], {}).get("verdict", "未复核")
        values = [item["id"], item["name"], labels.get(item["status"], item["status"]), labels.get(item.get("observed_status"), "未记录"), f"{score:.1%}" if score is not None else "未记录", item.get("reason", ""), review]
        rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in values) + "</tr>")
    scope = "单模块复检" if report.get("inspection_scope") == "checkpoint" else "整板检测"
    fields = [("报告编号", report.get("id", "")), ("检测时间", report.get("created_at", "")), ("检测范围", scope), ("板卡型号", report.get("version", {}).get("model", "未识别")), ("结果", report.get("verdict", "")), ("检测帧号", report.get("sequence", "未记录"))]
    metadata = "".join(f"<tr><th>{escape(title)}</th><td>{escape(str(value))}</td></tr>" for title, value in fields)
    return ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>板卡检测报告</title><style>body{font-family:sans-serif;margin:24px;color:#17251f}h1{font-size:22px}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #87978f;padding:8px;text-align:left;overflow-wrap:anywhere}th{background:#edf3ef}p{font-size:12px}@media print{body{margin:0}thead{display:table-header-group}tr{break-inside:avoid}}</style>"
            f"<h1>板卡检测报告 · {scope}</h1><table aria-label='报告信息'><tbody>{metadata}</tbody></table>"
            "<table aria-label='模块检测结果'><thead><tr><th>编号</th><th>模块</th><th>最终结果</th><th>当帧证据</th><th>正常匹配分</th><th>判定原因</th><th>人工复核</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table><p>正常匹配分不是模型置信度。外观检查不代表电气功能正常；保留结果与当帧证据分列展示。</p></html>").encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    def json_response(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def file_response(self, path, download=False):
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/api/preview.mjpg":
            live = getattr(self.server, "live", None)
            if live is None:
                self.json_response({"error": "未启用设备相机"}, 409)
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            sequence = -1
            try:
                while not live.stop_event.is_set():
                    packet = live.preview(sequence)
                    if packet is None:
                        continue
                    sequence, data = packet
                    header = f"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\nX-Sequence: {sequence}\r\n\r\n".encode("ascii")
                    self.wfile.write(header + data + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            self.close_connection = True
            return
        if route in {"/api/production", "/api/production/export.html"}:
            ledger = getattr(self.server, "ledger", None)
            summary = ledger.summary() if ledger else {"total": 0, "passed": 0, "failed": 0, "first_passed": 0, "pass_rate": None, "first_pass_rate": None, "boards": []}
            if route == "/api/production":
                self.json_response(summary)
            else:
                data = production_table(summary)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="M2-production.html"')
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
                self.end_headers()
                self.wfile.write(data)
        elif route == "/api/config":
            live = getattr(self.server, "live", None)
            self.json_response({"app_version": "0.3.2", "device_camera": live.camera.source if live else None, "ocr_ready": self.server.engine.ocr is not None, "checkpoints": [{"id": code, "name": name, "bounds": bounds} for code, name, bounds in REGIONS], "samples": [{"id": "bare", "name": "清晰空板参考", "url": "/media/bare.jpg"}, {"id": "posts", "name": "铜柱实拍样本", "url": "/media/posts.jpg"}]})
        elif route in {"/api/live", "/api/live/frame.jpg", "/api/calibration"}:
            live = getattr(self.server, "live", None)
            if live is None:
                self.json_response({"error": "未启用设备相机"}, 409)
            elif route == "/api/live":
                status = live.status()
                announcer = getattr(self.server, "announcer", None)
                if announcer:
                    status["audio"] = announcer.state()
                self.json_response(status)
            elif route == "/api/calibration":
                self.json_response(live.decisions.state())
            else:
                try:
                    sequence = int(parse_qs(urlparse(self.path).query).get("sequence", ["-1"])[0])
                except ValueError:
                    self.send_error(400)
                    return
                snapshot = live.snapshot(sequence)
                if snapshot is None:
                    self.send_error(410, "Frame expired")
                    return
                data = snapshot[1]
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
        elif route == "/api/history":
            reports = []
            for path in sorted(OUTPUT.glob("*.json"), reverse=True)[:30]:
                try:
                    report = json.loads(path.read_text(encoding="utf-8"))
                    reports.append({key: report[key] for key in ["id", "created_at", "verdict", "image_url", "version"]})
                except (OSError, ValueError, KeyError):
                    continue
            self.json_response(reports)
        elif route in {"/media/bare.jpg", "/media/posts.jpg"}:
            self.file_response(BARE_PHOTO if "bare" in route else POST_PHOTO)
        elif route == "/vendor/lucide.js":
            self.file_response(ROOT / "static/lucide.min.js")
        elif route.startswith("/output/"):
            name = route.removeprefix("/output/")
            if "/" in name or "\\" in name or not name or Path(name).suffix not in {".jpg", ".json", ".html"}:
                self.send_error(404)
                return
            if Path(name).suffix == ".html":
                path = OUTPUT / Path(name).with_suffix(".json")
                try:
                    data = table_report(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, ValueError, KeyError, TypeError):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
                self.send_header("X-Content-Type-Options", "nosniff")
                if parse_qs(urlparse(self.path).query).get("download") == ["1"]:
                    self.send_header("Content-Disposition", f'attachment; filename="{Path(name).name}"')
                self.end_headers()
                self.wfile.write(data)
                return
            self.file_response(OUTPUT / name, download=parse_qs(urlparse(self.path).query).get("download") == ["1"])
        else:
            routes = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            if route not in routes:
                self.send_error(404)
                return
            self.file_response(ROOT / "static" / routes[route])

    def do_POST(self):
        route = urlparse(self.path).path
        if route == "/api/audio/control":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 128:
                    raise ValueError("播报请求无效")
                payload = json.loads(self.rfile.read(length))
                if set(payload) != {"enabled"} or not isinstance(payload["enabled"], bool):
                    raise ValueError("播报开关无效")
                announcer = getattr(self.server, "announcer", None)
                if announcer is None:
                    raise ValueError("音频未就绪")
                announcer.enabled = payload["enabled"]
                self.json_response(announcer.state())
            except (ValueError, KeyError, TypeError) as error:
                self.json_response({"error": str(error)}, 400)
            return
        if route in {"/api/live/control", "/api/live/reset", "/api/live/recheck", "/api/calibrate", "/api/live/save", "/api/camera/control"}:
            self.live_action(route)
            return
        if route == "/api/review":
            self.save_review()
            return
        if route != "/api/inspect":
            self.send_error(404)
            return
        if self.headers.get("X-Inspection-Mode", "empty") != "empty":
            self.json_response({"error": "接线检测尚未实现，本次不会输出检测结论"}, 409)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BYTES:
                self.json_response({"error": "请选择不超过 16 MB 的图片"}, 413)
                return
            image = decode_image(self.rfile.read(length))
        except Exception:
            self.json_response({"error": "图片无法读取，请使用有效的 JPEG、PNG 或 WebP 图片"}, 400)
            return
        try:
            report = self.server.engine.inspect(image)
            report = self.persist(image, report)
            self.json_response(report)
        except Exception as error:
            print(f"[error] inspect: {type(error).__name__}: {error}", flush=True)
            self.json_response({"error": "检测失败，请重试并检查服务日志"}, 500)

    def persist(self, image, report):
        return persist_report(image, report)

    def live_action(self, route):
        live = getattr(self.server, "live", None)
        if live is None:
            self.json_response({"error": "未启用设备相机"}, 409)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length < 4096:
                raise ValueError("请求体无效")
            payload = json.loads(self.rfile.read(length))
            if route == "/api/live/reset":
                if payload != {"reset": True}:
                    raise ValueError("重置请求无效")
                self.json_response(live.reset_session())
                return
            if route == "/api/camera/control":
                if not isinstance(payload.get("low_latency"), bool):
                    raise ValueError("曝光状态无效")
                self.json_response(live.camera.set_low_latency(payload["low_latency"]))
                return
            if route == "/api/live/control":
                if not isinstance(payload.get("enabled"), bool):
                    raise ValueError("状态无效")
                live.enabled = payload["enabled"]
                self.json_response({"enabled": live.enabled})
                return
            if route == "/api/live/recheck":
                if set(payload) != {"checkpoint"} or not isinstance(payload["checkpoint"], str):
                    raise ValueError("请选择单个复验模块")
                image, report = live.recheck(payload["checkpoint"])
                report["verdict"] = f"{report['items'][0]['name']}复验：{report['verdict']}"
                saved = self.persist(image, report)
                ledger = getattr(self.server, "ledger", None)
                if ledger:
                    ledger.record_recheck(saved)
                self.json_response(saved)
                return
            snapshot = live.snapshot(int(payload.get("sequence", -1)))
            if snapshot is None:
                raise ValueError("画面已过期，请重新获取实时画面")
            if route == "/api/calibrate":
                result = live.decisions.calibrate(snapshot[2], snapshot[3], payload.get("label", "pass"), payload.get("checkpoint"))
                if payload.get("label", "pass") == "pass" and payload.get("checkpoint") is None:
                    self.server.engine.load_station_reference(live.decisions.directory)
                self.json_response(result)
            else:
                self.json_response(self.persist(snapshot[2], snapshot[3]))
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.json_response({"error": str(error)}, 400)

    def save_review(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length < 4096:
                raise ValueError("请求体过大")
            payload = json.loads(self.rfile.read(length))
            identity = payload["id"]
            if not isinstance(identity, str) or not identity or any(character not in "0123456789Tabcdef-" for character in identity):
                raise ValueError("记录编号无效")
            path = OUTPUT / f"{identity}.json"
            with self.server.engine.lock:
                report = json.loads(path.read_text(encoding="utf-8"))
                checkpoint = payload["checkpoint"]
                if checkpoint not in {item["id"] for item in report["items"]}:
                    raise ValueError("检查点无效")
                verdict = payload.get("verdict")
                reviews = report.setdefault("manual_reviews", {})
                if verdict is None:
                    reviews.pop(checkpoint, None)
                elif verdict in {"正常", "异常"}:
                    reviews[checkpoint] = {"verdict": verdict, "at": datetime.now(timezone.utc).isoformat()}
                else:
                    raise ValueError("复核结论无效")
                report["verdict"] = "人工标记异常" if any(entry["verdict"] == "异常" for entry in reviews.values()) else "人工确认通过" if report["items"] and len(reviews) == len(report["items"]) else "待复核"
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            self.json_response(report)
        except (ValueError, KeyError, TypeError, OSError):
            self.json_response({"error": "无法保存复核，请确认记录和检查项有效"}, 400)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--camera", default="auto", help="USB camera path, or auto for a single connected USB camera")
    modes.add_argument("--image-only", action="store_true", help="Use bundled sample images without opening a camera")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("INSPECTION_DATA_DIR", ROOT / "data")))
    arguments = parser.parse_args()
    if arguments.image_only:
        arguments.camera = None
    data_directory = arguments.data_dir.expanduser().resolve()
    data_directory.mkdir(parents=True, exist_ok=True)
    OUTPUT = data_directory / "output"
    calibration_directory = data_directory / "calibration"
    if not calibration_directory.exists():
        shutil.copytree(ROOT / "calibration", calibration_directory)
    cv2.setNumThreads(2)
    engine = InspectionEngine()
    engine.load_ocr(threads=1 if arguments.camera else 2)
    if engine.ocr is None:
        raise SystemExit(f"OCR initialization failed: {engine.ocr_error}. Run ./install.sh first.")
    server = ThreadingHTTPServer((arguments.host, arguments.port), Handler)
    server.engine = engine
    server.live = None
    server.ledger = ProductionLedger(data_directory / "production.sqlite3")
    server.announcer = ResultAnnouncer(ROOT / "static" / "audio")
    def complete_cycle(image, report):
        saved = server.ledger.record(persist_report(image, report))
        (OUTPUT / f"{saved['id']}.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
        server.announcer.announce(saved["cycle_id"], saved["outcome"])
        return saved
    if arguments.camera:
        engine.ocr_interval = 3.0
        engine.calibrated = True
        engine.start_ocr_worker()
        camera = DeviceCamera(arguments.camera, arguments.camera_width, arguments.camera_height)
        decisions = CalibratedDecisions(calibration_directory)
        engine.load_station_reference(decisions.directory)
        server.live = LiveInspection(camera, engine, decisions, on_complete=complete_cycle)
        server.live.last_result = server.ledger.last_completed()
        if server.live.last_result:
            server.live.cycle_phase = "wait_removal"
        server.live.start()
    print(f"Board inspection: http://{arguments.host}:{server.server_port} | OCR={engine.ocr is not None}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server.live is not None:
            server.live.close()
        engine.close_ocr_worker()
        server.ledger.db.close()
        server.server_close()


if __name__ == "__main__":
    main()