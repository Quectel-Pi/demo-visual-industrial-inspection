import argparse
import base64
import json
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np


def request_json(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=40) as response:
        return json.load(response)


def benchmark(base):
    stamps = []
    last_frame = None
    with urlopen(base + "/api/preview.mjpg", timeout=10) as response:
        for _ in range(45):
            length = None
            while True:
                line = response.readline()
                if not line:
                    raise RuntimeError("预览连接中断")
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":", 1)[1])
                if line == b"\r\n" and length is not None:
                    break
            last_frame = response.read(length)
            stamps.append(time.monotonic())
    image = cv2.imdecode(np.frombuffer(last_frame, np.uint8), cv2.IMREAD_COLOR)
    if image is None or float(image.std()) < 2:
        raise RuntimeError("相机帧为空或无法解码")
    gaps = np.diff(stamps)
    return {"received_fps": round(44 / (stamps[-1] - stamps[0]), 2), "p95_gap_ms": round(float(np.percentile(gaps, 95)) * 1000, 1), "frame_size": [image.shape[1], image.shape[0]], "status": request_json(base + "/api/live")}


def verify_browser(base, driver, reuse=False):
    session = Path("/tmp/board-inspection-webdriver-session").read_text() if reuse else request_json(driver + "/session", {"capabilities": {"alwaysMatch": {"browserName": "firefox"}}})["value"]["sessionId"]
    Path("/tmp/board-inspection-webdriver-session").write_text(session)
    endpoint = driver + "/session/" + session
    request_json(endpoint + "/url", {"url": base})
    ready = """
const done = arguments[arguments.length - 1];
const started = performance.now();
function check() {
    if (document.querySelector('#live-preview')?.naturalWidth > 0 && !document.querySelector('#capture-button').disabled) return done(true);
  if (performance.now() - started > 15000) return done(false);
  setTimeout(check, 100);
}
check();
"""
    preview_ready = request_json(endpoint + "/execute/async", {"script": ready, "args": []})["value"]
    script = """
const bounds = {};
for (const selector of ['.topbar','.summary-strip','#viewer','.source-toolbar','#device-calibration','.detail-panel','.results-panel','.result-footer']) {
  const box = document.querySelector(selector).getBoundingClientRect();
  bounds[selector] = {x:box.x,y:box.y,width:box.width,height:box.height,inside:box.x>=0 && box.y>=0 && box.right<=innerWidth+1 && box.bottom<=innerHeight+1};
}
return {viewport:[innerWidth,innerHeight],overflow:document.documentElement.scrollWidth>innerWidth || document.documentElement.scrollHeight>innerHeight,stream:document.querySelector('#live-preview').naturalWidth,identity:document.querySelector('#device-name').textContent,revision:document.querySelector('#device-revision').textContent,bounds};
"""
    layout = request_json(endpoint + "/execute/sync", {"script": script, "args": []})["value"]
    screenshot = request_json(endpoint + "/screenshot")["value"]
    Path("/tmp/board-inspection-display.png").write_bytes(base64.b64decode(screenshot))
    interaction = """
document.querySelector('[data-view="inspection"]').click();
const frozen = document.querySelector('#live-preview').hidden && !document.querySelector('#board-image').hidden;
document.querySelector('[data-view="live"]').click();
return {inspection:frozen,live:!document.querySelector('#live-preview').hidden};
"""
    layout["interactions"] = request_json(endpoint + "/execute/sync", {"script": interaction, "args": []})["value"]
    layout["preview_ready"] = preview_ready
    identity_check = """
const done=arguments[arguments.length-1];
fetch('/api/live').then(response=>response.json()).then(data=>{
    const report=data.report;
    const identity=report?.version.device_name ? report.version : report?.model_gate?.identity || report?.version;
    done({
        name:document.querySelector('#device-name').textContent===(identity?.device_name || '设备未识别'),
        model:document.querySelector('#device-revision').textContent===(identity?.model || (report?.model_gate?.status==='confirmed' ? 'M2' : '型号未识别'))
    });
}).catch(error=>done({error:error.message}));
"""
    layout["identity_from_ocr"] = request_json(endpoint + "/execute/async", {"script": identity_check, "args": []})["value"]
    if not preview_ready or layout["overflow"] or not all(box["inside"] for box in layout["bounds"].values()) or layout["bounds"]["#viewer"]["height"] < 300 or not layout["identity_from_ocr"].get("name") or not layout["identity_from_ocr"].get("model"):
        raise RuntimeError("设备界面验收失败：" + json.dumps(layout, ensure_ascii=False))
    return layout


def verify_recovery(driver):
        session = Path("/tmp/board-inspection-webdriver-session").read_text()
        endpoint = driver + "/session/" + session
        before = request_json(endpoint + "/execute/sync", {"script": "return document.querySelector('#live-preview').getAttribute('src');", "args": []})["value"]
        stopped = False
        recovered = False
        try:
                subprocess.run(["systemctl", "stop", "board-inspection-demo.service"], check=True)
                stopped = request_json(endpoint + "/execute/async", {"script": """
const done=arguments[arguments.length-1], started=performance.now();
function check() {
    if (!document.querySelector('#preview-error').hidden) return done(true);
    if (performance.now()-started>10000) return done(false);
    setTimeout(check,100);
}
check();
""", "args": []})["value"]
        finally:
                subprocess.run(["systemctl", "start", "board-inspection-demo.service"], check=True)
        recovered = request_json(endpoint + "/execute/async", {"script": """
const before=arguments[0], done=arguments[arguments.length-1], started=performance.now();
function check() {
    const image=document.querySelector('#live-preview');
    if (image.getAttribute('src')!==before && image.naturalWidth>0 && !document.querySelector('#capture-button').disabled && document.querySelector('#preview-error').hidden) return done(true);
    if (performance.now()-started>20000) return done(false);
    setTimeout(check,100);
}
check();
""", "args": [before]})["value"]
        history = request_json(endpoint + "/execute/async", {"script": """
const done=arguments[arguments.length-1], started=performance.now();
document.querySelector('#history-button').click();
function opened() {
    if (document.querySelector('#history-dialog').open) {
        document.querySelector('#close-history').click();
        return resumed();
    }
    if (performance.now()-started>10000) return done(false);
    setTimeout(opened,100);
}
function resumed() {
    fetch('/api/live').then(response=>response.json()).then(data=>{
        if(data.enabled && !document.querySelector('#live-preview').hidden && document.querySelector('#live-preview').naturalWidth>0) return done(true);
        if(performance.now()-started>15000) return done(false);
        setTimeout(resumed,100);
    }).catch(()=>done(false));
}
opened();
""", "args": []})["value"]
        result = {"disconnect_visible": stopped, "restart_recovered": recovered, "history_close_resumes": history}
        if not all(result.values()):
                raise RuntimeError("实时恢复验收失败：" + json.dumps(result))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    parser.add_argument("--webdriver")
    parser.add_argument("--reuse-session", action="store_true")
    parser.add_argument("--restart-check", action="store_true")
    arguments = parser.parse_args()
    base = arguments.base
    result = benchmark(base)
    report = result["status"].get("report") or {}
    result["status"] = {key: value for key, value in result["status"].items() if key != "report"}
    result["detection"] = {key: report.get(key) for key in ["fps", "duration_ms", "located", "version"]}
    if arguments.webdriver:
        result["browser"] = verify_browser(base, arguments.webdriver, arguments.reuse_session)
        if arguments.restart_check:
            result["recovery"] = verify_recovery(arguments.webdriver)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()