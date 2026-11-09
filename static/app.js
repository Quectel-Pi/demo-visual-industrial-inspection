const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const state = { mode: 'empty', source: '/media/posts.jpg', sourceName: '铜柱实拍样本', report: null, selected: null, filter: 'all', busy: false, stream: null, reviews: {}, objectUrl: null, deviceLive: false, liveTimer: null };
const names = { match: '参考一致', review: '待复核', suspect: '疑似缺失', pending: '待检测', pass: '通过', missing: '缺失', damaged: '损坏', unavailable: '无法检测' };
const colors = { match: '#54c6a6', review: '#e8b34f', suspect: '#f07a7a', pending: '#a8b7b1', pass: '#54c6a6', missing: '#f07a7a', damaged: '#f07a7a', unavailable: '#e8b34f' };
state.previewMode = 'live';
state.telemetry = {};
state.previewStreamId = null;
state.previewRetryTimer = null;
state.liveGeneration = 0;
state.historyResume = false;
state.recheckBusy = false;
state.recheckReport = null;
state.recheckImage = null;
state.checkpointRechecks = {};
let listSignature = '';
let config;
let toastTimer;

function icons() { globalThis.lucide?.createIcons(); $$('svg[data-lucide]').forEach((element) => element.removeAttribute('data-lucide')); }
function toast(message) { $('#toast').textContent = message; $('#toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 5500); }
function node(tag, className, text) { const element = document.createElement(tag); if (className) element.className = className; if (text !== undefined) element.textContent = text; return element; }
function icon(name) { const element = document.createElement('i'); element.dataset.lucide = name; return element; }
function statusBadge(status, text) { return node('span', `badge ${status === 'pending' ? 'neutral' : status}`, text || names[status]); }
function items() { const status = config?.device_camera ? 'unavailable' : 'pending'; const base = state.report?.items || [...(config?.checkpoints || []).map((item) => ({ ...item, status })), ...['左上铜柱', '右上铜柱', '左下铜柱', '右下铜柱'].map((name, index) => ({ id: `B0${index + 1}`, name, status }))]; return base.map((item) => state.deviceLive ? state.checkpointRechecks[item.id]?.report.items[0] || item : item); }
function isPass(item) { return ['match', 'pass'].includes(item.status); }
function itemLabel(item) { return names[item.status]; }
function needsAttention(item) { return !['match', 'pass', 'pending'].includes(item.status); }

function renderList() {
  const all = items();
  const signature = JSON.stringify([state.selected, state.filter, all.map((item) => [item.id, item.name, item.status, Boolean(item.temporal_hold), Boolean(item.session_latched)])]);
  if (signature === listSignature) return;
  listSignature = signature;
  $('#checklist').replaceChildren();
  $('#post-grid').replaceChildren();
  for (const item of all) {
    const isPost = item.id.startsWith('B');
    if (!isPost && state.filter === 'attention' && !needsAttention(item)) continue;
    const button = node('button', `${isPost ? 'post-cell' : 'check-row'}${state.selected === item.id ? ' selected' : ''}`);
    button.type = 'button';
    button.dataset.checkpoint = item.id;
    button.setAttribute('aria-label', `${item.id} ${item.name} ${itemLabel(item)}`);
    if (isPost) {
      const top = node('span', 'cell-top'); top.append(icon('component'), node('span', '', item.name));
      button.append(top, statusBadge(item.status, item.status === 'match' ? '安装样本一致' : itemLabel(item)));
    } else {
      button.append(node('span', 'check-id', item.id), node('span', 'check-name', item.name), statusBadge(item.status, itemLabel(item)), icon('chevron-right'));
    }
    button.title = state.deviceLive && needsAttention(item) ? `单模块复检 ${item.name}` : item.name;
    button.addEventListener('click', () => { selectItem(item.id); if (state.deviceLive && needsAttention(item)) recheckSelected(); });
    $(isPost ? '#post-grid' : '#checklist').append(button);
  }
  if (!$('#checklist').children.length) $('#checklist').append(node('p', 'empty-list', '没有符合筛选条件的检查项'));
  $('#attention-count').textContent = all.filter((item) => item.id.startsWith('A') && needsAttention(item)).length;
  icons();
}

function drawOverlay() {
  const overlay = $('#overlay');
  overlay.replaceChildren();
  overlay.hidden = !$('#show-boxes').checked || Boolean(state.stream) || Boolean(config?.device_camera && state.deviceLive && state.previewMode === 'live');
  if (overlay.hidden) return;
  if (!state.report) return;
  overlay.setAttribute('viewBox', `0 0 ${state.report.width} ${state.report.height}`);
  overlay.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  for (const item of state.report.items) {
    if (!item.polygon.length) continue;
    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    const color = colors[item.status];
    polygon.setAttribute('points', item.polygon.map((point) => point.join(',')).join(' '));
    polygon.setAttribute('stroke', color);
    polygon.setAttribute('fill', state.selected === item.id ? `${color}44` : `${color}0a`);
    polygon.setAttribute('class', state.selected === item.id ? 'selected' : '');
    polygon.setAttribute('tabindex', '0');
    polygon.setAttribute('role', 'button');
    polygon.setAttribute('aria-label', `${item.id} ${item.name}`);
    polygon.addEventListener('click', () => selectItem(item.id));
    polygon.addEventListener('keydown', (event) => { if (['Enter', ' '].includes(event.key)) { event.preventDefault(); selectItem(item.id); } });
    overlay.append(polygon);
    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    const left = Math.max(3, Math.min(...item.polygon.map((point) => point[0])));
    const top = Math.max(18, Math.min(...item.polygon.map((point) => point[1])) - 5);
    label.setAttribute('x', left); label.setAttribute('y', top); label.setAttribute('fill', color); label.textContent = item.id;
    overlay.append(label);
  }
}

function selectItem(identity) {
  state.selected = identity;
  const cachedRecheck = state.deviceLive ? state.checkpointRechecks[identity] : null;
  const recheck = cachedRecheck?.report || null;
  const item = (recheck?.items || items()).find((candidate) => candidate.id === identity);
  if (!item) return;
  $('#detail-name').textContent = item.name;
  $('#detail-id').textContent = recheck ? `${identity} / 复检 · 帧 ${recheck.sequence}` : `${identity} / ${item.name}`;
  $('#detail-heading').textContent = itemLabel(item);
  $('#detail-reason').textContent = item.reason || '尚未执行图像检测';
  $('#detail-badge').className = `badge ${item.status === 'pending' ? 'neutral' : item.status}`;
  $('#detail-badge').textContent = itemLabel(item);
  $('#detail-score').textContent = item.scores?.pass != null ? `正常参考匹配 ${Math.round(item.scores.pass * 100)}% · 门槛 ${Math.round((item.pass_threshold ?? 0.65) * 100)}%` : item.score == null ? '' : `参考相似度 ${Math.round(item.score * 100)} / 100（不是模型置信度）`;
  $('#recheck-button').disabled = state.recheckBusy || !state.deviceLive;
  $('#recheck-button').title = `仅复检 ${item.name}`;
  $('#recheck-button span').textContent = state.recheckBusy ? '复检中…' : '单模块复检';
  $('#export-button span').textContent = recheck || state.report?.inspection_scope === 'checkpoint' ? '导出复检表' : '导出整板表';
  $$('.review-actions button').forEach((button) => { button.disabled = !state.report || Boolean(state.report.live); });
  $('#manual-note').textContent = state.reviews[identity] ? `人工复核：${state.reviews[identity].verdict} · ${new Date(state.reviews[identity].at).toLocaleTimeString()}` : '人工复核：尚未记录';
  const canvas = $('#detail-canvas');
  const context = canvas.getContext('2d');
  context.fillStyle = '#e8eeeb'; context.fillRect(0, 0, canvas.width, canvas.height);
  const detailReport = recheck || state.report;
  const detailImage = recheck ? cachedRecheck.image : $('#board-image');
  if (detailReport && item.polygon?.length && detailImage?.complete && detailImage.naturalWidth) {
    const left = Math.max(0, Math.min(...item.polygon.map((point) => point[0])) - 10);
    const top = Math.max(0, Math.min(...item.polygon.map((point) => point[1])) - 10);
    const right = Math.min(detailReport.width, Math.max(...item.polygon.map((point) => point[0])) + 10);
    const bottom = Math.min(detailReport.height, Math.max(...item.polygon.map((point) => point[1])) + 10);
    const width = right - left; const height = bottom - top;
    if (width > 0 && height > 0) {
      const scale = Math.min(canvas.width / width, canvas.height / height);
      context.drawImage(detailImage, left, top, width, height, (canvas.width - width * scale) / 2, (canvas.height - height * scale) / 2, width * scale, height * scale);
    }
  }
  renderList(); drawOverlay();
}

function renderReport() {
  const report = state.report;
  const all = items();
  const cachedIdentity = !report?.version.device_name && report?.model_gate?.identity;
  const identity = cachedIdentity || report?.version;
  $('#verdict').textContent = report ? (report.blocking_reason || report.verdict) : '等待检测';
  $('#match-count').textContent = report ? all.filter(isPass).length : '—';
  $('#review-count').textContent = report ? all.filter((item) => config?.device_camera ? ['missing', 'damaged'].includes(item.status) : !isPass(item)).length : '—';
  $('#post-count').textContent = `${report ? all.filter((item) => item.id.startsWith('B') && isPass(item)).length : '—'} / 4`;
  $('#elapsed').textContent = report ? (report.duration_ms / 1000).toFixed(2) : '—';
  $('#board-version').textContent = identity?.model || '等待识别型号';
  $('#device-name').textContent = identity?.device_name || '设备未识别';
  $('#device-revision').textContent = identity?.model || (report?.model_gate?.status === 'confirmed' ? 'M2' : '型号未识别');
  document.title = `${identity?.device_name || '板卡'} · 装配检测`;
  $('#version-reason').textContent = report?.model_gate?.status === 'confirmed' ? (report.located ? '正在检测器件' : '型号已确认，正在定位器件') : report?.version.reason || (config?.ocr_ready ? '等待识别型号' : 'OCR 尚未就绪');
  $('#ocr-status').className = `badge ${cachedIdentity ? 'pass' : report?.version.status || 'neutral'}`;
  $('#ocr-status').textContent = report?.model_gate?.status === 'confirmed' || report?.version.status === 'match' ? '已识别' : report ? '识别型号中' : 'OCR';
  $('#position-state').textContent = report ? (report.located ? `已定位 · ${report.alignment.inliers} 个特征内点` : '未定位到开发板') : '等待定位';
  $('#source-label').textContent = config?.device_camera && state.deviceLive ? (state.previewMode === 'live' ? '实时预览' : '检测帧') : `${state.sourceName} · ${report ? '已检测' : '尚未检测'}`;
  if (config?.device_camera) {
    $('#image-size').textContent = `预览 ${state.telemetry.preview_fps || 0} FPS · ${state.telemetry.adaptive_exposure ? '自适应' : '锁定'} ${state.telemetry.exposure_ms || 0} ms`;
    $('#service-state').textContent = state.deviceLive ? `检测 ${report?.fps || 0} 次/秒` : '检测已暂停';
  }
  $('#report-id').textContent = report ? report.id : '尚无检测记录';
  $('#export-button').disabled = !report;
  if (report) selectItem(state.selected || report.items.find((item) => item.status === 'suspect')?.id || report.items[0].id);
  else { renderList(); drawOverlay(); resetDetail(); }
}

function resetDetail() {
  $('#recheck-button').disabled = true;
  $('#detail-name').textContent = '局部复核'; $('#detail-heading').textContent = '等待检测结果';
  $('#detail-reason').textContent = '—'; $('#detail-score').textContent = ''; $('#detail-id').textContent = 'CHECKPOINT';
  $('#detail-badge').className = 'badge neutral'; $('#detail-badge').textContent = '未选择';
  $('#detail-canvas').getContext('2d').clearRect(0, 0, 440, 220);
  $$('.review-actions button').forEach((button) => { button.disabled = true; });
  $('#manual-note').textContent = '人工复核：尚未记录';
}

function stopCamera() {
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null; $('#camera-video').srcObject = null; $('#camera-video').hidden = true;
  $('#image-stage').hidden = false; $('#capture-button').disabled = true;
  $('#camera-button span').textContent = '开启相机';
}

async function setSource(url, name, sample) {
  stopCamera();
  if (state.objectUrl && state.objectUrl !== url) { URL.revokeObjectURL(state.objectUrl); state.objectUrl = null; }
  state.source = url; state.sourceName = name; state.report = null; state.selected = null; state.reviews = {};
  $('#board-image').src = url;
  $$('.sample-button').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.sample === sample)));
  renderReport();
}

function setBusy(busy) {
  state.busy = busy; $('#scan-progress').hidden = !busy;
  $('#inspect-button span').textContent = busy ? '检测中…' : '开始检测';
  $$('#upload-button, #camera-button, #history-button, .sample-button, [data-mode]').forEach((button) => { button.disabled = busy; });
  $('#capture-button').disabled = busy || !state.stream;
  $('#inspect-button').disabled = busy || state.mode !== 'empty';
}

async function inspect() {
  if (state.busy || state.mode !== 'empty') return;
  if (config?.device_camera) { await toggleDeviceLive(!state.deviceLive); return; }
  try {
    if (state.stream) await captureFrame();
    setBusy(true);
    const imageResponse = await fetch(state.source);
    if (!imageResponse.ok) throw new Error('图像读取失败');
    const response = await fetch('/api/inspect', { method: 'POST', headers: { 'Content-Type': 'application/octet-stream', 'X-Inspection-Mode': state.mode }, body: await imageResponse.blob() });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '检测失败');
    state.report = data; state.source = data.image_url; state.selected = null; state.reviews = {};
    await new Promise((resolve, reject) => { const image = $('#board-image'); image.onload = resolve; image.onerror = reject; image.src = data.image_url; });
    renderReport();
    toast(data.located ? '检测完成，待复核项请查看局部图像' : '未能定位开发板，请重新拍摄');
  } catch (error) { toast(error.message || '检测服务连接失败'); }
  finally { setBusy(false); }
}

async function captureFrame() {
  const video = $('#camera-video');
  if (!state.stream || !video.videoWidth) throw new Error('相机画面尚未就绪');
  const canvas = document.createElement('canvas'); canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.94));
  if (!blob) throw new Error('拍摄失败');
  const url = URL.createObjectURL(blob);
  await setSource(url, '相机采集'); state.objectUrl = url;
}

$('#inspect-button').addEventListener('click', inspect);
$('#upload-button').addEventListener('click', () => $('#image-input').click());
$('#image-input').addEventListener('change', async (event) => {
  const file = event.target.files[0]; if (!file) return;
  if (file.size > 16 * 1024 * 1024) { toast('请选择不超过 16 MB 的图片'); event.target.value = ''; return; }
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { toast('请选择 JPEG、PNG 或 WebP 图片'); event.target.value = ''; return; }
  const url = URL.createObjectURL(file); await setSource(url, file.name); state.objectUrl = url; event.target.value = '';
});
$('#camera-button').addEventListener('click', async () => {
  if (config?.device_camera) { await toggleDeviceLive(!state.deviceLive); return; }
  if (state.stream) { stopCamera(); return; }
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('浏览器相机不可用，请在 localhost 或 HTTPS 页面开启，或上传拍摄图片');
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
    $('#camera-video').srcObject = state.stream; $('#camera-video').hidden = false; $('#image-stage').hidden = true;
    $('#source-label').textContent = '相机实时预览'; $('#capture-button').disabled = false; $('#camera-button span').textContent = '关闭相机';
    await $('#camera-video').play();
  } catch (error) { stopCamera(); toast(error.name === 'NotAllowedError' ? '相机权限被拒绝，请允许相机访问或上传图片' : error.message); }
});
$('#capture-button').addEventListener('click', () => { if (config?.device_camera) saveLive(false); else captureFrame().catch((error) => toast(error.message)); });
$$('.sample-button').forEach((button) => button.addEventListener('click', () => { const sample = config.samples.find((candidate) => candidate.id === button.dataset.sample); setSource(sample.url, sample.name, sample.id); }));
$$('[data-mode]').forEach((button) => button.addEventListener('click', () => {
  state.mode = button.dataset.mode;
  $$('[data-mode]').forEach((candidate) => candidate.setAttribute('aria-selected', String(candidate === button)));
  $('#mode-notice').hidden = state.mode === 'empty'; $('#inspect-button').disabled = state.mode !== 'empty';
  if (config?.device_camera) toggleDeviceLive(state.mode === 'empty');
}));
$$('[data-filter]').forEach((button) => button.addEventListener('click', () => { state.filter = button.dataset.filter; $$('[data-filter]').forEach((candidate) => candidate.classList.toggle('active', candidate === button)); renderList(); }));
$('#show-boxes').addEventListener('change', drawOverlay);
$('#fast-exposure').addEventListener('change', async (event) => {
  const control = event.target;
  control.disabled = true;
  try { const result = await postJson('/api/camera/control', { low_latency: control.checked }); control.checked = result.low_latency; }
  catch (error) { control.checked = Boolean(state.telemetry.low_latency); toast(error.message); }
  finally { control.disabled = false; }
});
$('#fullscreen-button').addEventListener('click', async () => { try { if (document.fullscreenElement) await document.exitFullscreen(); else await $('#viewer').requestFullscreen(); } catch { toast('此浏览器不支持画面全屏'); } });
$('#board-image').addEventListener('load', () => { if (!config?.device_camera) { $('#image-size').textContent = `${$('#board-image').naturalWidth} × ${$('#board-image').naturalHeight} / FRONT`; if (state.selected) selectItem(state.selected); } });
$('#board-image').addEventListener('error', () => toast('图片无法显示，请重新选择有效图片'));
async function saveReview(verdict) {
  if (!state.selected || !state.report || state.busy) return;
  const checkpoint = state.selected;
  $$('.review-actions button').forEach((button) => { button.disabled = true; });
  try {
    const response = await fetch('/api/review', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ id: state.report.id, checkpoint, verdict }) });
    const report = await response.json();
    if (!response.ok) throw new Error(report.error);
    state.report = report; state.reviews = report.manual_reviews || {}; renderReport();
    toast(verdict ? '人工复核已保存' : '人工复核已清除');
  } catch (error) { toast(error.message || '保存失败'); selectItem(checkpoint); }
}
for (const [selector, verdict] of [['#manual-pass', '正常'], ['#manual-fail', '异常']]) $(selector).addEventListener('click', () => saveReview(verdict));
$('#manual-reset').addEventListener('click', () => saveReview(null));
function downloadReport(report) {
  const anchor = node('a'); anchor.href = `${report.table_url || report.report_url.replace(/\.json$/, '.html')}?download=1`; anchor.download = `${report.id}.html`; document.body.append(anchor); anchor.click(); anchor.remove();
}
$('#export-button').addEventListener('click', () => {
  const report = state.deviceLive ? state.checkpointRechecks[state.selected]?.report || state.report : state.report;
  if (!report) return;
  if (report.live) { saveLive(true); return; }
  downloadReport(report);
});
$('#history-button').addEventListener('click', async () => {
  state.historyResume = Boolean(config?.device_camera && state.deviceLive);
  if (config?.device_camera && state.deviceLive) await toggleDeviceLive(false);
  $('#history-list').replaceChildren(node('p', 'empty-list', '正在读取…')); $('#history-dialog').showModal();
  try {
    const response = await fetch('/api/history'); if (!response.ok) throw new Error('历史记录读取失败');
    const history = await response.json(); $('#history-list').replaceChildren();
    if (!history.length) $('#history-list').append(node('p', 'empty-list', '暂无检测记录'));
    for (const entry of history) {
      const button = node('button', 'history-entry'); const image = node('img'); image.src = entry.image_url; image.alt = '检测原图';
      const text = node('div'); text.append(node('strong', '', entry.version.text), node('small', '', `${new Date(entry.created_at).toLocaleString()} · ${entry.verdict}`)); button.append(image, text, icon('chevron-right'));
      button.addEventListener('click', async () => {
        try { const response = await fetch(`/output/${entry.id}.json`); if (!response.ok) throw new Error('记录无法读取'); const report = await response.json(); await setSource(report.image_url, '历史记录'); state.report = report; state.reviews = report.manual_reviews || {}; await $('#board-image').decode(); renderReport(); $('#history-dialog').close(); }
        catch (error) { toast(error.message); }
      });
      $('#history-list').append(button);
    }
    icons();
  } catch (error) { $('#history-list').replaceChildren(node('p', 'empty-list', error.message)); }
});
$('#close-history').addEventListener('click', () => $('#history-dialog').close());
$('#history-dialog').addEventListener('close', () => {
  const resume = state.historyResume && (!state.report || state.report.live);
  state.historyResume = false;
  if (resume) toggleDeviceLive(true);
});
globalThis.addEventListener('beforeunload', stopCamera);

async function postJson(route, payload, timeout = 5000) {
  const response = await fetch(route, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: AbortSignal.timeout(timeout) });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || '设备操作失败');
  return result;
}

async function recheckSelected() {
  const checkpoint = state.selected;
  if (!checkpoint || state.recheckBusy || !state.deviceLive) return;
  const generation = state.liveGeneration;
  const requestedCycle = state.telemetry.cycle?.cycle_id;
  state.recheckBusy = true; state.recheckReport = null; state.recheckImage = null;
  delete state.checkpointRechecks[checkpoint];
  selectItem(checkpoint);
  try {
    const report = await postJson('/api/live/recheck', { checkpoint }, 15000);
    if (generation !== state.liveGeneration || report.stream_id !== state.telemetry.stream_id || requestedCycle !== state.telemetry.cycle?.cycle_id) return;
    const image = new Image(); image.src = report.image_url; await decodeWithTimeout(image);
    if (generation !== state.liveGeneration || report.stream_id !== state.telemetry.stream_id || requestedCycle !== state.telemetry.cycle?.cycle_id) return;
    state.recheckReport = report; state.recheckImage = image;
    state.checkpointRechecks[checkpoint] = { report, image };
    await refreshProduction();
    toast(`${report.items[0].name}复验：${names[report.items[0].status]}`);
  } catch (error) { toast(error.message || '复验失败'); }
  finally { state.recheckBusy = false; renderReport(); renderCycle(state.telemetry); }
}
$('#recheck-button').addEventListener('click', recheckSelected);

async function refreshProduction() {
  const response = await fetch('/api/production', { cache: 'no-store', signal: AbortSignal.timeout(3000) });
  if (!response.ok) throw new Error('批次统计读取失败');
  const summary = await response.json();
  $('#production-total').textContent = summary.total;
  $('#production-pass').textContent = summary.passed;
  $('#production-fail').textContent = summary.failed;
  $('#production-first-rate').textContent = summary.first_pass_rate == null ? '—' : `${(summary.first_pass_rate * 100).toFixed(1)}%`;
  $('#production-rate').textContent = summary.pass_rate == null ? '—' : `${(summary.pass_rate * 100).toFixed(1)}%`;
}

$('#production-export').addEventListener('click', () => {
  const anchor = node('a'); anchor.href = '/api/production/export.html'; anchor.download = 'M2-production.html'; document.body.append(anchor); anchor.click(); anchor.remove();
});
$('#audio-enabled').addEventListener('change', async () => {
  const input = $('#audio-enabled'); input.disabled = true;
  try { const audio = await postJson('/api/audio/control', { enabled: input.checked }); input.checked = audio.enabled; }
  catch (error) { input.checked = !input.checked; toast(error.message); }
  finally { input.disabled = false; }
});

function renderCycle(data) {
  const cycle = data.cycle;
  if (!cycle) return;
  $('#elapsed-label').textContent = '本轮剩余时间';
  $('#elapsed').textContent = cycle.remaining_seconds.toFixed(1);
  if (cycle.phase === 'running') $('#verdict').textContent = '检测中';
  else if (cycle.phase === 'wait_removal') $('#verdict').textContent = cycle.last_result ? `${Object.keys(state.checkpointRechecks).length && items().every(isPass) && !cycle.last_result.cycle_interrupted ? '复检后合格' : cycle.last_result.verdict} · 待移板` : '等待移板';
  else $('#verdict').textContent = '等待 M2';
  if (cycle.last_result) $('#report-id').textContent = `第 ${cycle.last_result.board_number} 块 · ${cycle.last_result.verdict}`;
  if (data.audio) {
    $('#audio-enabled').checked = data.audio.enabled;
    $('#audio-status').textContent = data.audio.enabled ? data.audio.error || '' : '已静音';
  }
}

$('#reset-session').addEventListener('click', async () => {
  const button = $('#reset-session');
  button.disabled = true;
  state.liveGeneration += 1;
  clearTimeout(state.liveTimer);
  try {
    await postJson('/api/live/reset', { reset: true });
    state.report = null; state.reviews = {}; state.selected = null; state.recheckReport = null; state.recheckImage = null; state.checkpointRechecks = {};
    renderReport();
    await toggleDeviceLive(true);
    toast('新一轮检测已开始，等待确认 M2 型号');
  } catch (error) {
    toast(error.message);
    if (state.deviceLive) pollDevice(state.liveGeneration);
  } finally { button.disabled = false; }
});

async function toggleDeviceLive(enabled) {
  try {
    await postJson('/api/live/control', { enabled });
    state.deviceLive = enabled;
    state.liveGeneration += 1;
    updateDeviceView();
    clearTimeout(state.liveTimer);
    $('#inspect-button span').textContent = enabled ? '暂停检测' : '继续实时检测';
    $('#camera-button span').textContent = enabled ? '设备相机在线' : '恢复设备相机';
    $('#capture-button').disabled = !state.report;
    if (enabled) pollDevice(state.liveGeneration);
    else { $('#position-state').textContent = '检测已暂停'; $('#source-label').textContent = '设备画面 · 已暂停'; }
  } catch (error) { toast(error.message); }
}

function updateDeviceView() {
  if (!config?.device_camera) return;
  const streaming = state.deviceLive && state.previewMode === 'live';
  $('#live-preview').hidden = !streaming;
  $('#board-image').hidden = streaming;
  if (streaming && !$('#live-preview').getAttribute('src')) reconnectPreview();
  if (!streaming) { clearTimeout(state.previewRetryTimer); $('#live-preview').removeAttribute('src'); }
  $$('[data-view]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.view === state.previewMode)));
  drawOverlay();
}

function reconnectPreview() {
  if (!config?.device_camera || !state.deviceLive || state.previewMode !== 'live') return;
  clearTimeout(state.previewRetryTimer);
  state.previewStreamId = state.telemetry.stream_id || null;
  $('#live-preview').src = `/api/preview.mjpg?connection=${Date.now()}`;
}

$('#live-preview').addEventListener('error', () => {
  if (!state.deviceLive || state.previewMode !== 'live') return;
  $('#position-state').textContent = '预览连接中断，正在自动重连';
  $('#preview-error').hidden = false;
  $('#preview-error').textContent = '预览连接中断，正在自动重连';
  clearTimeout(state.previewRetryTimer);
  state.previewRetryTimer = setTimeout(reconnectPreview, 700);
});
$('#live-preview').addEventListener('load', () => { if (!state.telemetry.error) $('#preview-error').hidden = true; });

async function decodeWithTimeout(image) {
  let timer;
  try { await Promise.race([image.decode(), new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('检测帧读取超时，正在重新获取')), 2500); })]); }
  finally { clearTimeout(timer); }
}

$$('[data-view]').forEach((button) => button.addEventListener('click', () => {
  state.previewMode = button.dataset.view;
  updateDeviceView();
  renderReport();
}));

async function pollDevice(generation = state.liveGeneration) {
  if (!state.deviceLive || generation !== state.liveGeneration) return;
  try {
    const response = await fetch('/api/live', { signal: AbortSignal.timeout(2500), cache: 'no-store' });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '实时服务不可用');
    if (!state.deviceLive || generation !== state.liveGeneration) return;
    state.telemetry = data;
    if (state.activeCycleId !== data.cycle?.cycle_id) {
      state.activeCycleId = data.cycle?.cycle_id;
      state.recheckReport = null; state.recheckImage = null; state.checkpointRechecks = {};
    }
    const result = data.cycle?.last_result;
    if (result && result.id !== state.lastCompletedId) {
      state.lastCompletedId = result.id;
      state.recheckReport = null; state.recheckImage = null; state.checkpointRechecks = {};
      await refreshProduction();
      toast(`第 ${result.board_number} 块：${result.verdict}`);
    }
    if (data.cycle?.phase === 'wait_removal' && result) data.report = result;
    $('#preview-error').hidden = !data.error;
    $('#preview-error').textContent = data.error || '';
    if (data.stream_id && data.stream_id !== state.previewStreamId) {
      state.report = null;
      state.recheckReport = null; state.recheckImage = null; state.checkpointRechecks = {};
      state.previewStreamId = data.stream_id;
      reconnectPreview();
      await calibrationState();
      if (!state.deviceLive || generation !== state.liveGeneration) return;
    }
    if (!$('#fast-exposure').disabled) $('#fast-exposure').checked = Boolean(data.low_latency);
    if (data.report?.board_presence === 'absent' || data.report?.model_gate?.status === 'rejected') { state.recheckReport = null; state.recheckImage = null; state.checkpointRechecks = {}; }
    if (data.report && (data.report.id !== state.report?.id || data.report.sequence !== state.report?.sequence)) {
      const image = new Image(); image.src = data.report.image_url; await decodeWithTimeout(image);
      if (!state.deviceLive || generation !== state.liveGeneration) return;
      state.report = data.report; state.source = data.report.image_url; state.sourceName = '设备实时相机';
      $('#board-image').src = data.report.image_url;
      await decodeWithTimeout($('#board-image'));
      renderReport(); $('#capture-button').disabled = false;
    } else if (!data.report) {
      state.report = null; renderReport(); $('#verdict').textContent = '无法检测'; $('#position-state').textContent = data.error || '等待相机';
      $('#capture-button').disabled = true;
    }
    renderCycle(data);
  } catch (error) {
    if (!state.deviceLive || generation !== state.liveGeneration) return;
    state.report = null; renderReport();
    $('#preview-error').hidden = false; $('#preview-error').textContent = '连接中断，正在恢复实时画面';
    $('#position-state').textContent = error.message || '设备相机连接中断'; $('#verdict').textContent = '连接恢复中'; reconnectPreview();
  }
  finally { if (state.deviceLive && generation === state.liveGeneration) state.liveTimer = setTimeout(() => pollDevice(generation), 600); }
}

async function saveLive(download) {
  if (!state.report?.live) return;
  try {
    const report = await postJson('/api/live/save', { sequence: state.report.sequence });
    if (download) {
      downloadReport(report);
    }
    toast('当前设备画面与检测结果已保存');
  } catch (error) { toast(error.message); }
}

async function calibrationState() {
  const response = await fetch('/api/calibration', { signal: AbortSignal.timeout(2500), cache: 'no-store' });
  if (!response.ok) throw new Error('完整板标准读取失败');
  const counts = await response.json();
  const complete = Object.values(counts).filter((entry) => entry.pass > 0).length;
  $('#calibration-state').textContent = `完整标准 ${complete} / ${config.checkpoints.length + 4} 项`;
}

$('#calibrate-button').addEventListener('click', async () => {
  if (!state.report?.located) { toast('请先将完整开发板正面放入相机画面'); return; }
  const choice = $('#calibration-label').value;
  const checkpoint = choice === 'all' ? null : state.selected;
  if (choice !== 'all' && !checkpoint) { toast('请先选择需要标定的接口或铜柱'); return; }
  const label = choice === 'all' ? 'pass' : choice;
  const message = choice === 'all' ? `确认当前实物的${config.checkpoints.length}个接口和4个铜柱均完整正常？本次将采集全部位置为通过标准。` : `确认选中位置 ${checkpoint} 当前确实为“${names[label]}”？仅采集这一位置。`;
  const resume = state.deviceLive;
  if (resume) { await toggleDeviceLive(false); if (state.deviceLive) return; }
  const sequence = state.report.sequence;
  $('#calibrate-button').disabled = true;
  try {
    if (!confirm(message)) return;
    await postJson('/api/calibrate', { sequence, label, checkpoint }); await calibrationState(); toast('标定已保存，连续三帧确认后更新结果');
  }
  catch (error) { toast(error.message); }
  finally { $('#calibrate-button').disabled = false; if (resume) await toggleDeviceLive(true); }
});

async function boot() {
  icons();
  try {
    const response = await fetch('/api/config'); if (!response.ok) throw new Error(); config = await response.json();
    $('#service-state').textContent = '本地检测服务在线';
    if (config.device_camera) {
      document.body.classList.add('device-mode');
      $('.topbar').insertBefore($('.mode-switch'), $('.top-status'));
      $('#device-view-modes').hidden = false;
      $('#exposure-control').hidden = false;
      $('#camera-button').hidden = true;
      $('#service-state').textContent = '设备端检测服务在线';
      $('#match-label').textContent = '通过'; $('#failure-label').textContent = '缺失 / 损坏'; $('#attention-label').textContent = '未通过';
      $('#scope-notice').textContent = '外观检查 · 不代表电气功能正常';
      $('#reset-session').hidden = false;
      $('#recheck-button').hidden = false;
      $('#production-strip').hidden = false;
      $('.sample-strip').hidden = true; $('#upload-button').hidden = true; $('#device-calibration').hidden = false;
      $('.review-actions').hidden = true; $('#manual-note').hidden = true;
      $('#capture-button').title = '保存当前设备帧'; $('#capture-button').setAttribute('aria-label', '保存当前设备帧');
      $('#source-label').textContent = `设备实时相机 · ${config.device_camera}`;
      await refreshProduction(); await calibrationState(); await toggleDeviceLive(true);
    } else { await setSource(state.source, state.sourceName, 'posts'); }
    renderReport();
  } catch { $('#service-state').textContent = '检测服务离线'; $('#inspect-button').disabled = true; toast('无法连接本地检测服务，请检查服务状态'); }
}
boot();