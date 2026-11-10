'use strict';

const $ = id => document.getElementById(id);
const cameras = ['A', 'B'];
const state = {
  videos: [], limits: {max_upload_bytes: 1024 ** 3}, busy: false,
  sources: {A: {mode: 'library'}, B: {mode: 'library'}},
  jobId: null, pollTimer: null, pollGeneration: 0,
};
const statusNames = {
  queued: '분석 대기', pending: '대기', running: '분석 중', analyzing: '영상 분석 중',
  resolving: '원본 확인 중', verifying: '검수 근거 확인 중', loading: '기존 근거 확인 중',
  completed: '완료', completed_with_errors: '일부 처리 실패', failed: '처리 실패', interrupted: '실행 중단',
  reused: '기존 근거 연결 완료', stored_ai_review: '기존 AI 검수 연결', fresh_analysis: '새 영상 분석',
};
const terminal = new Set(['completed', 'completed_with_errors', 'failed', 'interrupted']);
const modeNames = {stored_ai_review: '기존 AI 검수', fresh_analysis: '새 분석', stored_automatic: '기존 자동 분석', failed: '처리 실패'};
const correspondenceNames = {
  ai_visual_review: '저장된 AI 검수의 동작 대응', reviewed_actions: '검수된 동작 대응',
  stored_ai_review: '저장된 AI 검수의 동작 대응', operator_selection: '사용자가 확인한 사건 대응',
  user_selection: '사용자가 확인한 사건 대응', operator_request: '사용자가 확인한 사건 대응',
  verified_ai_action_correspondence: '저장된 AI 검수의 동작 대응',
  operator_pair_selection: '사용자가 확인한 사건 대응',
};
const fmt = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function safeUrl(value) {
  if (typeof value !== 'string' || !value.startsWith('/') || value.startsWith('//')) return null;
  const url = new URL(value, location.origin);
  return url.origin === location.origin ? url.pathname + url.search + url.hash : null;
}
async function api(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(url, {...options, signal: controller.signal});
    let data;
    try { data = await response.json(); }
    catch { throw new Error('서버 응답을 읽지 못했습니다. 실행 중인 프로그램을 확인해 주세요.'); }
    if (!response.ok) {
      const error = new Error(data.error || data.message || `요청 실패 (${response.status})`);
      error.status = response.status;
      throw error;
    }
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('서버 응답이 늦어지고 있습니다. 잠시 후 상태를 다시 확인해 주세요.');
    throw error;
  } finally { clearTimeout(timeout); }
}
function showError(message) {
  $('form-error').textContent = message;
  $('form-error').hidden = !message;
}
function setBusy(busy) {
  state.busy = busy;
  document.querySelectorAll('#pair-form button, #pair-form input, #pair-form select').forEach(node => { node.disabled = busy; });
  $('submit-pair').querySelector('span').textContent = busy ? 'A/B 작업 진행 중' : 'A/B 연결 분석';
}
function setPreview(camera, url) {
  const key = camera.toLowerCase(), player = $('preview-' + key);
  player.pause();
  $('preview-' + key + '-error').hidden = true;
  if (url) {
    player.src = url;
    player.hidden = false;
    $('empty-' + key).hidden = true;
    player.load();
  } else {
    player.removeAttribute('src');
    player.load();
    player.hidden = true;
    $('empty-' + key).hidden = false;
  }
}
function updateSource(camera) {
  const source = state.sources[camera], key = camera.toLowerCase();
  if (source.mode === 'library') {
    const row = state.videos.find(video => video.id === $('library-' + key).value);
    source.video = row || null;
    $('source-' + key + '-state').textContent = row ? '선택 완료' : '입력 대기';
    setPreview(camera, row ? safeUrl(row.preview_url) : null);
    $('source-' + key + '-meta').textContent = row
      ? [row.width && row.height ? `${row.width} × ${row.height}` : '', Number.isFinite(row.duration_seconds) ? `${fmt(row.duration_seconds)}초` : '', row.has_review ? 'AI 검수 기록 있음' : '자동 분석 기록'].filter(Boolean).join(' · ')
      : '기존 영상의 검수 근거도 함께 불러올 수 있습니다.';
  } else {
    $('source-' + key + '-state').textContent = source.file ? '파일 선택 완료' : '입력 대기';
    setPreview(camera, source.objectUrl || null);
    $('source-' + key + '-meta').textContent = source.file
      ? `${source.file.name} · ${fmt(source.file.size / (1024 ** 2), 1)} MB`
      : `영상 파일을 선택하세요. 파일당 최대 ${fmt(state.limits.max_upload_bytes / (1024 ** 3), 1)} GB`;
  }
}
function changeMode(camera, mode) {
  if (state.busy) return;
  if (state.sources[camera].mode !== mode) $('same-event').checked = false;
  state.sources[camera].mode = mode;
  const key = camera.toLowerCase();
  $('library-' + key + '-panel').hidden = mode !== 'library';
  $('upload-' + key + '-panel').hidden = mode !== 'upload';
  document.querySelectorAll(`[data-view="${camera}"][data-mode]`).forEach(button => {
    const selected = button.dataset.mode === mode;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  $('upload-' + key + '-progress').hidden = true;
  updateSource(camera);
}
function validateSources() {
  for (const camera of cameras) {
    const source = state.sources[camera];
    if (source.mode === 'library' && !source.video) throw new Error(`${camera} 시점 영상을 선택해 주세요.`);
    if (source.mode === 'upload' && !source.file) throw new Error(`${camera} 시점 영상 파일을 선택해 주세요.`);
    if (source.mode === 'upload' && source.file.size > state.limits.max_upload_bytes) {
      throw new Error(`${camera} 시점 파일이 업로드 한도(${fmt(state.limits.max_upload_bytes / (1024 ** 3), 1)} GB)를 초과합니다.`);
    }
    if (source.mode === 'upload' && source.file.size === 0) throw new Error(`${camera} 시점 파일이 비어 있습니다.`);
  }
}
function upload(camera) {
  const source = state.sources[camera];
  if (source.mode === 'library') return Promise.resolve({video_id: source.video.id});
  if (source.uploadId) return Promise.resolve({upload_id: source.uploadId});
  const box = $('upload-' + camera.toLowerCase() + '-progress');
  box.hidden = false;
  box.querySelector('progress').value = 0;
  box.querySelector('span').textContent = '업로드 준비';
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/pair-uploads');
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.setRequestHeader('X-Filename', encodeURIComponent(source.file.name));
    xhr.timeout = 600000;
    xhr.upload.onprogress = event => {
      if (!event.lengthComputable) return;
      const percent = Math.min(100, Math.round(event.loaded / event.total * 100));
      box.querySelector('progress').value = percent;
      box.querySelector('span').textContent = percent === 100 ? '원본 확인 중' : `${percent}% 업로드`;
    };
    xhr.onload = () => {
      let data;
      try { data = JSON.parse(xhr.responseText); }
      catch { reject(new Error(`${camera} 시점 업로드 응답을 읽지 못했습니다.`)); return; }
      if (xhr.status < 200 || xhr.status >= 300 || !data.upload_id) {
        reject(new Error(data.error || `${camera} 시점 업로드에 실패했습니다.`)); return;
      }
      source.uploadId = data.upload_id;
      source.upload = data;
      box.querySelector('progress').value = 100;
      box.querySelector('span').textContent = '업로드 완료';
      resolve({upload_id: data.upload_id});
    };
    xhr.onerror = () => reject(new Error(`${camera} 시점 업로드 연결이 끊겼습니다.`));
    xhr.ontimeout = () => reject(new Error(`${camera} 시점 업로드 시간이 초과됐습니다.`));
    xhr.onabort = () => reject(new Error(`${camera} 시점 업로드가 중단됐습니다.`));
    xhr.send(source.file);
  });
}
function jobUrl(id) { return '/api/pair-jobs/' + encodeURIComponent(id); }
function viewStatus(view) {
  if (!view) return '대기';
  const label = statusNames[view.status] || '처리 중';
  return view.file ? `${label} · ${view.file}` : label;
}
function renderJob(job) {
  $('job-progress').hidden = false;
  $('job-status').textContent = statusNames[job.status] || '상태 확인 중';
  $('progress-a').textContent = viewStatus(job.views?.A);
  $('progress-b').textContent = viewStatus(job.views?.B);
  $('job-permalink').href = '/pair?job=' + encodeURIComponent(job.job_id);
  if (terminal.has(job.status)) {
    $('progress-heading').textContent = job.result ? 'A/B 연결 분석이 끝났습니다.' : '분석을 완료하지 못했습니다.';
    $('progress-message').textContent = job.error || (job.status === 'interrupted' ? '프로그램 실행이 중단된 작업입니다. 입력 영상을 확인하고 새 작업을 시작해 주세요.' : '시점별 근거와 최종 보조 판정을 아래에서 확인하세요.');
  } else {
    $('progress-heading').textContent = '두 시점의 근거를 확인하고 있습니다.';
    $('progress-message').textContent = '이 페이지를 다시 열어도 같은 작업의 진행 상태를 확인할 수 있습니다.';
  }
  if (job.result) renderResult(job);
}
function seek(player, seconds) {
  const apply = () => { player.pause(); player.currentTime = Math.min(seconds, Math.max(0, player.duration - 0.001)); };
  if (player.readyState >= 1) apply();
  else { player.addEventListener('loadedmetadata', apply, {once: true}); player.load(); }
  player.scrollIntoView({block: 'center', behavior: 'auto'});
}
function renderView(camera, view) {
  const card = el('article', 'result-view');
  card.id = 'result-view-' + camera.toLowerCase();
  const header = el('div', 'result-view-header');
  header.append(el('span', 'camera-mark' + (camera === 'B' ? ' camera-b' : ''), camera));
  const title = el('div');
  title.append(el('p', 'eyebrow', camera + ' VIEW'), el('h3', '', view.file || camera + ' 시점'));
  header.append(title, el('span', 'processing-label', modeNames[view.processing_mode] || '결과 확인'));
  card.append(header);
  const verdicts = el('div', 'view-verdicts');
  for (const [label, value] of [['자동 분석', view.automatic_decision], ['직접 관측 근거', view.direct_decision], ['시점 보조 판정', view.assisted_decision]]) {
    const cell = el('div');
    cell.append(el('span', '', label), el('strong', '', value || 'REVIEW'));
    verdicts.append(cell);
  }
  card.append(verdicts);
  const preview = safeUrl(view.preview_url);
  let player;
  if (preview) {
    player = el('video', 'view-video');
    player.controls = true; player.preload = 'metadata'; player.playsInline = true;
    player.setAttribute('aria-label', camera + ' 시점 결과 영상'); player.src = preview;
    card.append(player);
  } else { card.append(el('div', 'missing-media', '이 시점의 재생 가능한 미리보기가 없습니다.')); }
  const body = el('div', 'view-body');
  if (view.error) body.append(el('p', 'message error', view.error));
  if (view.rationale) body.append(el('p', 'rationale', view.rationale));
  if (player && typeof view.contact_time_seconds === 'number' && Number.isFinite(view.contact_time_seconds) && view.contact_time_seconds >= 0) {
    const button = el('button', 'seek-button', `이 시점의 접촉 ${fmt(view.contact_time_seconds, 3)}초로 이동`);
    button.type = 'button'; button.dataset.seekCamera = camera;
    button.addEventListener('click', () => seek(player, view.contact_time_seconds)); body.append(button);
  }
  if (Array.isArray(view.evidence_urls) && view.evidence_urls.length) {
    const strip = el('div', 'evidence-strip'); strip.setAttribute('aria-label', camera + ' 시점 확대 근거');
    view.evidence_urls.forEach((url, index) => {
      const path = safeUrl(url); if (!path) return;
      const link = el('a'); link.href = path; link.target = '_blank'; link.rel = 'noopener';
      const image = el('img'); image.src = path; image.loading = 'lazy'; image.alt = `${camera} 시점 검수 근거 ${index + 1}`;
      link.append(image, el('span', '', `확대 근거 ${index + 1} ↗`)); strip.append(link);
    }); body.append(strip);
  }
  const details = el('details', 'provenance-details');
  details.append(el('summary', '', '판정 출처와 원본 확인'));
  details.append(el('p', '', `검수 출처: ${view.review_source === 'ai_visual_review' ? 'AI 시각 검수' : view.review_source || '자동 분석'}`));
  if (view.source_sha256) details.append(el('p', '', '원본 SHA256: ' + view.source_sha256));
  if (Array.isArray(view.reason_codes) && view.reason_codes.length) details.append(el('p', '', view.reason_codes.join(' · ')));
  body.append(details); card.append(body); return card;
}
function renderResult(job) {
  const result = job.result;
  if (!result || !['IN', 'OUT', 'REVIEW'].includes(result.decision)) throw new Error('최종 판정 응답 형식이 올바르지 않습니다.');
  $('pair-result').hidden = false;
  $('result-banner').dataset.verdict = result.decision;
  $('final-verdict').textContent = result.decision;
  const views = result.views || {};
  const aiUsed = cameras.some(camera => views[camera]?.processing_mode === 'stored_ai_review');
  $('result-provenance').textContent = aiUsed ? '기존 AI 검수 근거를 활용한 보조 결과' : '두 시점의 분석·대응 검토 결과';
  const sources = $('source-cameras'); sources.replaceChildren();
  const sourceIds = Array.isArray(result.source_camera_ids) ? result.source_camera_ids.filter(id => cameras.includes(id)) : [];
  sources.append(el('span', '', sourceIds.length ? '최종 판정의 접촉 근거' : '확인된 접촉 근거 추가 필요'));
  sourceIds.forEach(camera => { const link = el('a', 'source-link', camera + ' 시점 보기 ↓'); link.href = '#result-view-' + camera.toLowerCase(); sources.append(link); });
  const reasons = $('result-reasons'); reasons.replaceChildren();
  const messages = Array.isArray(result.reasons) ? result.reasons : [];
  (messages.length ? messages : ['시점별 판정 근거를 확인해 주세요.']).forEach(message => reasons.append(el('li', '', String(message))));
  const eventSource = correspondenceNames[result.event_correspondence_source] || '저장된 사건 대응 기록';
  $('correspondence-status').textContent = result.same_event_confirmed ? `같은 사건 연결 확인 · ${eventSource}` : '동일 사건 연결이 확인되지 않았습니다.';
  $('result-timing').textContent = typeof result.elapsed_seconds === 'number' ? `이번 작업 ${fmt(result.elapsed_seconds, 1)}초${aiUsed ? ' · 기존 검수 시간은 포함하지 않습니다.' : ''}` : '';
  $('result-json').href = jobUrl(job.job_id);
  $('result-json').download = 'pair-' + job.job_id + '.json';
  const grid = $('result-views'); grid.replaceChildren();
  cameras.forEach(camera => grid.append(renderView(camera, views[camera] || {})));
}
async function pollJob(id, generation) {
  if (generation !== state.pollGeneration) return;
  try {
    const job = await api(jobUrl(id));
    if (generation !== state.pollGeneration) return;
    if (job.job_id !== id) throw new Error('서버의 작업 ID가 요청과 다릅니다.');
    renderJob(job); $('retry-poll').hidden = true;
    if (terminal.has(job.status)) { setBusy(false); await loadHistory(); return; }
    state.pollTimer = setTimeout(() => pollJob(id, generation), 1200);
  } catch (error) {
    if (generation !== state.pollGeneration) return;
    $('progress-message').textContent = error.message;
    const unavailable = [400, 403, 404].includes(error.status);
    $('retry-poll').hidden = unavailable;
    if (unavailable) {
      $('progress-heading').textContent = '작업을 열 수 없습니다.';
      $('job-status').textContent = '작업 확인 필요';
      setBusy(false);
    }
  }
}
function openJob(id, push = false) {
  if (!/^[a-zA-Z0-9_-]{1,128}$/.test(id)) { showError('잘못된 작업 ID입니다.'); return; }
  clearTimeout(state.pollTimer); state.pollGeneration += 1; state.jobId = id;
  $('pair-result').hidden = true; $('job-progress').hidden = false;
  $('progress-heading').textContent = '작업 상태를 불러오고 있습니다.';
  $('progress-a').textContent = '확인 중'; $('progress-b').textContent = '확인 중';
  $('retry-poll').hidden = true; showError(''); setBusy(true);
  if (push) history.pushState({}, '', '/pair?job=' + encodeURIComponent(id));
  pollJob(id, state.pollGeneration);
}
async function loadHistory() {
  try {
    const data = await api('/api/pair-jobs');
    const list = $('job-history'); list.replaceChildren();
    const jobs = Array.isArray(data.jobs) ? data.jobs : [];
    if (!jobs.length) { list.append(el('p', 'muted', '아직 A/B 작업이 없습니다. 두 영상을 선택해 시작하세요.')); return; }
    jobs.slice(0, 12).forEach(job => {
      if (!job.job_id) return;
      const item = el('a', 'history-item'); item.href = '/pair?job=' + encodeURIComponent(job.job_id);
      const names = cameras.map(camera => job.views?.[camera]?.file).filter(Boolean);
      const name = el('div', 'history-name', names.length ? names.join(' ↔ ') : 'A/B 작업');
      const date = new Date(job.created_at || job.updated_at || '');
      name.append(el('small', '', Number.isNaN(date.getTime()) ? job.job_id : date.toLocaleString('ko-KR', {timeZone: 'Asia/Seoul'}) + ' KST'));
      const decision = job.decision || job.result?.decision;
      const label = el('span', 'history-label', decision || statusNames[job.status] || '확인 중');
      if (decision) label.dataset.verdict = decision;
      item.append(name, label);
      item.addEventListener('click', event => { if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return; event.preventDefault(); openJob(job.job_id, true); $('job-progress').scrollIntoView({block: 'start'}); });
      list.append(item);
    });
  } catch (error) { $('job-history').replaceChildren(el('p', 'muted', '작업 목록을 불러오지 못했습니다. ' + error.message)); }
}

document.querySelectorAll('[data-mode][data-view]').forEach(button => button.addEventListener('click', () => { showError(''); changeMode(button.dataset.view, button.dataset.mode); }));
for (const camera of cameras) {
  const key = camera.toLowerCase();
  $('library-' + key).addEventListener('change', () => { showError(''); $('same-event').checked = false; updateSource(camera); });
  $('file-' + key).addEventListener('change', event => {
    showError('');
    const source = state.sources[camera];
    if (source.objectUrl) URL.revokeObjectURL(source.objectUrl);
    source.file = event.target.files[0] || null;
    source.uploadId = null; source.upload = null;
    $('same-event').checked = false;
    source.objectUrl = source.file ? URL.createObjectURL(source.file) : null;
    $('file-' + key + '-name').textContent = source.file?.name || 'MP4 · MOV · MKV · AVI · WEBM';
    $('upload-' + key + '-progress').hidden = true; updateSource(camera);
  });
  $('preview-' + key).addEventListener('error', () => { if (!$('preview-' + key).hidden) $('preview-' + key + '-error').hidden = false; });
}
$('pair-form').addEventListener('submit', async event => {
  event.preventDefault(); if (state.busy) return;
  showError('');
  try { validateSources(); } catch (error) { showError(error.message); return; }
  setBusy(true); $('pair-result').hidden = true;
  try {
    // Both original files are staged concurrently; the analysis begins only
    // after both inputs have been received and validated by the server.
    const uploads = await Promise.allSettled(cameras.map(upload));
    const failure = uploads.find(result => result.status === 'rejected');
    if (failure) throw failure.reason;
    const views = {A: uploads[0].value, B: uploads[1].value};
    const job = await api('/api/pair-jobs', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({views, same_event_confirmed: $('same-event').checked, reuse_review: $('reuse-review').checked})});
    if (!job.job_id) throw new Error('작업 ID가 응답에 없습니다. 최근 작업 목록을 확인해 주세요.');
    openJob(job.job_id, true);
    $('job-progress').scrollIntoView({block: 'start', behavior: 'auto'});
    loadHistory();
  } catch (error) { showError(error.message); setBusy(false); loadHistory(); }
});
$('retry-poll').addEventListener('click', () => { if (state.jobId) { $('retry-poll').hidden = true; pollJob(state.jobId, state.pollGeneration); } });
$('refresh-history').addEventListener('click', loadHistory);
window.addEventListener('popstate', () => {
  const id = new URLSearchParams(location.search).get('job');
  if (id) openJob(id);
  else { clearTimeout(state.pollTimer); state.pollGeneration += 1; state.jobId = null; setBusy(false); $('job-progress').hidden = true; $('pair-result').hidden = true; }
});
async function initialize() {
  try {
    const library = await api('/api/pair-library');
    state.videos = Array.isArray(library.videos) ? library.videos : [];
    if (library.limits?.max_upload_bytes > 0) state.limits = library.limits;
    $('library-count').textContent = `라이브러리 ${state.videos.length}개 영상`;
    for (const camera of cameras) {
      const select = $('library-' + camera.toLowerCase());
      state.videos.forEach(video => { const option = el('option', '', video.file); option.value = video.id; select.append(option); });
      if (!state.videos.length) changeMode(camera, 'upload');
    }
  } catch (error) {
    $('library-count').textContent = '라이브러리 연결 확인 필요';
    showError('라이브러리를 불러오지 못했습니다. ' + error.message);
    cameras.forEach(camera => changeMode(camera, 'upload'));
  }
  loadHistory();
  const id = new URLSearchParams(location.search).get('job');
  if (id) openJob(id);
}
initialize();
