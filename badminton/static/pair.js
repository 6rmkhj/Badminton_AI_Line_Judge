'use strict';

const $ = id => document.getElementById(id);
const cameras = ['A', 'B'];
const state = {
  videos: [], limits: {max_upload_bytes: 1024 ** 3}, busy: false,
  sources: {A: {mode: 'library'}, B: {mode: 'library'}},
  jobId: null, currentJob: null, pollTimer: null, pollGeneration: 0,
  dashboardJob: null, receivedAt: 0, connected: true, frozenDelta: 0, announcement: '',
  transfer: null, submissionError: '',
};
const statusNames = {
  queued: '분석 대기', pending: '대기', running: '분석 중', analyzing: '영상 분석 중',
  resolving: '원본 확인 중', verifying: '검수 근거 확인 중', loading: '기존 근거 확인 중',
  completed: '완료', completed_with_errors: '일부 처리 실패', failed: '처리 실패', interrupted: '실행 중단',
  reused: '기존 근거 연결 완료', stored_ai_review: '기존 AI 검수 연결', fresh_analysis: '새 영상 분석',
};
const terminal = new Set(['completed', 'completed_with_errors', 'failed', 'interrupted']);
const analysisStages = [
  ['prepare', '영상 준비'], ['calibration', '코트 확인'], ['detection', '셔틀콕 후보 탐지'],
  ['tracking', '궤적 연결'], ['contact', '착지·라인 확인'], ['evidence', '근거 영상 생성'], ['save', '결과 저장'],
];
const workflowLabels = ['입력 전송', '분석 대기', '시점별 분석', '결과 연결'];
function secondsText(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return '—';
  const seconds = Math.floor(value), hours = Math.floor(seconds / 3600);
  return hours ? `${hours}:${String(Math.floor(seconds / 60) % 60).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`
    : `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
}
function durationText(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? secondsText(value) : '—';
}
function validSeconds(value) { return typeof value === 'number' && Number.isFinite(value) && value >= 0; }
function tickDelta() { return state.connected ? Math.max(0, (performance.now() - state.receivedAt) / 1000) : state.frozenDelta; }
function announce(message) {
  if (message !== state.announcement) { $('progress-announcement').textContent = message; state.announcement = message; }
}
function currentPhase(job) {
  if (job.status === 'queued') return 'queued';
  if (terminal.has(job.status)) return job.status;
  return job.phase === 'linking' ? 'linking' : 'analyzing';
}
function setConnection(message = '') {
  const box = $('connection-status'); box.hidden = !message; box.textContent = message;
  $('job-progress').classList.toggle('connection-lost', Boolean(message));
}
function drawWorkflow(phase, error = false, lastPhase = null) {
  const positions = {upload: 0, queued: 1, analyzing: 2, linking: 3, completed: 4, completed_with_errors: 4};
  const effective = error && phase !== 'completed_with_errors' && Object.hasOwn(positions, lastPhase) ? lastPhase : phase;
  const active = positions[effective] ?? 2;
  const list = $('workflow-steps'); list.replaceChildren();
  workflowLabels.forEach((label, index) => {
    const errorHere = error && (index === active || phase === 'completed_with_errors' && index === 2);
    const done = index < active && !errorHere;
    const current = index === active;
    const item = el('li', errorHere ? 'attention' : done ? 'done' : current ? 'current' : 'pending');
    if (current) item.setAttribute('aria-current', 'step');
    item.append(el('span', 'workflow-number', done ? '✓' : String(index + 1)), el('span', '', label));
    if (current) item.append(el('small', '', error ? '확인 필요' : '진행 중'));
    list.append(item);
  });
}
function beginSubmission(refineJob = null) {
  clearTimeout(state.pollTimer); state.pollGeneration += 1;
  state.jobId = null; state.dashboardJob = null; state.currentJob = null;
  state.connected = true; state.submissionError = ''; state.announcement = '';
  state.transfer = {started: performance.now(), stopped: null, views: {}};
  for (const camera of cameras) {
    const source = state.sources[camera];
    const ready = Boolean(refineJob || source.mode === 'library' || source.uploadId);
    state.transfer.views[camera] = {status: ready ? 'ready' : 'uploading', loaded: 0,
      total: !refineJob && source.mode === 'upload' ? source.file?.size || 0 : 0,
      file: refineJob?.views?.[camera]?.file || (source.mode === 'upload' ? source.file?.name : source.video?.file) || camera + ' 시점'};
  }
  $('pair-result').hidden = true; $('job-progress').hidden = false;
  $('job-permalink').hidden = true; $('retry-poll').hidden = true;
  setConnection(); renderTransfer();
  $('job-progress').scrollIntoView({block: 'start', behavior: 'auto'});
}
function renderTransfer() {
  if (!state.transfer || state.dashboardJob) return;
  const failed = Boolean(state.submissionError);
  const ready = cameras.filter(camera => state.transfer.views[camera].status === 'ready').length;
  $('progress-heading').textContent = failed ? '입력 준비를 완료하지 못했습니다.' : ready === 2 ? '분석 작업을 준비하고 있습니다.' : '분석할 영상을 준비하고 있습니다.';
  $('job-status').textContent = failed ? '입력 확인 필요' : ready === 2 ? '작업 생성 중' : '입력 준비 중';
  $('job-status').dataset.state = failed ? 'failed' : 'upload';
  $('overall-progress').textContent = `입력 ${ready} / 2개 준비 완료`;
  $('progress-message').textContent = state.submissionError || (ready === 2 ? '두 영상의 입력 확인이 끝났습니다. 분석 작업을 만들고 있습니다.' : '전송이 끝나면 프로그램이 영상 형식과 프레임 수를 확인합니다.');
  $('elapsed-label').textContent = '입력 준비 경과';
  $('elapsed-caption').textContent = failed ? '실패 시점까지의 시간' : '업로드와 원본 확인 시간';
  $('queue-time').hidden = true; drawWorkflow('upload', failed);
  const grid = $('progress-views'); grid.replaceChildren();
  for (const camera of cameras) grid.append(transferCard(camera, state.transfer.views[camera]));
  updateClocks();
  announce(failed ? state.submissionError : ready === 2 ? '두 영상 입력 확인 완료. 분석 작업을 준비합니다.' : '영상 입력을 준비하고 있습니다.');
}
function cardHeader(camera, file, status) {
  const header = el('div', 'analysis-card-header');
  header.append(el('span', 'mini-camera' + (camera === 'B' ? ' camera-b' : ''), camera));
  const title = el('div', 'analysis-card-title'); title.append(el('strong', '', camera + ' 시점'), el('span', 'analysis-filename', file || '영상 확인 중'));
  header.append(title, el('span', 'analysis-card-status', status)); return header;
}
function transferCard(camera, view) {
  const card = el('article', 'analysis-card transfer-card'); card.dataset.camera = camera;
  const labels = {uploading: '전송 중', checking: '원본 확인 중', ready: '입력 준비 완료', failed: '입력 실패'};
  card.dataset.state = view.status;
  card.append(cardHeader(camera, view.file, labels[view.status] || '대기'));
  const title = el('h3', 'analysis-stage-title', view.status === 'checking' ? '전송 완료 · 영상을 확인합니다.' : labels[view.status]);
  title.id = 'progress-' + camera.toLowerCase(); card.append(title);
  const bar = el('progress', 'frame-progress'); bar.max = view.total || 1;
  if (view.status === 'uploading' && !view.loaded) bar.removeAttribute('value');
  else bar.value = view.status === 'ready' ? (view.total || 1) : Math.min(view.loaded, view.total || 1);
  bar.setAttribute('aria-label', camera + ' 시점 파일 전송');
  card.append(bar);
  const details = view.total ? `${fmt(view.loaded / 1024 ** 2, 1)} / ${fmt(view.total / 1024 ** 2, 1)} MB 전송` : '기존 영상 입력';
  card.append(el('p', 'frame-count', view.status === 'ready' ? '✓ 입력 확인 완료' : details));
  card.append(el('p', 'analysis-detail', view.status === 'checking' ? '영상 정보를 읽고 있습니다. 파일 전송과 별도로 시간이 걸릴 수 있습니다.' : view.status === 'failed' ? '입력을 확인하고 다시 분석을 시작해 주세요.' : view.status === 'ready' ? '두 시점이 모두 준비되면 분석을 시작합니다.' : '이 창을 열어 두세요. 두 시점을 함께 전송합니다.'));
  return card;
}
function renderAnalysisCard(camera, view, job) {
  const progress = view.progress || {}, running = !terminal.has(job.status) && !terminal.has(view.status) && job.status !== 'queued' && view.status !== 'queued';
  const done = view.status === 'completed', failed = ['failed', 'interrupted'].includes(view.status);
  const reused = view.processing_mode === 'stored_ai_review' || progress.last_stage === 'stored_review' || progress.stage === 'stored_review';
  const index = Number.isInteger(progress.stage_index) && progress.stage_index >= 1 && progress.stage_index <= 7 ? progress.stage_index : null;
  const known = typeof progress.stage === 'string';
  const stageLabel = done ? (reused ? '기존 검수 근거를 불러왔습니다.' : '시점 분석을 마쳤습니다.') : progress.stage_label || statusNames[view.status] || '상태 확인 중';
  const card = el('article', 'analysis-card'); card.dataset.camera = camera; card.dataset.state = failed ? 'failed' : done ? 'completed' : 'running';
  card.append(cardHeader(camera, view.file, failed ? statusNames[view.status] : done ? '처리 완료' : running && progress.stage ? '처리 중' : statusNames[view.status] || '대기'));
  const heading = el('div', 'analysis-current');
  heading.append(el('span', 'stage-position', reused ? '저장된 근거 활용' : known && index ? `${done ? 7 : index} / 7 단계` : done ? '처리 완료' : '현재 상태'));
  const title = el('h3', 'analysis-stage-title', stageLabel); title.id = 'progress-' + camera.toLowerCase(); heading.append(title);
  const stageClock = el('span', 'stage-clock'); stageClock.dataset.clock = 'stage'; stageClock.dataset.camera = camera; heading.append(stageClock);
  card.append(heading);
  if (validSeconds(progress.total_units) && progress.total_units > 0 && validSeconds(progress.completed_units)) {
    const count = Math.min(progress.total_units, progress.completed_units), bar = el('progress', 'frame-progress');
    bar.max = progress.total_units; bar.value = count;
    bar.setAttribute('aria-label', `${camera} 시점 ${progress.stage_label || '현재 단계'} 프레임 처리`);
    card.append(bar, el('p', 'frame-count', `${count.toLocaleString('ko-KR')} / ${progress.total_units.toLocaleString('ko-KR')} 프레임 처리`));
  }
  if (progress.detail && typeof progress.detail === 'string') card.append(el('p', 'analysis-detail', progress.detail));
  if (known && !reused) {
    const list = el('ol', 'analysis-stages'); list.setAttribute('aria-label', camera + ' 시점 분석 단계');
    const history = Array.isArray(view.stage_history) ? view.stage_history : [];
    analysisStages.forEach(([id, label], ordinal) => {
      const record = history.find(item => item.stage === id);
      const active = progress.stage === id || progress.last_stage === id && failed;
      const finished = record?.status === 'completed' || done && index === 7 || (index && ordinal + 1 < index);
      const status = active && failed ? 'failed' : finished ? 'done' : active ? 'current' : 'pending';
      const item = el('li', status); item.dataset.stage = id;
      item.append(el('span', 'stage-check', status === 'done' ? '✓' : status === 'failed' ? '!' : String(ordinal + 1)), el('span', 'stage-name', label));
      const stamp = el('span', 'stage-duration', status === 'done' ? durationText(record?.elapsed_seconds) : status === 'pending' ? '대기' : status === 'failed' ? '중단' : '진행 중');
      if (status === 'current') { stamp.dataset.clock = 'stage-row'; stamp.dataset.camera = camera; }
      item.append(stamp); if (status === 'current') item.setAttribute('aria-current', 'step');
      list.append(item);
    });
    card.append(list);
  } else card.append(el('p', 'analysis-detail', reused ? '저장된 검수의 원본 일치와 근거를 확인합니다. 새 프레임 분석은 실행하지 않습니다.' : '이 작업에는 세부 단계 기록이 없습니다. 처리 상태와 저장된 시간을 표시합니다.'));
  const footer = el('div', 'analysis-card-footer'); footer.append(el('span', '', '시점 처리 시간'));
  const elapsed = el('strong'); elapsed.dataset.clock = 'view'; elapsed.dataset.camera = camera; footer.append(elapsed); card.append(footer);
  if (view.error) card.append(el('p', 'analysis-detail analysis-error', view.error));
  return card;
}
function updateClocks() {
  if (state.transfer && !state.dashboardJob) {
    $('job-elapsed').textContent = secondsText(((state.transfer.stopped ?? performance.now()) - state.transfer.started) / 1000); return;
  }
  const job = state.dashboardJob; if (!job) return;
  const live = !terminal.has(job.status) && job.status !== 'queued';
  $('job-elapsed').textContent = validSeconds(job.elapsed_seconds) ? secondsText(job.elapsed_seconds + (live ? tickDelta() : 0)) : '—';
  if (validSeconds(job.queue_seconds)) {
    $('queue-time').textContent = `분석 전 대기 ${secondsText(job.queue_seconds + (job.status === 'queued' ? tickDelta() : 0))}`;
  }
  document.querySelectorAll('[data-clock][data-camera]').forEach(node => {
    const view = job.views?.[node.dataset.camera] || {}, progress = view.progress || {};
    const advancing = live && !terminal.has(view.status) && view.status !== 'queued';
    if (node.dataset.clock === 'view') {
      node.textContent = validSeconds(view.elapsed_seconds) ? secondsText(view.elapsed_seconds + (advancing ? tickDelta() : 0)) : '—'; return;
    }
    const stageSeconds = progress.stage_elapsed_seconds;
    const seconds = validSeconds(stageSeconds) ? stageSeconds + (advancing ? tickDelta() : 0) : null;
    const prefix = ['failed', 'interrupted'].includes(view.status) ? '중단 단계' : view.status === 'completed' ? '마지막 단계' : '현재 단계';
    node.textContent = seconds === null ? '' : node.dataset.clock === 'stage' ? `${prefix} ${secondsText(seconds)}` : secondsText(seconds);
  });
  if (live && state.connected && tickDelta() > 10 && $('connection-status').hidden) {
    setConnection('상태 응답을 기다리고 있습니다. 표시된 단계는 마지막으로 확인한 상태입니다.');
  }
}
setInterval(updateClocks, 1000);
const modeNames = {stored_ai_review: '기존 AI 검수', fresh_analysis: '새 분석', stored_automatic: '기존 자동 분석', review_guided_reanalysis: '원본 집중 재분석', failed: '처리 실패'};
const correspondenceNames = {
  ai_visual_review: '저장된 AI 검수의 동작 대응', reviewed_actions: '검수된 동작 대응',
  stored_ai_review: '저장된 AI 검수의 동작 대응', operator_selection: '사용자가 확인한 사건 대응',
  user_selection: '사용자가 확인한 사건 대응', operator_request: '사용자가 확인한 사건 대응',
  verified_ai_action_correspondence: '저장된 AI 검수의 동작 대응',
  operator_pair_selection: '사용자가 확인한 사건 대응',
};
const observationNames = {
  line_ambiguous: '접촉과 라인 관계 불명확', contact_occluded: '접촉 구간 가림',
  contact_out_of_view: '접촉 위치 관측 불가', no_contact_in_clip: '클립에 접촉 구간 없음',
  target_ambiguous: '대상 셔틀 식별 불명확', contact_observed: '접촉 관측',
  observed: '접촉 관측', contact_confirmed: '접촉 확인', contact_interval: '최초 접촉 구간 관측',
  automatic_unresolved: '자동 접촉 검증 미완료', failed: '영상 처리 실패',
};
const requirementNames = {
  same_event: '같은 착지 사건의 시점 대응', event_correspondence: '같은 착지 사건의 시점 대응',
  contact_observed: '최초 바닥 접촉 관측', first_contact: '최초 바닥 접촉 관측',
  target_identity: '해당 랠리의 셔틀 식별', target_contact_identity_confirmed: '해당 랠리의 셔틀 식별',
  boundary_evidence: '접촉점과 실제 라인 경계의 관계', boundary: '접촉점과 실제 라인 경계의 관계',
  source_integrity: '원본 영상 일치 확인',
};
const fallbackIssues = {
  SAME_EVENT_UNCONFIRMED: ['A/B 사건 대응 미확인', '두 영상이 같은 랠리의 같은 착지 사건인지 확인되지 않았습니다.', '양쪽에서 같은 타격과 후속 동작을 대조하세요. 확인한 경우에만 같은 사건 항목을 선택하고 다시 제출하세요.'],
  KNOWN_DIFFERENT_RALLY: ['서로 다른 랠리', '기존 동작 검수에서 서로 다른 랠리로 확인된 입력입니다.', '한쪽 영상을 같은 착지 사건이 담긴 올바른 반대 시점 영상으로 교체하세요.'],
  DUPLICATE_SOURCE: ['동일 영상 중복 입력', 'A와 B의 영상 내용이 같습니다. 두 입력이 다른 시점의 관측을 제공하지 않습니다.', '같은 사건을 다른 위치에서 촬영한 영상을 선택하세요.'],
  NO_QUALIFIED_CONTACT: ['유효한 최초 접촉 근거 부족', '양쪽 모두 최초 접촉과 라인 관계를 확정할 근거가 부족합니다.', '아래 시점별 관측 사유와 연속 프레임 근거를 확인하세요. 최초 접촉이 촬영되지 않았다면 해당 구간이 보이는 원본이 필요합니다.'],
  CONFLICTING_VIEW_DECISIONS: ['시점별 판정 충돌', '두 시점의 직접 판정이 서로 다릅니다.', '같은 착지 사건을 판단했는지 확인하고 접촉점·라인 경계·적용 규칙을 다시 검토하세요.'],
  VIEW_ANALYSIS_FAILED: ['영상 처리 실패', '한쪽 영상의 분석을 완료하지 못했습니다.', '아래 시점의 오류를 확인하고 입력 파일이나 실행 환경을 수정한 뒤 다시 분석하세요.'],
  SOURCE_CHANGED: ['원본 일치 확인 실패', '현재 입력과 저장된 근거의 원본 일치를 확인하지 못했습니다.', '올바른 원본을 선택하거나 기존 검수 활용을 끄고 새로 분석하세요.'],
  RULES_MISMATCH: ['경기 규칙 불일치', '두 시점에 적용된 경기 규칙이 다릅니다.', '단식·복식 및 서비스·랠리 구분을 확인하고 같은 규칙으로 다시 검토하세요.'],
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
  $('refine-pair').disabled = busy;
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
    const updateTransfer = (status, loaded) => {
      const transfer = state.transfer?.views[camera];
      if (transfer) { transfer.status = status; if (loaded !== undefined) transfer.loaded = loaded; renderTransfer(); }
    };
    const fail = message => { updateTransfer('failed'); reject(new Error(message)); };
    xhr.open('POST', '/api/pair-uploads');
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.setRequestHeader('X-Filename', encodeURIComponent(source.file.name));
    xhr.timeout = 600000;
    xhr.upload.onprogress = event => {
      if (!event.lengthComputable) return;
      const percent = Math.min(100, Math.round(event.loaded / event.total * 100));
      box.querySelector('progress').value = percent;
      box.querySelector('span').textContent = percent === 100 ? '원본 확인 중' : `${percent}% 업로드`;
      updateTransfer(percent === 100 ? 'checking' : 'uploading', event.loaded);
    };
    xhr.upload.onload = () => updateTransfer('checking', source.file.size);
    xhr.onload = () => {
      let data;
      try { data = JSON.parse(xhr.responseText); }
      catch { fail(`${camera} 시점 업로드 응답을 읽지 못했습니다.`); return; }
      if (xhr.status < 200 || xhr.status >= 300 || !data.upload_id) {
        fail(data.error || `${camera} 시점 업로드에 실패했습니다.`); return;
      }
      source.uploadId = data.upload_id;
      source.upload = data;
      box.querySelector('progress').value = 100;
      box.querySelector('span').textContent = '업로드 완료';
      updateTransfer('ready', source.file.size);
      resolve({upload_id: data.upload_id});
    };
    xhr.onerror = () => fail(`${camera} 시점 업로드 연결이 끊겼습니다.`);
    xhr.ontimeout = () => fail(`${camera} 시점 업로드 시간이 초과됐습니다.`);
    xhr.onabort = () => fail(`${camera} 시점 업로드가 중단됐습니다.`);
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
  state.dashboardJob = job; state.receivedAt = performance.now(); state.transfer = null;
  state.connected = true; state.frozenDelta = 0; setConnection();
  $('job-progress').hidden = false;
  $('job-status').textContent = statusNames[job.status] || '상태 확인 중';
  $('job-status').dataset.state = job.status;
  const phase = currentPhase(job), failed = ['failed', 'interrupted', 'completed_with_errors'].includes(job.status);
  const completed = cameras.filter(camera => job.views?.[camera]?.status === 'completed').length;
  $('overall-progress').textContent = phase === 'queued' ? '분석 대기열에 등록되었습니다.' : phase === 'linking' ? 'A/B 근거를 하나의 결과로 연결 중' : `시점 ${completed} / 2개 처리 완료`;
  $('elapsed-label').textContent = '분석 경과';
  $('elapsed-caption').textContent = terminal.has(job.status) ? '작업 종료 시점의 처리 시간' : phase === 'queued' ? '분석이 시작되면 시간을 표시합니다.' : '대기 시간을 제외한 실제 처리 시간';
  $('queue-time').hidden = !validSeconds(job.queue_seconds);
  const lastPhase = job.last_phase || (job.started_at === null ? 'queued' : cameras.every(camera => job.views?.[camera]?.status === 'completed') ? 'linking' : 'analyzing');
  drawWorkflow(phase, failed, lastPhase);
  const grid = $('progress-views'); grid.replaceChildren();
  for (const camera of cameras) grid.append(renderAnalysisCard(camera, job.views?.[camera] || {}, job));
  $('job-permalink').href = '/pair?job=' + encodeURIComponent(job.job_id);
  $('job-permalink').hidden = false;
  if (terminal.has(job.status)) {
    $('progress-heading').textContent = failed ? job.result ? '일부 시점의 처리를 완료하지 못했습니다.' : lastPhase === 'linking' ? '두 시점의 결과를 연결하지 못했습니다.' : lastPhase === 'queued' ? '분석 대기 중 작업이 중단되었습니다.' : '분석을 완료하지 못했습니다.' : 'A/B 연결 분석이 끝났습니다.';
    $('progress-message').textContent = job.error || (job.status === 'interrupted' ? '프로그램 실행이 중단된 작업입니다. 입력 영상을 확인하고 새 작업을 시작해 주세요.' : '시점별 근거와 최종 보조 판정을 아래에서 확인하세요.');
  } else {
    $('progress-heading').textContent = phase === 'queued' ? '분석을 시작할 차례를 기다립니다.' : phase === 'linking' ? '두 시점의 결과를 연결하고 있습니다.' : 'A/B 영상을 분석하고 있습니다.';
    $('progress-message').textContent = phase === 'queued' ? '앞선 작업이 끝나면 자동으로 시작합니다. 창을 열어 두세요.' : phase === 'linking' ? '사건 대응과 시점별 판정 근거를 함께 확인해 최종 결과를 저장합니다.' : '시점별 현재 단계와 처리 프레임을 아래에서 확인하세요. 이 작업은 다시 열어도 이어서 확인할 수 있습니다.';
  }
  updateClocks();
  announce(`${$('job-status').textContent}. ` + cameras.map(camera => `${camera} 시점 ${job.views?.[camera]?.progress?.stage_label || statusNames[job.views?.[camera]?.status] || '확인 중'}`).join('. '));
  if (job.result) renderResult(job);
}
function seek(player, seconds) {
  const apply = () => { player.pause(); player.currentTime = Math.min(seconds, Math.max(0, player.duration - 0.001)); };
  if (player.readyState >= 1) apply();
  else { player.addEventListener('loadedmetadata', apply, {once: true}); player.load(); }
  player.scrollIntoView({block: 'center', behavior: 'auto'});
}
function textItems(value) {
  const items = Array.isArray(value) ? value : value ? [value] : [];
  return items.map(item => {
    if (typeof item === 'string') return requirementNames[item] || item;
    if (item && typeof item === 'object') return item.detail || item.title || item.message || requirementNames[item.code] || item.code || '';
    return '';
  }).filter(item => typeof item === 'string' && item.trim());
}
function appendTextList(parent, label, values, className = '') {
  const items = textItems(values);
  if (!items.length) return;
  const group = el('div', className);
  group.append(el('strong', 'diagnostic-label', label));
  const list = el('ul');
  items.forEach(item => list.append(el('li', '', item)));
  group.append(list); parent.append(group);
}
function renderDiagnostics(result) {
  const panel = $('review-diagnostics'), grid = $('diagnostics-issues');
  panel.hidden = result.decision !== 'REVIEW'; grid.replaceChildren();
  $('refine-panel').hidden = panel.hidden || result.refinement_available !== true;
  $('refine-error').hidden = true;
  if (panel.hidden) return;
  const diagnostics = result.review_diagnostics || {};
  let issues = Array.isArray(diagnostics.issues) ? diagnostics.issues.filter(issue => issue && typeof issue === 'object') : [];
  if (!issues.length) {
    issues = (Array.isArray(result.reason_codes) ? result.reason_codes : []).map(code => {
      const entry = fallbackIssues[code];
      return entry ? {code, title: entry[0], detail: entry[1], remediation: entry[2], camera_ids: cameras} : null;
    }).filter(Boolean);
  }
  $('diagnostics-count').textContent = issues.length ? `${issues.length}개 확인 항목` : '시점별 근거 확인';
  $('diagnostics-summary').textContent = typeof diagnostics.summary === 'string' && diagnostics.summary
    ? diagnostics.summary : '최종 판정을 보류한 이유와 필요한 후속 조치를 구분했습니다. 같은 사건 확인만으로 접촉·라인 근거가 생기지는 않습니다.';
  issues.forEach(issue => {
    const card = el('article', 'diagnostic-issue');
    if (typeof issue.code === 'string') card.dataset.issueCode = issue.code;
    const header = el('div', 'diagnostic-issue-heading');
    header.append(el('h3', '', typeof issue.title === 'string' ? issue.title : '추가 확인 필요'));
    const cameraIds = Array.isArray(issue.camera_ids) ? issue.camera_ids.filter(camera => cameras.includes(camera)) : [];
    if (cameraIds.length) header.append(el('span', 'diagnostic-cameras', cameraIds.join(' / ') + ' 시점'));
    card.append(header);
    textItems(issue.detail).forEach(detail => card.append(el('p', 'diagnostic-detail', detail)));
    appendTextList(card, '다음 조치', issue.remediation, 'diagnostic-remediation');
    if (issue.resolvable_with_existing_inputs === true) card.append(el('p', 'diagnostic-availability', '기존 영상에서 추가 검토할 수 있는 항목입니다.'));
    if (issue.resolvable_with_existing_inputs === false) card.append(el('p', 'diagnostic-availability', '현재 입력 외의 추가 관측·자료가 필요합니다.'));
    grid.append(card);
  });
}
function renderLandingCandidate(camera, view, player) {
  const candidate = view.landing_candidate;
  if (!candidate || candidate.status !== 'approximate' || candidate.inference_only !== true
      || candidate.contact_observed !== false || !validSeconds(candidate.time_seconds)) return null;
  const inferred = candidate.point_kind === 'inferred_object_center' || candidate.kind === 'occluded_descent_and_rebound';
  const panel = el('section', 'landing-candidate'); panel.dataset.candidateCamera = camera;
  const heading = el('div', 'candidate-heading');
  heading.append(el('strong', '', inferred ? '가림 구간의 착지 추정' : '착지 추정 구간'), el('span', 'candidate-badge', '추정 · 접촉 미확인')); panel.append(heading);
  const interval = candidate.time_interval;
  const hasInterval = Array.isArray(interval) && interval.length === 2 && interval.every(validSeconds) && interval[0] <= interval[1];
  const stamp = hasInterval ? `${fmt(interval[0], 2)}–${fmt(interval[1], 2)}초` : `${fmt(candidate.time_seconds, 2)}초 부근`;
  const frames = candidate.frame_interval;
  const hasFrames = Array.isArray(frames) && frames.length === 2 && frames.every(value => Number.isInteger(value) && value >= 0) && frames[0] <= frames[1];
  panel.append(el('p', 'candidate-time', stamp));
  if (hasFrames || Number.isInteger(candidate.frame_index)) panel.append(el('p', 'candidate-frames', hasFrames ? `프레임 ${frames[0]}–${frames[1]}` : `프레임 ${candidate.frame_index} 부근`));
  const point = candidate.point_px, size = candidate.analysis_size;
  if (Array.isArray(point) && point.length === 2 && point.every(value => typeof value === 'number' && Number.isFinite(value))
      && size && validSeconds(size.width) && size.width > 0 && validSeconds(size.height) && size.height > 0) {
    const x = Math.round(Math.max(0, Math.min(1, point[0] / size.width)) * 100), y = Math.round(Math.max(0, Math.min(1, point[1] / size.height)) * 100);
    panel.append(el('p', 'candidate-position', `영상 속 추정 위치 · 왼쪽에서 ${x}% / 위에서 ${y}%`));
  }
  panel.append(el('p', 'candidate-note', inferred
    ? '가림 전후의 이동 경로로 위치를 추정했습니다. 이 구간의 물체 위치와 실제 바닥 접촉은 직접 확인하지 못했습니다.'
    : candidate.kind === 'descent_and_rebound'
    ? '물체가 내려온 뒤 올라가는 움직임을 찾았습니다. 바닥 접촉이나 같은 랠리의 셔틀임을 확정한 것은 아닙니다.'
    : '이동 경로에서 검토할 후보 위치를 찾았습니다. 실제 착지인지 영상에서 확인해 주세요.'));
  if (player) {
    const button = el('button', 'seek-button', '착지 추정 구간부터 보기'); button.type = 'button'; button.dataset.candidateSeekCamera = camera;
    button.addEventListener('click', () => seek(player, hasInterval ? interval[0] : candidate.time_seconds)); panel.append(button);
  }
  return panel;
}
function renderView(camera, view, diagnostics) {
  const card = el('article', 'result-view');
  card.id = 'result-view-' + camera.toLowerCase();
  const header = el('div', 'result-view-header');
  header.append(el('span', 'camera-mark' + (camera === 'B' ? ' camera-b' : ''), camera));
  const title = el('div');
  title.append(el('p', 'eyebrow', camera + ' VIEW'), el('h3', '', view.file || camera + ' 시점'));
  header.append(title, el('span', 'processing-label', modeNames[view.processing_mode] || '결과 확인'));
  card.append(header);
  const verdicts = el('div', 'view-verdicts');
  const analysisLabel = view.processing_mode === 'review_guided_reanalysis' ? '후보 구간 재분석' : '자동 분석';
  for (const [label, value] of [[analysisLabel, view.automatic_decision], ['직접 관측 근거', view.direct_decision], ['시점 보조 판정', view.assisted_decision]]) {
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
  const candidatePanel = renderLandingCandidate(camera, view, player);
  if (candidatePanel) body.append(candidatePanel);
  const trajectoryContext = safeUrl(view.trajectory_context_url);
  if (trajectoryContext) {
    const figure = el('figure', 'trajectory-context');
    const link = el('a'); link.href = trajectoryContext; link.target = '_blank'; link.rel = 'noopener';
    const image = el('img'); image.src = trajectoryContext; image.loading = 'lazy';
    image.alt = camera + ' 시점의 이동 경로와 후보 구간 확대'; link.append(image);
    const inferred = view.landing_candidate?.point_kind === 'inferred_object_center' || view.landing_candidate?.kind === 'occluded_descent_and_rebound';
    figure.append(link, el('figcaption', '', candidatePanel ? inferred
      ? '관측 경로와 가림 구간의 추정 위치 확대 ↗ · 관측과 추정을 구분해 표시했습니다. 추정 위치는 직접 확인한 바닥 접촉점이 아닙니다.'
      : '이동 경로와 추정 위치 확대 ↗ · 표시점은 추적한 물체의 중심이며 실제 바닥 접촉점과 다를 수 있습니다.'
      : '이동 경로와 검토 구간 확대 ↗'));
    body.append(figure);
  }
  const replayUrl = safeUrl(view.trajectory_replay_url);
  if (replayUrl) {
    const section = el('section', 'trajectory-replay');
    const heading = el('h4', '', '추적 경로 영상'); heading.id = 'trajectory-replay-heading-' + camera.toLowerCase();
    section.setAttribute('aria-labelledby', heading.id); section.append(heading);
    const replay = el('video'); replay.controls = true; replay.preload = 'metadata'; replay.playsInline = true;
    replay.src = replayUrl; replay.setAttribute('aria-label', camera + ' 시점 추적 경로 구간 영상'); section.append(replay);
    const origin = view.trajectory_replay_info?.source_time_origin;
    section.append(el('p', 'replay-timeline-note', validSeconds(origin)
      ? `원본 ${fmt(origin, 2)}초부터의 구간입니다. 영상 위 프레임·시각은 원본 기준이며, 재생 막대는 이 구간의 시작을 0초로 표시합니다.`
      : '영상 위 프레임·시각은 원본 기준입니다. 이 영상은 원본에서 잘라낸 구간이므로 재생 막대 시간과 다릅니다.'));
    section.append(el('p', 'replay-candidate-note', '경로는 관측한 지점만 연결하며, 누락된 프레임 사이에는 선을 잇지 않습니다. 착지 표시는 추정이며 바닥 접촉이나 IN/OUT 확정을 뜻하지 않습니다.'));
    body.append(section);
  }
  if (diagnostics && typeof diagnostics === 'object') {
    const observation = diagnostics.observation_status;
    const information = el('div', 'view-diagnosis');
    const observationLabel = typeof diagnostics.observation_label === 'string' ? diagnostics.observation_label : observationNames[observation];
    if (observationLabel || (typeof observation === 'string' && observation)) information.append(el('p', 'observation-status', observationLabel || '관측 상태 추가 확인 필요'));
    appendTextList(information, '빠진 근거', diagnostics.missing_requirements);
    appendTextList(information, '이 시점의 후속 조치', diagnostics.follow_up_actions);
    if (information.childNodes.length) body.append(information);
  }
  if (!candidatePanel && player && typeof diagnostics?.focus_time_seconds === 'number' && Number.isFinite(diagnostics.focus_time_seconds) && diagnostics.focus_time_seconds >= 0) {
    const button = el('button', 'seek-button', `검토 구간 ${fmt(diagnostics.focus_time_seconds, 3)}초로 이동`);
    button.type = 'button'; button.dataset.reviewSeekCamera = camera;
    button.addEventListener('click', () => seek(player, diagnostics.focus_time_seconds)); body.append(button);
  } else if (!candidatePanel && player && typeof view.contact_time_seconds === 'number' && Number.isFinite(view.contact_time_seconds) && view.contact_time_seconds >= 0) {
    const button = el('button', 'seek-button', `${view.qualified === true ? '판정 근거' : '후보'} 구간 ${fmt(view.contact_time_seconds, 3)}초로 이동`);
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
  const reviewSource = {ai_visual_review: 'AI 시각 검수', ai_review_guided_search: 'AI 검수의 후보 구간을 참고한 재분석', automatic: '자동 분석'}[view.review_source] || view.review_source || '자동 분석';
  details.append(el('p', '', `검수 출처: ${reviewSource}`));
  if (view.source_sha256) details.append(el('p', '', '원본 SHA256: ' + view.source_sha256));
  if (Array.isArray(view.reason_codes) && view.reason_codes.length) details.append(el('p', '', view.reason_codes.join(' · ')));
  body.append(details); card.append(body); return card;
}
function renderResult(job) {
  const result = job.result;
  if (!result || !['IN', 'OUT', 'REVIEW'].includes(result.decision)) throw new Error('최종 판정 응답 형식이 올바르지 않습니다.');
  state.currentJob = job;
  $('pair-result').hidden = false;
  $('result-banner').dataset.verdict = result.decision;
  $('final-verdict').textContent = result.decision;
  const views = result.views || {};
  const aiUsed = cameras.some(camera => views[camera]?.processing_mode === 'stored_ai_review');
  const refinedCameras = Array.isArray(result.refinement_applied_camera_ids)
    ? result.refinement_applied_camera_ids.filter(camera => cameras.includes(camera))
    : cameras.filter(camera => views[camera]?.processing_mode === 'review_guided_reanalysis');
  $('result-provenance').textContent = refinedCameras.length
    ? `AI 검수의 후보 구간을 참고한 원본 집중 재분석 · ${refinedCameras.join('/')} 시점`
    : (result.refinement_requested === true || job.request?.refine_review_regions === true)
      ? '두 시점 새 분석 · 적용된 AI 후보 구간 없음'
      : aiUsed ? '기존 AI 검수 근거를 활용한 보조 결과' : '두 시점의 분석·대응 검토 결과';
  const sources = $('source-cameras'); sources.replaceChildren();
  const sourceIds = Array.isArray(result.source_camera_ids) ? result.source_camera_ids.filter(id => cameras.includes(id)) : [];
  const hasDirectEvidence = cameras.some(camera => views[camera]?.qualified === true && ['IN', 'OUT'].includes(views[camera]?.direct_decision));
  sources.append(el('span', '', sourceIds.length ? '최종 판정의 접촉 근거' : hasDirectEvidence ? '시점별 접촉 근거 있음 · 연결 조건 확인 필요' : '확인된 접촉 근거 추가 필요'));
  sourceIds.forEach(camera => { const link = el('a', 'source-link', camera + ' 시점 보기 ↓'); link.href = '#result-view-' + camera.toLowerCase(); sources.append(link); });
  const reasons = $('result-reasons'); reasons.replaceChildren();
  const messages = Array.isArray(result.reasons) ? result.reasons : [];
  (messages.length ? messages : ['시점별 판정 근거를 확인해 주세요.']).forEach(message => reasons.append(el('li', '', String(message))));
  const eventSource = correspondenceNames[result.event_correspondence_source] || '저장된 사건 대응 기록';
  $('correspondence-status').textContent = result.same_event_confirmed ? `같은 사건 연결 확인 · ${eventSource}` : '동일 사건 연결이 확인되지 않았습니다.';
  $('result-timing').textContent = typeof result.elapsed_seconds === 'number' ? `이번 작업 ${fmt(result.elapsed_seconds, 1)}초${aiUsed ? ' · 기존 검수 시간은 포함하지 않습니다.' : ''}` : '';
  $('result-json').href = jobUrl(job.job_id);
  $('result-json').download = 'pair-' + job.job_id + '.json';
  renderDiagnostics(result);
  const grid = $('result-views'); grid.replaceChildren();
  cameras.forEach(camera => grid.append(renderView(camera, views[camera] || {}, result.decision === 'REVIEW' ? result.review_diagnostics?.views?.[camera] : null)));
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
    state.frozenDelta = tickDelta(); state.connected = false;
    setConnection('연결 확인 필요 · ' + error.message + ' 표시된 단계와 시간은 마지막 확인 상태입니다.');
    updateClocks();
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
  clearTimeout(state.pollTimer); state.pollGeneration += 1; state.jobId = id; state.currentJob = null;
  state.dashboardJob = null; state.transfer = null; state.connected = true; state.frozenDelta = 0;
  $('pair-result').hidden = true; $('job-progress').hidden = false;
  $('progress-heading').textContent = '작업 상태를 불러오고 있습니다.';
  $('job-status').textContent = '상태 확인 중'; $('job-status').dataset.state = 'loading';
  $('job-elapsed').textContent = '—'; $('elapsed-label').textContent = '분석 경과';
  $('elapsed-caption').textContent = '저장된 작업 상태를 확인합니다.'; $('queue-time').hidden = true;
  $('overall-progress').textContent = '진행 정보를 불러오는 중';
  $('progress-message').textContent = '이 작업의 마지막 처리 상태와 시간을 확인하고 있습니다.';
  const grid = $('progress-views'); grid.replaceChildren();
  for (const camera of cameras) grid.append(renderAnalysisCard(camera, {status: 'queued'}, {status: 'queued'}));
  $('job-permalink').hidden = true; drawWorkflow('queued'); setConnection();
  announce('작업 상태를 불러오고 있습니다.');
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
  setBusy(true); beginSubmission();
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
  } catch (error) {
    showError(error.message); state.submissionError = error.message;
    if (state.transfer) state.transfer.stopped = performance.now();
    renderTransfer(); setBusy(false); loadHistory();
  }
});
$('retry-poll').addEventListener('click', () => { if (state.jobId) { $('retry-poll').hidden = true; pollJob(state.jobId, state.pollGeneration); } });
$('refine-pair').addEventListener('click', async () => {
  const job = state.currentJob;
  if (state.busy || !job || job.result?.decision !== 'REVIEW' || job.result.refinement_available !== true) return;
  const views = {};
  for (const camera of cameras) {
    const source = job.request?.views?.[camera];
    if (typeof source?.upload_id === 'string') views[camera] = {upload_id: source.upload_id};
    else if (typeof source?.video_id === 'string') views[camera] = {video_id: source.video_id};
    else {
      $('refine-error').textContent = '저장된 작업의 입력 정보를 읽지 못했습니다. A/B 영상을 다시 선택해 주세요.';
      $('refine-error').hidden = false; return;
    }
  }
  setBusy(true); $('refine-error').hidden = true; beginSubmission(job);
  try {
    const created = await api('/api/pair-jobs', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({views, same_event_confirmed: job.request?.same_event_confirmed === true, reuse_review: false, refine_review_regions: true})});
    if (!created.job_id) throw new Error('재분석 작업 ID가 응답에 없습니다. 최근 작업 목록을 확인해 주세요.');
    openJob(created.job_id, true);
    $('job-progress').scrollIntoView({block: 'start', behavior: 'auto'});
    loadHistory();
  } catch (error) {
    state.submissionError = error.message;
    if (state.transfer) state.transfer.stopped = performance.now();
    renderTransfer(); showError(error.message); setBusy(false);
  }
});
$('refresh-history').addEventListener('click', loadHistory);
window.addEventListener('popstate', () => {
  const id = new URLSearchParams(location.search).get('job');
  if (id) openJob(id);
  else { clearTimeout(state.pollTimer); state.pollGeneration += 1; state.jobId = null; state.dashboardJob = null; state.transfer = null; setBusy(false); $('job-progress').hidden = true; $('pair-result').hidden = true; }
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
