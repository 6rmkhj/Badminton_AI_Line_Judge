/* Server hardware, never browser hardware. No analysis starts from this dialog. */
(() => {
  let snapshot = null;
  let loading = false;
  const dialog = document.createElement('dialog');
  dialog.id = 'resource-dialog';
  dialog.className = 'resource-dialog';
  dialog.setAttribute('aria-labelledby', 'resource-title');
  dialog.innerHTML = `
    <div class="resource-heading"><div><p class="resource-eyebrow">BEFORE ANALYSIS</p><h2 id="resource-title">이 환경에서 어떻게 분석할까요?</h2></div><button type="button" id="resource-close" aria-label="장치 설정 닫기">×</button></div>
    <p class="resource-intro">서버가 실행 중인 컴퓨터의 장치를 확인하고, 분석에 사용할 자원을 정합니다.</p>
    <p id="resource-loading" role="status">실행 환경을 확인하고 있습니다…</p>
    <div id="resource-content" hidden>
      <div class="resource-machine"><span class="resource-machine-icon" aria-hidden="true">▣</span><div><strong id="resource-system"></strong><span id="resource-hardware"></span></div><span class="resource-detected">감지 완료</span></div>
      <form id="resource-form">
        <fieldset id="resource-fields">
          <legend>사용 장치</legend>
          <label class="resource-device"><input type="radio" name="resource-device" value="cpu" checked><span><strong>CPU</strong><small id="resource-cpu"></small></span><span class="resource-device-state">선택됨</span></label>
          <div id="resource-accelerators"></div>
          <p id="resource-gpu-scope" class="resource-gpu-note" hidden>GPU에서 후보 마스크 연산과 연결된 모델의 추론을 실행합니다. 영상 처리·코트 보정·궤적 연결은 CPU를 함께 사용합니다. GPU 오류 시 CPU로 자동 전환하지 않습니다.</p>
          <div class="resource-limits">
            <label for="resource-threads"><span>시점당 처리 스레드 <output id="resource-threads-value"></output></span><input id="resource-threads" type="range" min="1" max="1" step="1"><small>OpenCV 연산과 영상 인코딩·디코딩에 적용합니다. 전체 CPU 사용률(%) 제한은 아닙니다.</small></label>
            <label for="resource-parallel"><span>A/B 처리 방식</span><select id="resource-parallel"><option value="1">한 시점씩 · 자원 절약</option><option value="2">두 시점 동시 · 빠른 처리</option></select><small>동시 처리 시 두 시점이 각각 설정한 스레드를 사용합니다.</small></label>
            <label for="resource-memory"><span>프레임 버퍼 예산 <span>MiB</span></span><input id="resource-memory" type="number" min="64" step="1" required><small id="resource-memory-help">분석 프레임 보관 용량입니다. 초과 영상은 분석을 중단하며, 동시 처리 시 나눠 사용합니다. 프로그램 전체 RAM 상한은 아닙니다.</small></label>
            <label for="resource-gpu-memory" id="resource-gpu-memory-field" hidden><span>GPU 추론 메모리 예산 <output id="resource-gpu-memory-value"></output></span><input id="resource-gpu-memory" type="range" min="10" max="90" step="1" value="50" disabled><small>선택한 GPU의 메모리 비율을 PyTorch·ONNX 할당기에 각각 적용합니다. 여러 할당기의 합계나 전체 GPU 메모리·사용률의 상한은 아닙니다. 여유 메모리가 적으면 A/B 순차 처리를 권장합니다.</small></label>
          </div>
        </fieldset>
        <p id="resource-busy" class="resource-notice" hidden>진행 중이거나 대기 중인 분석이 있습니다. 완료 후 설정을 변경할 수 있습니다.</p>
        <p id="resource-summary" class="resource-summary"></p>
        <div class="resource-actions"><button type="button" id="resource-defaults">권장값으로 설정</button><button type="submit" id="resource-save">이 설정으로 준비 완료 <span aria-hidden="true">→</span></button></div>
      </form>
    </div>
    <p id="resource-error" class="resource-error" role="alert" hidden></p>
    <button type="button" id="resource-retry" hidden>환경 다시 확인</button>
    <p class="resource-footnote">서버를 다시 시작하면 환경과 설정을 다시 확인합니다. 설정 완료만으로 분석이 시작되지는 않습니다.</p>`;
  document.body.append(dialog);
  const byId = id => document.getElementById(id);
  const trigger = document.createElement('button');
  trigger.type = 'button'; trigger.id = 'resource-settings-button'; trigger.className = 'resource-settings-button';
  trigger.textContent = '장치 설정 확인';
  const anchor = document.querySelector('.site-header .local-status') || document.querySelector('.workspace-label');
  if (anchor) anchor.replaceWith(trigger); else document.body.prepend(trigger);
  const showError = message => { byId('resource-error').textContent = message; byId('resource-error').hidden = !message; };
  const gib = mb => Number.isFinite(mb) ? (mb / 1024).toLocaleString('ko-KR', {maximumFractionDigits: 1}) + ' GiB' : '확인 불가';
  function selected() {
    const device = dialog.querySelector('[name="resource-device"]:checked')?.value || 'cpu';
    const value = {device, cpu_threads: Number(byId('resource-threads').value), parallel_views: Number(byId('resource-parallel').value), frame_memory_mb: Number(byId('resource-memory').value)};
    if (device.startsWith('cuda:')) value.gpu_memory_percent = Number(byId('resource-gpu-memory').value);
    return value;
  }
  const deviceName = device => device === 'cpu' ? 'CPU' : snapshot?.environment.accelerators.find(item => item.id === device)?.name || device;
  function updateSummary() {
    const value = selected();
    byId('resource-threads-value').textContent = value.cpu_threads + '개';
    const cuda = value.device.startsWith('cuda:');
    byId('resource-gpu-scope').hidden = value.device === 'cpu';
    byId('resource-gpu-memory-field').hidden = !cuda;
    byId('resource-gpu-memory').disabled = !cuda || snapshot?.busy === true;
    byId('resource-gpu-memory-value').textContent = value.gpu_memory_percent + '%';
    for (const input of dialog.querySelectorAll('[name="resource-device"]')) {
      const badge = input.closest('.resource-device')?.querySelector('.resource-device-state');
      if (badge) badge.textContent = input.checked ? '선택됨' : input.disabled ? '사용 불가' : '사용 가능';
    }
    byId('resource-summary').textContent = `${deviceName(value.device)} · 시점당 ${value.cpu_threads}스레드 · ${value.parallel_views === 1 ? 'A/B 순차 처리' : 'A/B 동시 처리'} · 프레임 버퍼 ${gib(value.frame_memory_mb)}${cuda ? ' · GPU 할당 ' + value.gpu_memory_percent + '%' : ''}`;
  }
  function fill(settings) {
    const radio = [...dialog.querySelectorAll('[name="resource-device"]')].find(input => input.value === settings.device);
    if (radio) radio.checked = true;
    byId('resource-threads').value = settings.cpu_threads;
    byId('resource-parallel').value = settings.parallel_views;
    byId('resource-memory').value = settings.frame_memory_mb;
    byId('resource-gpu-memory').value = settings.gpu_memory_percent ?? 50;
    updateSummary();
  }
  function render(data) {
    snapshot = data;
    const env = data.environment;
    byId('resource-system').textContent = [env.system.os, env.system.release, env.system.machine].filter(Boolean).join(' · ');
    byId('resource-hardware').textContent = `사용 가능 CPU ${env.cpu.available_cores}개 · 메모리 ${gib(env.memory.total_mb)}`;
    byId('resource-cpu').textContent = env.cpu.name || '서버 CPU';
    const accelerators = byId('resource-accelerators'); accelerators.replaceChildren();
    for (const item of env.accelerators || []) {
      const card = document.createElement('label'); card.className = 'resource-device' + (item.selectable ? '' : ' resource-unavailable');
      const radio = document.createElement('input'); radio.type = 'radio'; radio.name = 'resource-device'; radio.value = item.id; radio.disabled = !item.selectable;
      const details = document.createElement('span');
      const name = document.createElement('strong'); name.textContent = item.name;
      const reason = document.createElement('small'); reason.textContent = item.reason || (item.selectable ? '추론 실행 확인 완료' : 'GPU 런타임·드라이버를 확인하세요.');
      const capacity = document.createElement('small'); capacity.textContent = `${item.id}${Number.isFinite(item.memory_mb) ? ' · 메모리 ' + gib(item.memory_mb) : ''}`;
      const badge = document.createElement('span'); badge.className = 'resource-device-state';
      details.append(name, reason, capacity); card.append(radio, details, badge); accelerators.append(card);
    }
    if (!accelerators.childElementCount) {
      const note = document.createElement('p'); note.className = 'resource-gpu-note';
      note.textContent = '실행 가능한 GPU를 확인하지 못했습니다. GPU 런타임 설치 후 서버를 다시 시작하거나 CPU를 사용하세요.'; accelerators.append(note);
    }
    byId('resource-threads').max = env.limits.cpu_threads.max;
    byId('resource-memory').min = env.limits.frame_memory_mb.min;
    byId('resource-memory').max = env.limits.frame_memory_mb.max;
    fill(data.settings);
    byId('resource-fields').disabled = data.busy;
    byId('resource-save').disabled = data.busy;
    byId('resource-defaults').disabled = data.busy;
    byId('resource-busy').hidden = !data.busy;
    byId('resource-content').hidden = false;
    trigger.textContent = data.confirmed ? `${data.settings.device === 'cpu' ? 'CPU ' + data.settings.cpu_threads + '스레드' : 'GPU ' + data.settings.device} · 장치 설정` : '장치 설정 확인';
  }
  async function refresh(open = true) {
    if (open && !dialog.open) dialog.showModal();
    if (loading) return;
    loading = true; showError('');
    byId('resource-loading').hidden = false; byId('resource-retry').hidden = true;
    byId('resource-content').hidden = true;
    try {
      const response = await fetch('/api/resources', {cache: 'no-store'});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '실행 환경을 확인하지 못했습니다.');
      render(data);
      if (!data.confirmed && !dialog.open) dialog.showModal();
    } catch (error) {
      snapshot = null; showError(error.message); byId('resource-retry').hidden = false;
      if (!dialog.open) dialog.showModal();
    } finally { loading = false; byId('resource-loading').hidden = true; }
  }
  trigger.addEventListener('click', () => refresh());
  byId('resource-close').addEventListener('click', () => dialog.close());
  byId('resource-retry').addEventListener('click', () => refresh());
  byId('resource-defaults').addEventListener('click', () => fill(snapshot.environment.defaults));
  byId('resource-form').addEventListener('input', updateSummary);
  byId('resource-form').addEventListener('submit', async event => {
    event.preventDefault(); showError(''); byId('resource-save').disabled = true;
    try {
      const response = await fetch('/api/resources', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(selected())});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '설정을 저장하지 못했습니다.');
      render(data); dialog.close(); trigger.focus();
    } catch (error) { showError(error.message); byId('resource-save').disabled = false; }
  });
  function guard(event) {
    if (snapshot?.confirmed) return;
    const target = event.target;
    const analysisAction = event.type === 'submit' ? target.id === 'pair-form' : target.closest?.('#analyze, #refine-pair');
    if (!analysisAction) return;
    event.preventDefault(); event.stopImmediatePropagation(); refresh();
  }
  document.addEventListener('submit', guard, true);
  document.addEventListener('click', guard, true);
  refresh(false);
})();
