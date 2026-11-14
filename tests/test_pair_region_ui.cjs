/* Browser integration with synthetic media and an explicit API fixture.
 * No video analysis is started. Run: node tests/test_pair_region_ui.cjs
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const {execFileSync} = require('node:child_process');
const {chromium} = require('playwright');
const output = process.env.REGION_UI_OUTPUT || fs.mkdtempSync(path.join(os.tmpdir(), 'pair-region-ui-'));
fs.mkdirSync(output, {recursive: true});
const staticRoot = path.resolve(__dirname, '../badminton/static');
for (const [name, size] of [['wide', '320x180'], ['portrait', '180x320']]) {
  execFileSync('ffmpeg', ['-v','error','-f','lavfi','-i',`testsrc2=size=${size}:rate=10`,'-t','1',
    '-c:v','libx264','-threads','1','-pix_fmt','yuv420p','-y',path.join(output, name + '.mp4')]);
}
const submissions = [], jobs = {}, errors = [];
fs.writeFileSync(path.join(output, 'estimate.svg'), '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="320"><rect width="640" height="320" fill="#e5ece0"/><path d="M80 270 L230 80 L550 80 L580 270 Z" fill="none" stroke="#607e42" stroke-width="6"/><rect x="290" y="170" width="45" height="30" fill="#a8c766" stroke="#355e30"/><text x="20" y="40" font-family="sans-serif" font-size="22">SYNTHETIC UI FIXTURE — NO ANALYSIS</text></svg>');
let uploadCount = 0, nextDecision = null, nextEvidenceWarnings = null;
const makeJob = (id, request, verdict = 'REVIEW', donor = 'A', kind = 'verified') => {
  const views = Object.fromEntries(['A', 'B'].map(camera => [camera, {
    status: 'completed', file: camera + '.mp4', preview_url: '/fixture/' + (camera === 'A' ? 'wide' : 'portrait') + '.mp4',
    processing_mode: request.views[camera].review_region ? 'user_region_reanalysis' : 'fresh_analysis',
    review_source: request.views[camera].review_region ? 'operator_region_guided_search' : 'automatic',
    automatic_decision: 'REVIEW', direct_decision: 'REVIEW', assisted_decision: 'REVIEW',
    rules: request.rules,
    rationale: 'Synthetic UI fixture only; no analysis was run.', evidence_urls: [], qualified: false,
    calibration_summary: {line_detection_status: camera === 'A' ? 'detected' : 'not_detected',
      detected_line_count: camera === 'A' ? 8 : 0, court_geometry_status: 'unconfirmed',
      ...(request.views[camera].review_region ? {
        primary_court_status: camera === 'A' ? 'selected' : 'review',
        judgment_scope: camera === 'A' ? 'near_half' : null,
        local_region_supported: camera === 'A',
      } : {}),
      detail: camera === 'A' ? '직선 후보 8개를 찾았습니다. 전체 코트 좌표 보정은 확인되지 않았습니다.' : '직선 후보를 찾지 못했습니다.'},
  }]));
  if (verdict !== 'REVIEW' && kind !== 'video_estimate') Object.assign(views[donor], {
    automatic_decision: verdict, direct_decision: verdict, assisted_decision: verdict, qualified: true,
  });
  if (kind === 'video_estimate') Object.assign(views[donor], {
    estimated_decision: verdict,
    video_estimate: {decision: verdict, eligible: verdict !== 'REVIEW', decision_kind: kind,
      source: 'native_temporal_landing_envelope', rules: request.rules,
      reason_codes: verdict === 'REVIEW' ? ['ESTIMATE_UNCERTAINTY_OVERLAPS_BOUNDARY'] : ['ESTIMATE_ENTIRE_REGION_' + verdict],
      reasons: verdict === 'REVIEW' ? ['접촉 가능 영역이 라인 양쪽에 걸쳐 있어 판정을 보류합니다.'] : ['원본 관측 경로의 접촉 가능 범위 전체를 비교했습니다.'],
      frame_interval: [2, 4], time_interval: [.2, .4], uncertainty_px: 6,
      member_event_ids: ['event_000', 'event_001', 'event_002'],
      native_evidence: {observed_frame_count: 7, contact_observed: false, identity_confirmed: false},
      artifact_url: '/fixture/estimate.svg'},
    assessment: {schema_version: 1, summary: '착지 후보 발견 · 직접 접촉 판정 보류', stages: [
      {label: '최초 바닥 접촉', status: 'blocked', status_label: '확인 필요', detail: '직접 접촉을 확인하지 못했습니다.'},
      {label: '대상 셔틀 식별', status: 'blocked', status_label: '확인 필요', detail: '대상 셔틀을 직접 확인하지 못했습니다.'},
    ]},
  });
  return {job_id: id, status: 'completed', request, views,
    result: {decision: verdict, decision_kind: verdict === 'REVIEW' ? 'unresolved' : kind, views, rules: request.rules,
      decision_scope: verdict === 'REVIEW' ? 'unresolved' : 'single_view',
      source_camera_ids: verdict === 'REVIEW' ? [] : [donor], reasons: verdict === 'REVIEW' ? ['실제 접촉 미확정'] : [donor + ' 시점의 유효한 판정'],
      warning_codes: ['SAME_EVENT_UNCONFIRMED'], warnings: ['두 시점의 동일 사건 연결은 확인되지 않았습니다.'],
      same_event_confirmed: false, refinement_available: true, refinement_applied_camera_ids: []}};
};
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const json = (status, value) => {res.writeHead(status, {'Content-Type': 'application/json'}); res.end(JSON.stringify(value));};
  if (url.pathname === '/api/pair-library') return json(200, {videos: [
    {id: 'v_a', file: 'Wide synthetic.mp4', preview_url: '/fixture/wide.mp4', width: 320, height: 180},
    {id: 'v_b', file: 'Portrait synthetic.mp4', preview_url: '/fixture/portrait.mp4', width: 180, height: 320},
  ]});
  if (url.pathname === '/api/pair-uploads') {
    for await (const chunk of req) {} // Exercise actual browser file transfer, no decoding/inference.
    uploadCount++; return json(201, {upload_id: 'upload-' + uploadCount, file: 'Uploaded synthetic.mp4'});
  }
  if (url.pathname === '/api/pair-jobs' && req.method === 'POST') {
    let body = ''; for await (const chunk of req) body += chunk;
    const request = JSON.parse(body); submissions.push(request);
    const id = 'fixture-' + submissions.length;
    jobs[id] = makeJob(id, request, nextDecision?.label || nextDecision || 'REVIEW', nextDecision?.camera || 'A', nextDecision?.kind || 'verified'); nextDecision = null;
    if (nextEvidenceWarnings) jobs[id].views.A.evidence_warnings = nextEvidenceWarnings;
    nextEvidenceWarnings = null;
    return setTimeout(() => json(202, {job_id: id, status: 'queued'}), 150);
  }
  if (url.pathname === '/api/pair-jobs') return json(200, {jobs: []});
  if (url.pathname.startsWith('/api/pair-jobs/')) return json(200, jobs[url.pathname.split('/').pop()]);
  if (url.pathname === '/favicon.ico') {res.writeHead(204); return res.end();}
  const name = url.pathname === '/pair' ? 'pair.html' : url.pathname.slice(1);
  const file = ['pair.html', 'pair.js', 'pair.css'].includes(name) ? path.join(staticRoot, name)
    : /^fixture\/(wide\.mp4|portrait\.mp4|estimate\.svg)$/.test(name) ? path.join(output, name.split('/')[1]) : null;
  if (!file) {res.writeHead(404); return res.end();}
  const data = fs.readFileSync(file), type = file.endsWith('.mp4') ? 'video/mp4' : file.endsWith('.svg') ? 'image/svg+xml' : file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : 'text/html';
  res.writeHead(200, {'Content-Type': type, 'Content-Length': data.length, 'Accept-Ranges': 'bytes'}); res.end(data);
});
(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = 'http://127.0.0.1:' + server.address().port;
  const browser = await chromium.launch({executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium', headless: true, args: ['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1100}});
    page.on('pageerror', error => errors.push(String(error)));
    const ready = async camera => page.waitForFunction(id => { const p = document.getElementById('preview-' + id); return p.readyState >= 2 && p.videoWidth > 0; }, camera);
    const submit = async () => {
      const response = page.waitForResponse(r => r.url().endsWith('/api/pair-jobs') && r.request().method() === 'POST');
      await page.locator('#submit-pair').click();
      assert.equal(await page.locator('#game-mode').isDisabled(), true);
      await response;
      await page.locator('#final-verdict').waitFor(); await page.waitForFunction(() => !document.getElementById('submit-pair').disabled);
      return submissions.at(-1);
    };
    const field = async (camera, name, value) => {await page.locator(`#region-${name}-${camera}`).fill(String(value)); await page.locator(`#region-${name}-${camera}`).press('Tab');};
    await page.goto(origin + '/pair');
    assert.equal(await page.locator('#game-mode').inputValue(), 'doubles');
    await page.locator('#library-a').selectOption('v_a'); await page.locator('#library-b').selectOption('v_b');
    await ready('a'); await ready('b');
    assert.equal(await page.locator('#region-tools-a').getAttribute('open'), null);
    const automatic = await submit();
    assert.deepEqual(automatic.rules, {game: 'doubles', phase: 'rally'});
    assert.match(await page.locator('#result-rules').innerText(), /복식 랠리.*바깥 사이드라인/);
    assert.deepEqual(automatic.views, {A: {video_id: 'v_a'}, B: {video_id: 'v_b'}});
    assert.match(await page.locator('#result-view-a .court-calibration').innerText(), /직선 후보 8개 탐지 · 전체 코트 보정 미확인/);
    assert.match(await page.locator('#result-view-b .court-calibration').innerText(), /직선 후보 미탐지 · 전체 코트 보정 미확인/);
    await page.locator('#edit-regions').click();
    // The native game select is keyboard-operable and applies to the next run.
    await page.locator('#game-mode').focus(); await page.keyboard.press('Home');
    await page.keyboard.press('ArrowDown'); await page.keyboard.press('Tab');
    assert.equal(await page.locator('#game-mode').inputValue(), 'singles');
    // Portrait overlay must exclude horizontal letterboxing.
    const portrait = await page.locator('#region-layer-b').boundingBox();
    const preview = await page.locator('#preview-b').boundingBox();
    assert(Math.abs(portrait.width / portrait.height - 180 / 320) < .01);
    assert(portrait.x > preview.x + 20);
    await page.locator('#region-draw-a').click();
    const box = await page.locator('#region-layer-a').boundingBox();
    await page.mouse.move(box.x + box.width * .15, box.y + box.height * .2); await page.mouse.down();
    await page.mouse.move(box.x + box.width * .7, box.y + box.height * .8, {steps: 8}); await page.mouse.up();
    assert(Math.abs(Number(await page.locator('#region-x-a').inputValue()) - 15) < .6);
    assert.equal(await page.locator('#region-layer-a').evaluate(el => getComputedStyle(el).pointerEvents), 'none');
    assert.equal(await page.locator('#preview-a').evaluate(p => p.controls), true);
    await page.locator('#preview-a').evaluate(p => {p.currentTime = .4;});
    await page.waitForFunction(() => Math.abs(document.getElementById('preview-a').currentTime - .4) < .05);
    await page.locator('#region-draw-a').click(); await page.keyboard.press('Escape');
    assert.equal(await page.locator('#region-draw-a').getAttribute('aria-pressed'), 'false');
    await field('b', 'width', 55); await field('b', 'x', 30); await field('b', 'height', 45); await field('b', 'y', 20);
    assert.equal(await page.locator('#reuse-review').isDisabled(), true);
    assert.equal(await page.locator('#reuse-review').isChecked(), false);
    await page.locator('#pair-form').screenshot({path: path.join(output, 'desktop-regions.png')});
    const manual = await submit();
    assert.deepEqual(manual.rules, {game: 'singles', phase: 'rally'});
    assert.match(await page.locator('#result-rules').innerText(), /단식 랠리.*안쪽 사이드라인/);
    assert.match(await page.locator('#result-view-a .calibration-rules').innerText(), /단식 랠리.*안쪽 사이드라인/);
    assert(Math.abs(manual.views.A.review_region.width - .55) < .01);
    assert.deepEqual(manual.views.B.review_region, {x: .3, y: .2, width: .55, height: .45});
    assert.equal(manual.reuse_review, false);
    assert.match(await page.locator('#result-view-a .court-calibration').innerText(), /주 코트 선택됨 · 가까운 코트 영역 판정 설정됨/);
    assert.match(await page.locator('#result-view-b .court-calibration').innerText(), /주 코트 선택 확인 필요/);
    assert.match(await page.locator('#result-provenance').innerText(), /사용자 지정 영역.*접촉 미확정/);
    await page.reload(); await page.locator('#final-verdict').waitFor(); await ready('b');
    assert.equal(await page.locator('#game-mode').inputValue(), 'singles');
    assert.equal(await page.locator('#region-x-b').inputValue(), '30');
    await page.locator('#edit-regions').click();
    for (const width of [375, 390, 1440]) {
      await page.setViewportSize({width, height: 1000});
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'horizontal overflow at ' + width);
      assert.equal(await page.locator('#region-width-b').inputValue(), '55');
      // ResizeObserver applies letterbox geometry asynchronously. Wait for that
      // geometry, then sample both rectangles in one frame instead of comparing
      // an old layer width with a newly resized selection from a second call.
      await page.waitForFunction(() => {
        const player = document.getElementById('preview-b'), layer = document.getElementById('region-layer-b');
        const expected = player.videoWidth * Math.min(player.clientWidth / player.videoWidth, player.clientHeight / player.videoHeight);
        return Math.abs(layer.getBoundingClientRect().width - expected) < 1;
      });
      const {before, selection} = await page.evaluate(() => ({
        before: document.getElementById('region-layer-b').getBoundingClientRect().toJSON(),
        selection: document.getElementById('region-box-b').getBoundingClientRect().toJSON(),
      }));
      assert(Math.abs(selection.width / before.width - .55) < .02, 'normalized overlay resize');
      if (width === 375) await page.locator('.source-card').nth(1).screenshot({path: path.join(output, 'mobile-375.png')});
    }
    // Source changes clear region. A staged upload can then be restored and reused.
    await page.locator('[data-view="A"][data-mode="upload"]').click();
    assert.equal(await page.locator('#region-box-a').isHidden(), true);
    await page.locator('#file-a').setInputFiles(path.join(output, 'wide.mp4')); await ready('a');
    await field('a', 'width', 65); await field('a', 'x', 10);
    const uploaded = await submit(); assert.equal(uploadCount, 1); assert.equal(uploaded.views.A.upload_id, 'upload-1');
    await page.reload(); await page.locator('#final-verdict').waitFor(); await ready('a');
    await page.locator('#edit-regions').click(); await field('a', 'height', 70);
    const reusedUpload = await submit(); assert.equal(uploadCount, 1); assert.equal(reusedUpload.views.A.upload_id, 'upload-1');
    assert.equal(reusedUpload.views.A.review_region.height, .7);
    // Existing refinement carries manual provenance rather than silently dropping it.
    const refinedResponse = page.waitForResponse(r => r.url().endsWith('/api/pair-jobs') && r.request().method() === 'POST');
    await page.locator('#refine-pair').click(); await refinedResponse; await page.locator('#final-verdict').waitFor();
    assert.deepEqual(submissions.at(-1).views.A.review_region, reusedUpload.views.A.review_region);
    assert.deepEqual(submissions.at(-1).rules, {game: 'singles', phase: 'rally'});
    await page.locator('#edit-regions').click();
    await page.locator('#region-clear-a').click(); await page.locator('#region-clear-b').click();
    assert.equal(await page.locator('#reuse-review').isDisabled(), false);
    const cleared = await submit(); assert.equal('review_region' in cleared.views.A, false); assert.equal('review_region' in cleared.views.B, false);
    nextDecision = 'IN';
    await submit();
    assert.equal(await page.locator('#final-verdict').innerText(), 'IN');
    assert.match(await page.locator('#result-provenance').innerText(), /A 시점의 유효한 판정/);
    assert.match(await page.locator('#correspondence-status').innerText(), /A 시점에서 판정.*동일 사건 연결은 미확인/);
    assert.match(await page.locator('#result-reasons .muted').innerText(), /동일 사건 연결은 확인되지 않았습니다/);
    assert.deepEqual(await page.locator('#result-view-b .view-verdicts strong').allTextContents(), ['REVIEW', 'REVIEW', 'REVIEW']);
    await page.locator('#pair-result').screenshot({path: path.join(output, 'single-view-in.png')});
    nextDecision = {camera: 'B', label: 'OUT'};
    await page.locator('#game-mode').selectOption('doubles');
    const bOnly = await submit();
    assert.deepEqual(bOnly.rules, {game: 'doubles', phase: 'rally'});
    assert.equal(await page.locator('#final-verdict').innerText(), 'OUT');
    assert.match(await page.locator('#result-provenance').innerText(), /B 시점의 유효한 판정/);
    assert.deepEqual(await page.locator('#result-view-a .view-verdicts strong').allTextContents(), ['REVIEW', 'REVIEW', 'REVIEW']);
    nextDecision = {camera: 'A', label: 'IN', kind: 'video_estimate'};
    await submit();
    assert.equal(await page.locator('#final-verdict').innerText(), 'IN');
    assert.equal(await page.locator('#final-verdict').getAttribute('aria-label'), '영상 추정 IN');
    assert.equal(await page.locator('#result-kind').innerText(), '영상 추정');
    assert.match(await page.locator('#result-provenance').innerText(), /A 시점.*원본 영상.*추정한 결과/);
    assert.match(await page.locator('#source-cameras').innerText(), /영상 추정의 원본 근거/);
    assert.match(await page.locator('#result-view-a .view-estimate-verdict').innerText(), /영상 추정 IN/);
    assert.deepEqual(await page.locator('#result-view-a .view-verdicts strong').allTextContents(), ['REVIEW', 'REVIEW', 'REVIEW']);
    assert.deepEqual(await page.locator('#result-view-b .view-verdicts strong').allTextContents(), ['REVIEW', 'REVIEW', 'REVIEW']);
    assert.equal(await page.locator('#result-view-a .direct-contact-assessment').getAttribute('open'), null);
    await page.locator('#result-view-a .direct-contact-assessment>summary').click();
    assert.equal(await page.locator('#result-view-a .assessment-stages li[data-status="blocked"]').count(), 2);
    assert.match(await page.locator('#result-view-a .video-estimate-panel').innerText(), /0\.200–0\.400초/);
    assert.match(await page.locator('#result-view-a .estimate-metrics').innerText(), /7프레임.*3개를 한 사건으로 묶음/s);
    await page.locator('#result-view-a .estimate-evidence').scrollIntoViewIfNeeded();
    await page.waitForFunction(() => document.querySelector('#result-view-a .estimate-evidence img')?.naturalWidth > 0);
    await page.locator('[data-estimate-seek-camera="A"]').click();
    await page.waitForFunction(() => Math.abs(document.querySelector('#result-view-a .view-video').currentTime - .2) < .05);
    await page.locator('#pair-result').screenshot({path: path.join(output, 'video-estimate-in.png')});
    await page.reload(); await page.locator('#final-verdict').waitFor();
    assert.equal(await page.locator('#result-kind').innerText(), '영상 추정');
    for (const width of [375, 390, 1440]) {
      await page.setViewportSize({width, height: 1000});
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'estimate horizontal overflow at ' + width);
      if (width === 375) await page.locator('#result-view-a .video-estimate-panel').screenshot({path: path.join(output, 'video-estimate-mobile.png')});
    }
    nextDecision = {camera: 'B', label: 'OUT', kind: 'video_estimate'};
    await submit();
    assert.equal(await page.locator('#final-verdict').getAttribute('aria-label'), '영상 추정 OUT');
    assert.match(await page.locator('#result-view-b .estimate-basis').innerText(), /영역 전체가 관측된 경기 경계 밖/);
    assert.deepEqual(await page.locator('#result-view-b .view-verdicts strong').allTextContents(), ['REVIEW', 'REVIEW', 'REVIEW']);
    nextDecision = {camera: 'A', label: 'REVIEW', kind: 'video_estimate'};
    await submit();
    assert.equal(await page.locator('#final-verdict').innerText(), 'REVIEW');
    assert.equal(await page.locator('#result-kind').isHidden(), true);
    assert.match(await page.locator('#result-view-a .video-estimate-panel').innerText(), /라인 양쪽에 걸쳐/);
    assert.match(await page.locator('#result-view-a .estimate-conclusion').innerText(), /판정 보류/);
    assert.equal(await page.locator('#result-view-a .assessment-stages li[data-status="blocked"]').count(), 2);
    nextDecision = 'IN'; await submit();
    assert.equal(await page.locator('#result-kind').isHidden(), true);
    assert.equal(await page.locator('#estimate-result-note').isHidden(), true);
    assert.equal(await page.locator('#result-view-a .video-estimate-panel').count(), 0);
    nextDecision = 'IN';
    nextEvidenceWarnings = [{artifact: 'trajectory_context', error_type: 'OSError',
      message: '한글 근거 이미지 저장 실패: <img src=x onerror="window.fixtureInjected=true">'}];
    await submit();
    assert.equal(await page.locator('#final-verdict').innerText(), 'IN');
    assert.match(await page.locator('#result-view-a .evidence-warnings').innerText(), /계산된 판정 결과는 유지했습니다/);
    assert.match(await page.locator('#result-view-a .evidence-warnings li').innerText(), /한글 근거 이미지 저장 실패.*<img/);
    assert.equal(await page.locator('#result-view-a .evidence-warnings img').count(), 0);
    assert.equal(await page.evaluate(() => window.fixtureInjected === true), false);
    await page.reload(); await page.locator('#final-verdict').waitFor();
    assert.equal(await page.locator('#final-verdict').innerText(), 'IN');
    assert.match(await page.locator('#result-view-a .evidence-warnings').innerText(), /계산된 판정 결과는 유지했습니다/);
    await page.setViewportSize({width: 375, height: 1000});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'warning horizontal overflow');
    await page.locator('#library-b').selectOption('v_a');
    assert.equal(await page.locator('#region-box-b').isHidden(), true);
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, 'browser-validation.json'), JSON.stringify({status: 'passed', fixture: 'Synthetic media and mocked API; no analysis executed',
      checks: ['automatic input sources preserved', 'drag selection', 'keyboard selection', 'native preview seeking', 'portrait letterboxing', 'escape cancellation',
        'per-view normalized payload', 'manual provenance', 'reload restoration', '375/390px no overflow', 'resize preservation', 'upload ID reuse', 'refine preservation', 'clear/reset',
        'singles/doubles keyboard selection', 'busy rule control disabled', 'rule payload/reload/refinement preservation', 'selected game labels', 'A-only IN with B REVIEW', 'B-only OUT with A REVIEW', 'correspondence warning preserves final verdict',
        'explicit video estimate IN/OUT provenance', 'strict REVIEW evidence remains blocked', 'native evidence image and interval seeking', 'estimate reload restoration', 'estimate mobile widths', 'near-line estimate remains REVIEW', 'verified result clears estimate label',
        'optional evidence warning preserves IN and reload', 'evidence warning escapes markup', 'evidence warning mobile width'],
      submissions: submissions.length, uploads: uploadCount, javascript_errors: errors}, null, 2));
    console.log(JSON.stringify({status: 'passed', output, submissions: submissions.length, uploads: uploadCount}));
  } finally {await browser.close(); await new Promise(resolve => server.close(resolve));}
})().catch(error => {console.error(error); server.close(); process.exitCode = 1;});
