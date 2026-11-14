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
let uploadCount = 0;
const makeJob = (id, request) => {
  const views = Object.fromEntries(['A', 'B'].map(camera => [camera, {
    status: 'completed', file: camera + '.mp4', preview_url: '/fixture/' + (camera === 'A' ? 'wide' : 'portrait') + '.mp4',
    processing_mode: request.views[camera].review_region ? 'user_region_reanalysis' : 'fresh_analysis',
    review_source: request.views[camera].review_region ? 'operator_region_guided_search' : 'automatic',
    automatic_decision: 'REVIEW', direct_decision: 'REVIEW', assisted_decision: 'REVIEW',
    rationale: 'Synthetic UI fixture only; no analysis was run.', evidence_urls: [], qualified: false,
    calibration_summary: {line_detection_status: camera === 'A' ? 'detected' : 'not_detected',
      detected_line_count: camera === 'A' ? 8 : 0, court_geometry_status: 'unconfirmed',
      detail: camera === 'A' ? '직선 후보 8개를 찾았습니다. 전체 코트 좌표 보정은 확인되지 않았습니다.' : '직선 후보를 찾지 못했습니다.'},
  }]));
  return {job_id: id, status: 'completed', request, views,
    result: {decision: 'REVIEW', views, source_camera_ids: [], reasons: ['실제 접촉 미확정'],
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
    const id = 'fixture-' + submissions.length; jobs[id] = makeJob(id, request);
    return json(202, {job_id: id, status: 'queued'});
  }
  if (url.pathname === '/api/pair-jobs') return json(200, {jobs: []});
  if (url.pathname.startsWith('/api/pair-jobs/')) return json(200, jobs[url.pathname.split('/').pop()]);
  if (url.pathname === '/favicon.ico') {res.writeHead(204); return res.end();}
  const name = url.pathname === '/pair' ? 'pair.html' : url.pathname.slice(1);
  const file = ['pair.html', 'pair.js', 'pair.css'].includes(name) ? path.join(staticRoot, name)
    : /^fixture\/(wide|portrait)\.mp4$/.test(name) ? path.join(output, name.split('/')[1]) : null;
  if (!file) {res.writeHead(404); return res.end();}
  const data = fs.readFileSync(file), type = file.endsWith('.mp4') ? 'video/mp4' : file.endsWith('.js') ? 'text/javascript' : file.endsWith('.css') ? 'text/css' : 'text/html';
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
      await page.locator('#submit-pair').click(); await response;
      await page.locator('#final-verdict').waitFor(); await page.waitForFunction(() => !document.getElementById('submit-pair').disabled);
      return submissions.at(-1);
    };
    const field = async (camera, name, value) => {await page.locator(`#region-${name}-${camera}`).fill(String(value)); await page.locator(`#region-${name}-${camera}`).press('Tab');};
    await page.goto(origin + '/pair');
    await page.locator('#library-a').selectOption('v_a'); await page.locator('#library-b').selectOption('v_b');
    await ready('a'); await ready('b');
    assert.equal(await page.locator('#region-tools-a').getAttribute('open'), null);
    const automatic = await submit();
    assert.deepEqual(automatic.views, {A: {video_id: 'v_a'}, B: {video_id: 'v_b'}});
    assert.match(await page.locator('#result-view-a .court-calibration').innerText(), /직선 후보 8개 탐지 · 전체 코트 보정 미확인/);
    assert.match(await page.locator('#result-view-b .court-calibration').innerText(), /직선 후보 미탐지 · 전체 코트 보정 미확인/);
    await page.locator('#edit-regions').click();
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
    assert(Math.abs(manual.views.A.review_region.width - .55) < .01);
    assert.deepEqual(manual.views.B.review_region, {x: .3, y: .2, width: .55, height: .45});
    assert.equal(manual.reuse_review, false);
    assert.match(await page.locator('#result-provenance').innerText(), /사용자 지정 영역.*접촉 미확정/);
    await page.reload(); await page.locator('#final-verdict').waitFor(); await ready('b');
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
    await page.locator('#edit-regions').click();
    await page.locator('#region-clear-a').click(); await page.locator('#region-clear-b').click();
    assert.equal(await page.locator('#reuse-review').isDisabled(), false);
    const cleared = await submit(); assert.equal('review_region' in cleared.views.A, false); assert.equal('review_region' in cleared.views.B, false);
    await page.locator('#library-b').selectOption('v_a');
    assert.equal(await page.locator('#region-box-b').isHidden(), true);
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(output, 'browser-validation.json'), JSON.stringify({status: 'passed', fixture: 'Synthetic media and mocked API; no analysis executed',
      checks: ['automatic payload unchanged', 'drag selection', 'keyboard selection', 'native preview seeking', 'portrait letterboxing', 'escape cancellation',
        'per-view normalized payload', 'manual provenance', 'reload restoration', '375/390px no overflow', 'resize preservation', 'upload ID reuse', 'refine preservation', 'clear/reset'],
      submissions: submissions.length, uploads: uploadCount, javascript_errors: errors}, null, 2));
    console.log(JSON.stringify({status: 'passed', output, submissions: submissions.length, uploads: uploadCount}));
  } finally {await browser.close(); await new Promise(resolve => server.close(resolve));}
})().catch(error => {console.error(error); server.close(); process.exitCode = 1;});
