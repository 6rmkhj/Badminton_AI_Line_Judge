/* Real browser input flow with an explicit metadata/API fixture; no AI analysis. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const {chromium} = require('playwright');
const staticRoot = path.resolve(__dirname, '../badminton/static');
const output = fs.mkdtempSync(path.join(os.tmpdir(), 'video-metadata-ui-'));
const errors = [], uploads = [], submissions = [];
const metadata = (name, bytes) => ({width: name.includes('portrait') ? 1080 : 1920,
  height: name.includes('portrait') ? 1920 : 1080, average_fps: 30000 / 1001,
  frame_count: 90, duration: 3.003, codec: 'h264', extension: '.mp4', size_bytes: bytes,
  bitrate_bps: 8000000, variable_frame_rate: name.includes('portrait')});
let failRetry = true;
const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const json = (status, data) => {res.writeHead(status, {'Content-Type': 'application/json'}); res.end(JSON.stringify(data));};
  if (url.pathname === '/api/pair-library') return json(200, {videos: []});
  if (url.pathname === '/api/pair-jobs' && req.method === 'GET') return json(200, {jobs: []});
  if (url.pathname === '/api/pair-uploads') {
    const name = decodeURIComponent(req.headers['x-filename']);
    let bytes = 0; for await (const chunk of req) bytes += chunk.length;
    uploads.push(name);
    if (name === 'retry.mp4' && failRetry) {failRetry = false; return json(400, {error: '영상 정보를 읽지 못했습니다.'});}
    return setTimeout(() => json(201, {upload_id: 'upload-' + name, file: name, metadata: metadata(name, bytes)}), ['slow.mp4', 'pending.mp4'].includes(name) ? 700 : 100);
  }
  if (url.pathname === '/api/pair-jobs' && req.method === 'POST') {
    let raw = ''; for await (const chunk of req) raw += chunk;
    submissions.push(JSON.parse(raw)); return json(202, {job_id: 'test-job', status: 'queued'});
  }
  if (url.pathname === '/api/pair-jobs/test-job') return json(200, {job_id:'test-job', status:'queued', phase:'queued', views:{A:{status:'queued'},B:{status:'queued'}}});
  if (url.pathname === '/resources.js') {res.writeHead(200, {'Content-Type':'text/javascript'}); return res.end('/* Resource dialog is tested separately. */');}
  if (url.pathname === '/favicon.ico') {res.writeHead(204); return res.end();}
  const name = url.pathname === '/pair' ? 'pair.html' : url.pathname.slice(1);
  if (!['pair.html', 'pair.js', 'pair.css', 'resources.css'].includes(name)) {res.writeHead(404);return res.end();}
  const file = path.join(staticRoot, name);
  if (!fs.existsSync(file)) {res.writeHead(404); return res.end();}
  res.writeHead(200, {'Content-Type': name.endsWith('.js') ? 'text/javascript' : name.endsWith('.css') ? 'text/css' : 'text/html'});
  res.end(fs.readFileSync(file));
});
(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = await chromium.launch({executablePath: '/usr/bin/chromium', headless:true,args:['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1100}});
    page.on('pageerror', error => errors.push(String(error)));
    const select = (camera, name) => page.locator('#file-' + camera).setInputFiles({name, mimeType:'video/mp4', buffer:Buffer.alloc(1000)});
    await page.goto('http://127.0.0.1:' + server.address().port + '/pair');
    await page.locator('#upload-a-panel').waitFor();
    await select('a', 'source.mp4');
    await page.waitForFunction(() => document.getElementById('source-a-state').textContent === '원본 확인 완료');
    const facts = await page.locator('#video-info-a-fields').innerText();
    for (const text of ['1920 × 1080 · FHD', '29.970 · 평균', '.MP4', 'H264', '3.00초', '8.00 Mbps']) assert(facts.includes(text), text);
    assert.equal(submissions.length, 0, 'Metadata probing must not submit AI analysis');
    await select('a', 'slow.mp4');
    await page.waitForFunction(() => document.getElementById('source-a-state').textContent === '영상 정보 확인 중');
    await select('a', 'portrait.mp4');
    await page.waitForFunction(() => document.getElementById('video-info-a-fields').textContent.includes('1080 × 1920'));
    await page.waitForTimeout(800);
    assert((await page.locator('#video-info-a-fields').innerText()).includes('1080 × 1920'));
    assert.match(await page.locator('#video-info-a-status').innerText(), /프레임 간격에 변동/);
    await select('b', 'retry.mp4');
    await page.locator('#video-info-b-retry').waitFor();
    assert.match(await page.locator('#video-info-b-status').innerText(), /읽지 못했습니다/);
    await page.locator('#video-info-b-retry').click();
    await page.waitForFunction(() => document.getElementById('source-b-state').textContent === '원본 확인 완료');
    assert.equal(uploads.filter(name => name === 'retry.mp4').length, 2);
    for (const width of [1440,390,375]) {
      await page.setViewportSize({width,height:1100});
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'overflow at ' + width);
      await page.locator('.source-card').first().screenshot({path:path.join(output, 'metadata-' + width + '.png')});
    }
    // Submit while B's initial metadata request is still pending: it must await
    // that request instead of staging the same source a second time.
    await select('b', 'pending.mp4');
    await page.waitForFunction(() => document.getElementById('source-b-state').textContent === '영상 정보 확인 중');
    const expectedUploads = uploads.filter(name => name !== 'pending.mp4').length + 1;
    await page.locator('#submit-pair').click();
    await page.waitForResponse(response => response.url().endsWith('/api/pair-jobs') && response.request().method() === 'POST');
    assert.equal(submissions.length,1);
    assert.equal(uploads.length,expectedUploads,'Confirmed and pending sources must not upload again');
    assert.equal(submissions[0].views.A.upload_id,'upload-portrait.mp4');
    assert.equal(submissions[0].views.B.upload_id,'upload-pending.mp4');
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({passed:true,uploads:uploads.length,screenshots:output}));
  } finally {await browser.close();server.closeAllConnections();server.close();}
})().catch(error => {console.error(error);process.exitCode=1;server.closeAllConnections();server.close();});
