/* Run: node tests/test_resource_ui.cjs
 * Exercises the shipped modal + A/B page in Chromium against an explicit API
 * fixture. This verifies UI behavior, not physical CPU/memory enforcement.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const os = require('node:os');
const path = require('node:path');
const {chromium} = require('playwright');

const staticRoot = path.resolve(__dirname, '../badminton/static');
const output = fs.mkdtempSync(path.join(os.tmpdir(), 'resource-ui-'));
const defaults = {device:'cpu', cpu_threads:2, parallel_views:1, frame_memory_mb:4096};
const environment = {
  system:{os:'Windows', release:'11', machine:'AMD64'},
  cpu:{name:'Fixture CPU · 서버 장치', logical_cores:16, available_cores:8},
  memory:{total_mb:16384, available_mb:12288},
  accelerators:[{id:'cuda:0', name:'Fixture GPU', memory_mb:8192, selectable:false,
    reason:'현재 분석 파이프라인은 CPU에서 실행됩니다.'}],
  defaults,
  limits:{cpu_threads:{min:1,max:8}, parallel_views:{min:1,max:2}, frame_memory_mb:{min:64,max:16384}},
  fingerprint:'fixture-server-environment',
};
let confirmed = false, busy = false, failSave = false, failRead = false;
let settings = {...defaults};
const saves = [], submissions = [], errors = [];
const snapshot = () => ({environment,settings,confirmed,busy});
const server = http.createServer(async (request, response) => {
  const pathname = new URL(request.url,'http://localhost').pathname;
  const json = (status, data) => {
    response.writeHead(status, {'Content-Type':'application/json'});
    response.end(JSON.stringify(data));
  };
  if (pathname === '/api/resources') {
    if (request.method === 'GET') return failRead
      ? json(503,{error:'실행 환경 확인 연결 오류'}) : json(200,snapshot());
    let raw = ''; for await (const chunk of request) raw += chunk;
    const proposed = JSON.parse(raw); saves.push(proposed);
    if (failSave) return json(500,{error:'설정을 저장하지 못했습니다. 다시 시도하세요.'});
    if (busy) return json(409,{error:'분석 완료 후 설정을 변경하세요.'});
    settings = proposed; confirmed = true;
    return json(200,snapshot());
  }
  if (pathname === '/api/pair-library') return json(200,{videos:[]});
  if (pathname === '/api/pair-jobs') {
    if (request.method === 'GET') return json(200,{jobs:[]});
    let raw = ''; for await (const chunk of request) raw += chunk;
    submissions.push(JSON.parse(raw));
    return json(409,{error:'이 UI 테스트에서는 분석을 실행하지 않습니다.'});
  }
  if (pathname === '/favicon.ico') {response.writeHead(204);return response.end();}
  const name = pathname === '/pair' ? 'pair.html' : pathname.slice(1);
  if (!['pair.html','pair.js','pair.css','resources.js','resources.css'].includes(name)) {
    response.writeHead(404);return response.end();
  }
  response.writeHead(200, {'Content-Type':name.endsWith('.js') ? 'text/javascript' :
    name.endsWith('.css') ? 'text/css' : 'text/html'});
  response.end(fs.readFileSync(path.join(staticRoot,name)));
});

(async () => {
  await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
  const browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH || '/usr/bin/chromium',
    headless:true,args:['--no-sandbox']});
  try {
    const page = await browser.newPage({viewport:{width:1440,height:1100}});
    page.on('pageerror', error => errors.push(String(error)));
    const dialog = page.locator('#resource-dialog');
    const openDialog = async () => {
      await page.locator('#resource-settings-button').click();
      await page.locator('#resource-content').waitFor();
    };
    const assertNoOverflow = async width => {
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
        'Page horizontal overflow at ' + width);
      assert(await dialog.evaluate(node => node.scrollWidth <= node.clientWidth + 1),
        'Dialog horizontal overflow at ' + width);
    };
    await page.goto('http://127.0.0.1:' + server.address().port + '/pair');
    await page.locator('#resource-content').waitFor();
    assert(await dialog.evaluate(node => node.open), 'First server visit must show resource settings');
    assert.match(await page.locator('#resource-system').innerText(), /Windows · 11 · AMD64/);
    assert.match(await page.locator('#resource-hardware').innerText(), /CPU 8개 · 메모리 16 GiB/);
    assert.match(await page.locator('#resource-cpu').innerText(), /서버 장치/);
    assert.match(await page.locator('#resource-accelerators').innerText(), /Fixture GPU/);
    assert.match(await page.locator('#resource-accelerators').innerText(), /CPU에서 실행/);
    assert.equal(await page.locator('#resource-threads').getAttribute('max'),'8');
    assert.equal(await page.locator('#resource-memory').getAttribute('max'),'16384');
    assert(await page.evaluate(() => document.getElementById('resource-dialog').contains(document.activeElement)),
      'Opening the modal must move keyboard focus inside');
    await assertNoOverflow(1440);
    await dialog.screenshot({path:path.join(output,'resources-desktop.png')});

    // Closing is allowed, but the capture-phase guard still prevents analysis.
    await page.locator('#resource-close').click();
    assert.equal(await dialog.evaluate(node => node.open),false);
    await page.locator('#submit-pair').click();
    await page.locator('#resource-content').waitFor();
    assert(await dialog.evaluate(node => node.open));
    assert.equal(submissions.length,0);
    assert.equal(await page.locator('#form-error').isVisible(),false,
      'Analysis form handler must not run before resource confirmation');

    await page.locator('#resource-threads').fill('3');
    await page.locator('#resource-parallel').selectOption('2');
    await page.locator('#resource-memory').fill('2048');
    assert.match(await page.locator('#resource-summary').innerText(), /3스레드 · A\/B 동시 처리/);
    await page.locator('#resource-save').click();
    await page.waitForFunction(() => !document.getElementById('resource-dialog').open);
    assert.deepEqual(saves.at(-1),{device:'cpu',cpu_threads:3,parallel_views:2,frame_memory_mb:2048});
    assert.equal(submissions.length,0,'Saving settings must not automatically start analysis');
    assert.equal(await page.evaluate(() => document.activeElement.id),'resource-settings-button');
    assert.match(await page.locator('#resource-settings-button').innerText(), /CPU 3스레드/);

    busy = true;
    await openDialog();
    for (const id of ['resource-threads','resource-parallel','resource-memory','resource-save','resource-defaults']) {
      assert(await page.locator('#' + id).isDisabled(),id + ' must be locked while analysis is pending/running');
    }
    assert(await page.locator('#resource-busy').isVisible());
    await page.locator('#resource-close').click();

    busy = false; failSave = true;
    await openDialog();
    await page.locator('#resource-memory').fill('1024');
    await page.locator('#resource-save').click();
    await page.locator('#resource-error').waitFor();
    assert.match(await page.locator('#resource-error').innerText(), /저장하지 못했습니다/);
    assert(await dialog.evaluate(node => node.open),'Save failure must leave correction UI open');
    assert.equal(await page.locator('#resource-save').isDisabled(),false);
    assert.equal(settings.frame_memory_mb,2048,'Failed save must retain effective server settings');
    failSave = false;
    await page.locator('#resource-save').click();
    await page.waitForFunction(() => !document.getElementById('resource-dialog').open);
    assert.equal(settings.frame_memory_mb,1024);

    // Network failure must stay actionable instead of showing stale hardware.
    failRead = true;
    await page.locator('#resource-settings-button').click();
    await page.locator('#resource-retry').waitFor();
    assert.match(await page.locator('#resource-error').innerText(), /연결 오류/);
    assert.equal(await page.locator('#resource-content').isVisible(),false);
    failRead = false;
    await page.locator('#resource-retry').click();
    await page.locator('#resource-content').waitFor();
    assert.equal(await page.locator('#resource-error').isVisible(),false);

    for (const width of [390,375]) {
      await page.setViewportSize({width,height:667});
      await assertNoOverflow(width);
      assert(await dialog.evaluate(node => node.scrollHeight > node.clientHeight),
        'The compact viewport must allow vertical dialog scrolling');
      await page.locator('#resource-close').focus();
      // Walk the actual keyboard order to the final action. Moving focus to the
      // browser chrome at the end of a native dialog's tab cycle is allowed.
      for (let step = 0; step < 10; step++) {
        await page.keyboard.press('Tab');
        assert(await page.evaluate(() => document.getElementById('resource-dialog').contains(document.activeElement)),
          'Page controls behind the modal must not enter its keyboard order');
        if (await page.evaluate(() => document.activeElement.id === 'resource-save')) break;
      }
      assert.equal(await page.evaluate(() => document.activeElement.id),'resource-save');
      assert(await dialog.evaluate(node => node.scrollTop > 0),'Keyboard navigation must scroll to the save action');
      const bounds = await page.locator('#resource-save').boundingBox();
      assert(bounds.y >= 0 && bounds.y + bounds.height <= 667,'Focused action must stay visible');
      await page.keyboard.press('Shift+Tab');
      assert.equal(await page.evaluate(() => document.activeElement.id),'resource-defaults');
      await page.locator('#resource-memory').focus();
      assert(await page.evaluate(() => document.getElementById('resource-dialog').contains(document.activeElement)));
      await page.screenshot({path:path.join(output,'resources-mobile-' + width + '.png')});
    }
    await page.keyboard.press('Escape');
    assert.equal(await dialog.evaluate(node => node.open),false);
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({passed:true,saveAttempts:saves.length,analysisRequests:submissions.length,
      viewports:[1440,390,375],screenshots:output}));
  } finally {
    await browser.close(); server.closeAllConnections(); server.close();
  }
})().catch(error => {
  console.error(error); process.exitCode=1; server.closeAllConnections(); server.close();
});
