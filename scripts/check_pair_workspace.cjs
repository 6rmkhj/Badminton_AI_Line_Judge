// Exercise the paired workspace against a live local server and real project data.
// Usage: node scripts/check_pair_workspace.cjs [server URL] [artifact directory]
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

(async () => {
  const url = process.argv[2] || 'http://127.0.0.1:8770';
  const output = path.resolve(process.argv[3] || 'runs/pair-validation/browser');
  fs.mkdirSync(output, {recursive:true});
  const report = JSON.parse(fs.readFileSync('runs/ai-reviewed-poc/visual-review.json', 'utf8'));
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_PATH || '/usr/bin/chromium', args:['--no-sandbox']});
  const page = await browser.newPage({viewport:{width:1440,height:1050}});
  const errors = [];
  const requests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (request.method() === 'POST' && request.url().endsWith('/api/pair-jobs')) requests.push(request.postDataJSON()); });
  const checks = {passed:false, errors, cases:{}, screenshotFiles:[], sourceHashes:{}};
  for (const filename of ['pair.html', 'pair.css', 'pair.js']) checks.sourceHashes[filename] = crypto.createHash('sha256').update(fs.readFileSync(path.join('badminton/static', filename))).digest('hex');
  async function navigate() {
    await page.goto(url+'/pair', {waitUntil:'domcontentloaded'});
    await page.waitForFunction(() => document.querySelector('#library-count').textContent.includes('74개'));
  }
  async function select(a, b) {
    await page.locator('#library-a').selectOption(a);
    await page.locator('#library-b').selectOption(b);
  }
  async function submit(expected) {
    const oldCount = requests.length;
    await page.locator('#submit-pair').click();
    await page.waitForFunction(() => !document.querySelector('#pair-result').hidden || !document.querySelector('#form-error').hidden, null, {timeout:90000});
    assert(await page.locator('#form-error').isHidden(), await page.locator('#form-error').innerText());
    assert.equal(await page.locator('#final-verdict').innerText(), expected);
    assert.equal(requests.length, oldCount+1, 'One click must submit only one paired job');
    const id = new URL(page.url()).searchParams.get('job');
    assert(id, 'The current job must have a recoverable permalink');
    const response = await page.request.get(url+'/api/pair-jobs/'+id);
    assert(response.ok());
    const job = await response.json();
    assert.equal(job.result.decision, expected);
    assert.equal(job.result.automatic_benchmark_eligible, false);
    assert.equal(job.result.ground_truth_created, false);
    await page.waitForFunction(() => !document.querySelector('#submit-pair').disabled);
    return job;
  }
  const idFor = file => { const row = report.videos.find(video => video.file === file); assert(row, file+' missing'); return row.video_id; };
  const a5=idFor('a시점 5.mp4'), b5=idFor('b시점 5.mp4');
  try {
    await navigate();
    assert.equal(await page.locator('#same-event').isChecked(), false);
    assert.equal(await page.locator('#reuse-review').isChecked(), true);
    await page.locator('#submit-pair').click();
    assert((await page.locator('#form-error').innerText()).includes('A 시점'));
    assert.equal(requests.length, 0);
    checks.emptyInputRejected = true;

    await select(a5,b5);
    assert(await page.locator('#form-error').isHidden(), 'Input changes must clear a resolved validation error');
    await page.waitForFunction(() => ['a','b'].every(camera => document.querySelector('#preview-'+camera).readyState >= 2));
    await page.screenshot({path:path.join(output,'pair-input-desktop.png')});
    checks.screenshotFiles.push('pair-input-desktop.png');
    const cached=await submit('IN');
    assert.equal(cached.result.views.A.direct_decision,'IN');
    assert.equal(cached.result.views.B.direct_decision,'REVIEW');
    assert(cached.result.source_camera_ids.includes('A'));
    assert.equal(cached.result.event_correspondence_source,'verified_ai_action_correspondence');
    checks.cases.cachedPair = {job_id:cached.job_id, decision:cached.result.decision, source_camera_ids:cached.result.source_camera_ids, event_correspondence_source:cached.result.event_correspondence_source};
    assert.equal(await page.locator('#result-views .result-view').count(),2);
    assert((await page.locator('#correspondence-status').innerText()).includes('동작 대응'));
    await page.locator('[data-seek-camera="A"]').click();
    const seekTime=cached.result.views.A.contact_time_seconds;
    await page.waitForFunction(t => {const video=document.querySelector('#result-view-a video');return video.readyState>=2&&!video.seeking&&Math.abs(video.currentTime-t)<.06;},seekTime);
    checks.contactSeek={camera:'A',expected:seekTime,actual:await page.locator('#result-view-a video').evaluate(video=>video.currentTime)};
    await page.locator('#result-view-b video').evaluate(video=>video.load());
    await page.waitForFunction(()=>document.querySelector('#result-view-b video').readyState>=2);
    for (const camera of ['a','b']) {
      await page.locator('#result-view-'+camera+' video').evaluate(async video=>{video.muted=true;await video.play();});
      await page.waitForFunction(camera=>document.querySelector('#result-view-'+camera+' video').readyState>=3,camera);
      await page.locator('#result-view-'+camera+' video').evaluate(video=>video.pause());
    }
    await page.locator('#result-view-a video').evaluate((video,time)=>video.currentTime=time,seekTime);
    await page.locator('#result-view-b video').evaluate(video=>video.currentTime=2.734167);
    await page.waitForFunction(()=>['a','b'].every(camera=>!document.querySelector('#result-view-'+camera+' video').seeking));
    await page.locator('#result-views img').evaluateAll(images=>images.forEach(image=>image.loading='eager'));
    await page.waitForFunction(()=>[...document.querySelectorAll('#result-views img')].every(image=>image.complete&&image.naturalWidth>0));
    await page.locator('#pair-result').screenshot({path:path.join(output,'pair-result-desktop.png')});
    checks.screenshotFiles.push('pair-result-desktop.png');
    checks.resultImages=await page.locator('#result-views img').count();
    const beforeReload=requests.length;
    await page.reload({waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>!document.querySelector('#pair-result').hidden);
    assert.equal(await page.locator('#final-verdict').innerText(),'IN');
    assert.equal(requests.length,beforeReload,'Reload must resume the saved job without resubmitting');
    checks.reloadRestoresJob=true;
    await page.setViewportSize({width:390,height:844});
    await page.locator('#result-view-a video').evaluate(async video=>{video.muted=true;video.load();await video.play();});
    await page.waitForFunction(()=>document.querySelector('#result-view-a video').readyState>=3);
    await page.locator('#result-view-a video').evaluate((video,time)=>{video.pause();video.currentTime=time;},seekTime);
    await page.waitForFunction(()=>!document.querySelector('#result-view-a video').seeking);
    await page.locator('#pair-result').evaluate(section=>section.scrollIntoView({block:'start'}));
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    await page.screenshot({path:path.join(output,'pair-result-mobile.png')});
    checks.screenshotFiles.push('pair-result-mobile.png');
    checks.mobileNoHorizontalOverflow=true;
    await page.setViewportSize({width:1440,height:1050});

    await navigate(); await select(b5,a5);
    const swapped=await submit('IN');
    assert(swapped.result.source_camera_ids.includes('B'));
    checks.cases.swappedPair={job_id:swapped.job_id,decision:swapped.result.decision,source_camera_ids:swapped.result.source_camera_ids};

    const unknown=report.filename_pair_audit.items.find(pair=>pair.category==='both_review');
    await navigate(); await select(unknown.views[0].video_id,unknown.views[1].video_id);
    const unresolved=await submit('REVIEW');
    checks.cases.bothUnresolved={job_id:unresolved.job_id,decision:unresolved.result.decision};

    await navigate(); await select(idFor('a시점 18.mp4'),idFor('b시점 18.mp4'));
    await page.locator('#same-event').check();
    const mismatch=await submit('REVIEW');
    assert.equal(mismatch.result.same_event_confirmed,false);
    checks.cases.knownMismatch={job_id:mismatch.job_id,decision:mismatch.result.decision,reason_codes:mismatch.result.reason_codes};

    await navigate(); await select(a5,a5); await page.locator('#same-event').check();
    const duplicate=await submit('REVIEW');
    checks.cases.duplicate={job_id:duplicate.job_id,decision:duplicate.result.decision,reason_codes:duplicate.result.reason_codes};

    await navigate();
    await page.locator('[data-view="A"][data-mode="upload"]').click();
    await page.locator('[data-view="B"][data-mode="upload"]').click();
    await page.locator('#file-a').setInputFiles(path.resolve('video/a시점 5.mp4'));
    await page.locator('#file-b').setInputFiles(path.resolve('video/b시점 5.mp4'));
    const uploaded=await submit('IN');
    assert.equal(uploaded.result.views.A.source_sha256,cached.result.views.A.source_sha256);
    assert.equal(uploaded.result.views.B.source_sha256,cached.result.views.B.source_sha256);
    assert.equal(uploaded.result.views.A.processing_mode,'stored_ai_review');
    checks.cases.uploadedExistingPair={job_id:uploaded.job_id,decision:uploaded.result.decision,processing_mode:uploaded.result.views.A.processing_mode,sourceHashesMatch:true};
    assert.equal(await page.locator('#upload-a-progress span').innerText(),'업로드 완료');
    assert.equal(await page.locator('#upload-b-progress span').innerText(),'업로드 완료');

    await page.goto(url+'/pair?job=missing-browser-check-job', {waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>document.querySelector('#progress-heading').textContent==='작업을 열 수 없습니다.');
    assert.equal(await page.locator('#submit-pair').isDisabled(),false);
    checks.missingJobDoesNotLockForm=true;

    await page.goto(url+'/pair?job='+cached.job_id, {waitUntil:'domcontentloaded'});
    await page.waitForFunction(()=>!document.querySelector('#pair-result').hidden);
    checks.finalJobUrl=page.url();
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
    checks.desktopNoHorizontalOverflow=true;
    assert.equal(errors.length,0);
    checks.passed=true;
  } finally {
    fs.writeFileSync(path.join(output,'pair-browser-check.json'),JSON.stringify(checks,null,2)+'\n');
    await browser.close();
  }
  console.log(JSON.stringify(checks,null,2));
})().catch(error=>{console.error(error);process.exit(1)});
