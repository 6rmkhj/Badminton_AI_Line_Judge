// Verify REVIEW explanations against real paired API results and a legacy-response rendering case.
// Usage: node scripts/check_review_diagnostics.cjs [server URL] [artifact directory]
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

(async () => {
  const url = process.argv[2] || 'http://127.0.0.1:8770';
  const output = path.resolve(process.argv[3] || 'runs/review-remediation/browser');
  fs.mkdirSync(output, {recursive:true});
  const reportBytes = fs.readFileSync('runs/ai-reviewed-poc/visual-review.json');
  const report = JSON.parse(reportBytes.toString('utf8'));
  const idFor = file => {
    const row = report.videos.find(video => video.file === file);
    assert(row, 'Missing source video: '+file); return row.video_id;
  };
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_PATH || '/usr/bin/chromium', args:['--no-sandbox']});
  const page = await browser.newPage({viewport:{width:1440,height:1050}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const checks = {passed:false, scope:'Live source-bound paired API results. Legacy rendering removes only optional diagnostics; refinement submission is intercepted with HTTP503 to test payload and recovery without claiming inference.', errors, cases:{}, sourceHashes:{}};
  checks.reportSha256 = crypto.createHash('sha256').update(reportBytes).digest('hex');
  checks.reportAssistedCounts = report.assisted_counts;
  for (const name of ['pair.html', 'pair.js', 'pair.css']) checks.sourceHashes[name] = crypto.createHash('sha256').update(fs.readFileSync(path.join('badminton/static',name))).digest('hex');
  async function navigate() {
    await page.goto(url+'/pair', {waitUntil:'domcontentloaded'});
    await page.waitForFunction(() => document.querySelector('#library-a').options.length > 1);
  }
  async function submit(a,b,expected,confirmed=false) {
    await navigate();
    await page.locator('#library-a').selectOption(idFor(a));
    await page.locator('#library-b').selectOption(idFor(b));
    if (confirmed) await page.locator('#same-event').check();
    await page.locator('#submit-pair').click();
    await page.waitForFunction(() => !document.querySelector('#pair-result').hidden || !document.querySelector('#form-error').hidden, null, {timeout:180000});
    assert(await page.locator('#form-error').isHidden(), await page.locator('#form-error').innerText());
    if (expected) assert.equal(await page.locator('#final-verdict').innerText(),expected);
    await page.waitForFunction(() => !document.querySelector('#submit-pair').disabled);
    const id = new URL(page.url()).searchParams.get('job');
    const response = await page.request.get(url+'/api/pair-jobs/'+id);
    assert(response.ok());
    const job = await response.json();
    if (expected) assert.equal(job.result.decision,expected);
    return job;
  }
  try {
    const unresolved = await submit('a시점 3.mp4','b시점 3.mp4','REVIEW');
    const diagnostics = unresolved.result.review_diagnostics;
    assert(diagnostics && Array.isArray(diagnostics.issues) && diagnostics.issues.length, 'The live backend must supply structured REVIEW diagnostics');
    assert.equal(await page.locator('#review-diagnostics').isVisible(),true);
    assert.equal(await page.locator('.diagnostic-issue').count(),diagnostics.issues.length);
    assert.equal(await page.locator('#diagnostics-summary').innerText(),diagnostics.summary);
    for (const camera of ['A','B']) {
      const expected = diagnostics.views?.[camera];
      assert(expected, camera+' diagnostic data missing');
      assert(await page.locator('#result-view-'+camera.toLowerCase()+' .view-diagnosis').isVisible());
      if (Number.isFinite(expected.focus_time_seconds)) {
        const button = page.locator('[data-review-seek-camera="'+camera+'"]');
        assert((await button.innerText()).includes('검토 구간'));
        await button.click();
        await page.waitForFunction(({camera,time}) => {
          const player = document.querySelector('#result-view-'+camera.toLowerCase()+' video');
          return player.readyState >= 2 && !player.seeking && Math.abs(player.currentTime-time) < .08;
        }, {camera,time:expected.focus_time_seconds});
        checks.cases['focus'+camera] = {seconds:expected.focus_time_seconds, labelDoesNotClaimContact:true};
      }
    }
    assert(!/\[object Object\]|undefined|NaN/.test(await page.locator('#review-diagnostics').innerText()));
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth),false);
    await page.locator('#review-diagnostics').screenshot({path:path.join(output,'review-causes-desktop.png')});
    checks.cases.contactLimitations = {job_id:unresolved.job_id, decision:'REVIEW', issue_codes:diagnostics.issues.map(issue=>issue.code), rendered_issues:diagnostics.issues.length};
    await page.setViewportSize({width:390,height:844});
    await page.locator('#review-diagnostics').scrollIntoViewIfNeeded();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth),false);
    await page.locator('#review-diagnostics').screenshot({path:path.join(output,'review-causes-mobile.png')});
    checks.mobileNoHorizontalOverflow = true;
    await page.reload({waitUntil:'domcontentloaded'});
    await page.waitForFunction(() => !document.querySelector('#review-diagnostics').hidden);
    assert.equal(await page.locator('.diagnostic-issue').count(),diagnostics.issues.length);
    checks.reloadPreservesDiagnostics = true;
    await page.setViewportSize({width:1440,height:1050});

    const mismatch = await submit('a시점 16.mp4','b시점 16.mp4','REVIEW',true);
    assert(mismatch.result.reason_codes.includes('KNOWN_DIFFERENT_RALLY'));
    assert(await page.locator('[data-issue-code="KNOWN_DIFFERENT_RALLY"]').isVisible());
    checks.cases.differentRally = {job_id:mismatch.job_id, decision:'REVIEW', same_event_confirmed:mismatch.result.same_event_confirmed};

    let refinementSource;
    for (const [a,b] of [
      ['8k 30 4k 24 a시점 2.mp4','8k 30 4k 24 b시점 2.mp4'],
      ['8k 30 4k 24 a시점 3.mp4','8k 30 4k 24 b시점 3.mp4'],
      ['a-3시점(uhd30 4k30).mp4','b-3시점(uhd30 4k30).mp4'],
    ]) {
      const candidate = await submit(a,b);
      if (candidate.result.decision === 'REVIEW' && candidate.result.refinement_available === true) {
        refinementSource = candidate; break;
      }
    }
    assert(refinementSource, 'Real library must expose a source-bound candidate-region refinement case');
    assert(await page.locator('#refine-pair').isVisible());
    assert((await page.locator('#refine-help').innerText()).includes('기존 AI 검수'));
    for (const camera of ['A','B']) {
      const focus = refinementSource.result.review_diagnostics?.views?.[camera]?.focus_time_seconds;
      if (!Number.isFinite(focus)) continue;
      const button = page.locator('[data-review-seek-camera="'+camera+'"]');
      assert((await button.innerText()).includes('검토 구간'));
      await button.click();
      await page.waitForFunction(({camera,time}) => {
        const player = document.querySelector('#result-view-'+camera.toLowerCase()+' video');
        return player.readyState >= 2 && !player.seeking && Math.abs(player.currentTime-time) < .08;
      }, {camera,time:focus});
      checks.cases.refinementFocus = {camera,seconds:focus,job_id:refinementSource.job_id,labelDoesNotClaimContact:true};
      break;
    }
    await page.locator('#library-a').selectOption(idFor('a시점 5.mp4'));
    await page.locator('#library-b').selectOption(idFor('b시점 5.mp4'));
    await page.locator('#same-event').check();
    let submitted;
    const submissionUrl = url+'/api/pair-jobs';
    await page.route(submissionUrl, async route => {
      if (route.request().method() !== 'POST') { await route.continue(); return; }
      submitted = route.request().postDataJSON();
      await route.fulfill({status:503, contentType:'application/json', body:JSON.stringify({error:'검증용 일시적 처리 실패'})});
    });
    await page.locator('#refine-pair').click();
    await page.waitForFunction(() => !document.querySelector('#refine-error').hidden);
    assert.equal(submitted.reuse_review,false);
    assert.equal(submitted.refine_review_regions,true);
    assert.equal(submitted.same_event_confirmed,refinementSource.request.same_event_confirmed);
    for (const camera of ['A','B']) assert.equal(submitted.views[camera].video_id,refinementSource.request.views[camera].video_id);
    assert.equal(await page.locator('#refine-pair').isDisabled(),false);
    assert.equal(await page.locator('#submit-pair').isDisabled(),false);
    checks.refinementSubmission = {scope:'Payload and HTTP503 recovery only; no inference launched by this case', source_job_id:refinementSource.job_id, usesSavedJobInputs:true, preservesOriginalConfirmation:true, reuse_review:false, refine_review_regions:true};
    await page.unroute(submissionUrl);

    const legacyUrl = url+'/api/pair-jobs/'+unresolved.job_id;
    await page.route(legacyUrl, async route => {
      const response = await route.fetch();
      const job = await response.json();
      delete job.result.review_diagnostics;
      await route.fulfill({response,json:job});
    });
    await page.goto(url+'/pair?job='+unresolved.job_id, {waitUntil:'domcontentloaded'});
    await page.waitForFunction(() => !document.querySelector('#review-diagnostics').hidden);
    assert.equal(await page.locator('#final-verdict').innerText(),'REVIEW');
    assert(await page.locator('.diagnostic-issue').count() > 0);
    assert((await page.locator('#diagnostics-summary').innerText()).includes('같은 사건 확인만으로'));
    checks.legacyResponseFallback = true;
    await page.unroute(legacyUrl);

    const resolved = await submit('a시점 5.mp4','b시점 5.mp4','IN');
    assert.equal(await page.locator('#review-diagnostics').isHidden(),true);
    assert.equal(await page.locator('.view-diagnosis').count(),0);
    checks.cases.resolvedDoesNotShowMissingEvidence = {job_id:resolved.job_id, decision:'IN'};
    for (const [name,a,b] of [
      ['reinspectedPair8','a시점 8.mp4','b시점 8.mp4'],
      ['reinspectedHd3','a시점(uhd30 hd30) 3.mp4','b시점(uhd30 hd30) 3.mp4'],
      ['verified8k6','8k 30 4k 24 a시점 6.mp4','8k 30 4k 24 b시점 6.mp4'],
    ]) {
      const linked = await submit(a,b,'IN');
      assert.equal(await page.locator('#review-diagnostics').isHidden(),true);
      assert(linked.result.source_camera_ids.length > 0);
      checks.cases[name] = {job_id:linked.job_id,decision:'IN',source_camera_ids:linked.result.source_camera_ids,event_correspondence_source:linked.result.event_correspondence_source};
      if (name === 'reinspectedPair8') await page.locator('#result-banner').screenshot({path:path.join(output,'resolved-pair8-banner.png')});
    }
    assert.equal(errors.length,0);
    checks.passed=true;
  } finally {
    fs.writeFileSync(path.join(output,'review-diagnostics-browser-check.json'),JSON.stringify(checks,null,2)+'\n');
    await browser.close();
  }
  console.log(JSON.stringify(checks,null,2));
})().catch(error=>{console.error(error);process.exit(1)});
