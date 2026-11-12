const { chromium } = require('playwright');
const fs = require('fs');
const crypto = require('crypto');
const assert = require('assert/strict');
const root = '/workspace/remediation/release054-browser';
const origin = 'http://127.0.0.1:8784';
const jobId = process.argv[2];
assert(/^[a-z0-9]+$/.test(jobId));
const out = `${root}/checks/${jobId}`;
fs.mkdirSync(out, {recursive: true});
const provenance = JSON.parse(fs.readFileSync(`${root}/app/${jobId}-provenance.json`, 'utf8'));
const digest = data => crypto.createHash('sha256').update(data).digest('hex');
const fileHash = path => digest(fs.readFileSync(path));
const expectedCancellations = [], failures = [], jsErrors = [], consoleErrors = [], badResponses = [], mutations = [];
function observe(page) {
  page.on('pageerror', error => jsErrors.push(String(error)));
  page.on('console', message => {if (message.type() === 'error') consoleErrors.push({text: message.text(), location: message.location()});});
  page.on('request', request => {if (!['GET', 'HEAD'].includes(request.method())) mutations.push({method: request.method(), url: request.url()});});
  page.on('requestfailed', request => {
    const row = {url: request.url(), error: request.failure()?.errorText};
    (row.error === 'net::ERR_ABORTED' ? expectedCancellations : failures).push(row);
  });
  page.on('response', response => {if (response.status() >= 400) badResponses.push({url: response.url(), status: response.status()});});
}
async function seek(page, selector, target) {
  await page.locator(selector).scrollIntoViewIfNeeded();
  const presented = await page.locator(selector).evaluate((video, time) => new Promise((resolve, reject) => {
    video.pause();
    const timeout = setTimeout(() => reject(new Error(`Selected video frame was not presented near${time}`)), 12000);
    function frame(now, metadata) {
      if (Math.abs(metadata.mediaTime - time) < .008) {
        clearTimeout(timeout);
        resolve({media_time: metadata.mediaTime, presented_frames: metadata.presentedFrames});
      } else video.requestVideoFrameCallback(frame);
    }
    video.requestVideoFrameCallback(frame);
    // Avoid choosing the previous encoded frame at a microsecond-rounded PTS
    // boundary. This presentation epsilon is below a millisecond, not a new
    // candidate time, and is recorded in the verification output.
    video.currentTime = time + .0005;
  }), target);
  await page.waitForFunction(({selector, target}) => {
    const video = document.querySelector(selector);
    return video && !video.seeking && video.readyState >= 2 && Math.abs(video.currentTime - target) < .07;
  }, {selector, target});
  await page.locator(selector).evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const state = await page.locator(selector).evaluate(video => ({current_time: video.currentTime, width: video.videoWidth,
    height: video.videoHeight, ready_state: video.readyState, decoded_frames: video.getVideoPlaybackQuality().totalVideoFrames}));
  return {...state, requested_candidate_time: target, presentation_epsilon_seconds: .0005,
    presented_media_time: presented.media_time, presented_frames: presented.presented_frames};
}
async function play(page, selector) {
  const before = await page.locator(selector).evaluate(video => video.getVideoPlaybackQuality().totalVideoFrames);
  await page.locator(selector).evaluate(async video => {video.muted = true; video.currentTime = 0; await video.play();});
  await page.waitForFunction(({selector, before}) => {
    const video = document.querySelector(selector);
    return video.currentTime >= .12 && video.getVideoPlaybackQuality().totalVideoFrames >= before + 3;
  }, {selector, before});
  return page.locator(selector).evaluate(video => {
    video.pause();
    return {current_time: video.currentTime, width: video.videoWidth, height: video.videoHeight,
      decoded_frames: video.getVideoPlaybackQuality().totalVideoFrames};
  });
}
(async () => {
  const browser = await chromium.launch({executablePath: '/usr/bin/chromium', headless: true,
    args: ['--no-sandbox', '--disable-dev-shm-usage']});
  const context = await browser.newContext({viewport: {width: 1440, height: 1050}, locale: 'ko-KR', timezoneId: 'UTC'});
  await context.tracing.start({screenshots: true, snapshots: true});
  const page = await context.newPage(); observe(page);
  const browserVersion = browser.version();
  await page.goto(`${origin}/pair?job=${jobId}`);
  await page.locator('#final-verdict').waitFor();
  assert.equal(await page.locator('#final-verdict').innerText(), 'REVIEW');
  assert.equal(await page.locator('#reuse-review').isChecked(), false);
  const job = await (await page.request.get(`${origin}/api/pair-jobs/${jobId}`)).json();
  assert(job.verification_scope.includes('Completed CLI-result replay'));
  assert.equal(job.elapsed_seconds, null);
  const checks = [], httpAssets = [], screenshots = [];
  for (const camera of ['A', 'B']) {
    const source = provenance.sources[camera], view = job.views[camera], cardSelector = `#result-view-${camera.toLowerCase()}`;
    const card = page.locator(cardSelector), candidate = view.landing_candidate;
    assert.deepEqual(candidate, source.public_landing_candidate);
    assert.deepEqual(job.result.views[camera].landing_candidate, candidate);
    assert.deepEqual(candidate.point_px, source.selected.candidate_px);
    assert.equal(candidate.frame_index, source.selected.frame_index);
    assert.equal(candidate.time_seconds, source.selected.time);
    assert.equal(candidate.inference_only, true);
    assert.equal(candidate.contact_observed, false);
    assert.equal(view.contact_time_seconds, null);
    assert.equal(view.qualified, false);
    assert.equal(await card.locator('.candidate-badge').innerText(), '추정 · 접촉 미확인');
    assert.deepEqual(await card.locator('.view-verdicts strong').allTextContents(), ['REVIEW', 'REVIEW', 'REVIEW']);
    const frameText = candidate.frame_interval ? `프레임 ${candidate.frame_interval[0]}–${candidate.frame_interval[1]}` : `프레임 ${candidate.frame_index} 부근`;
    assert.equal(await card.locator('.candidate-frames').innerText(), frameText);
    const x = Math.round(candidate.point_px[0] / candidate.analysis_size.width * 100);
    const y = Math.round(candidate.point_px[1] / candidate.analysis_size.height * 100);
    assert.equal(await card.locator('.candidate-position').innerText(), `영상 속 추정 위치 · 왼쪽에서 ${x}% / 위에서 ${y}%`);
    const inferred = candidate.point_kind === 'inferred_object_center';
    assert.equal(await card.locator('.candidate-heading strong').innerText(), inferred ? '가림 구간의 착지 추정' : '착지 추정 구간');
    assert((await card.locator('.candidate-note').innerText()).includes(inferred ? '직접 확인하지 못했습니다' : '확정한 것은 아닙니다'));
    await card.locator('.trajectory-context img').scrollIntoViewIfNeeded();
    await page.waitForFunction(selector => {
      const image = document.querySelector(selector);
      return image && image.complete && image.naturalWidth > 0;
    }, `${cardSelector} .trajectory-context img`);
    for (const url of new Set([view.preview_url, view.trajectory_context_url, view.trajectory_replay_url, ...view.evidence_urls].filter(Boolean))) {
      const response = await page.request.head(origin + url);
      assert(response.ok(), `${url}: ${response.status()}`);
      const relative = decodeURIComponent(url.split(`/views/${camera}/`)[1]);
      assert(source.artifact_sha256[relative], relative);
      httpAssets.push({camera, url, status: response.status(), content_type: response.headers()['content-type'],
        source_sha256: source.artifact_sha256[relative]});
    }
    // Verify delivered selected/context media bytes, not only file existence.
    const keyAssets = [view.preview_url, view.trajectory_context_url, view.trajectory_replay_url, view.evidence_urls[0]];
    for (const url of new Set(keyAssets.filter(Boolean))) {
      const response = await page.request.get(origin + url);
      assert(response.ok());
      const relative = decodeURIComponent(url.split(`/views/${camera}/`)[1]);
      assert.equal(digest(await response.body()), source.artifact_sha256[relative]);
      await response.dispose();
    }
    const sourcePlayer = `${cardSelector} .view-video`, replayPlayer = `${cardSelector} .trajectory-replay video`;
    await page.locator(sourcePlayer).evaluate(video => video.load());
    await page.waitForFunction(selector => document.querySelector(selector)?.readyState >= 1, sourcePlayer);
    const target = candidate.time_interval?.[0] ?? candidate.time_seconds;
    await card.locator('.landing-candidate .seek-button').click();
    await page.waitForFunction(({selector, target}) => Math.abs(document.querySelector(selector).currentTime - target) < .07,
      {selector: sourcePlayer, target});
    const buttonTime = await page.locator(sourcePlayer).evaluate(video => video.currentTime);
    assert.equal(await page.locator(replayPlayer).count(), 1);
    assert.notEqual(await page.locator(sourcePlayer).getAttribute('src'), await page.locator(replayPlayer).getAttribute('src'));
    await page.locator(replayPlayer).evaluate(video => video.load());
    await page.waitForFunction(selector => document.querySelector(selector)?.readyState >= 1, replayPlayer);
    const note = await card.locator('.replay-timeline-note').innerText();
    assert(note.includes('원본 기준') && note.includes('0초로 표시'));
    const sourcePlayback = await play(page, sourcePlayer), replayPlayback = await play(page, replayPlayer);
    const selectedSource = await seek(page, sourcePlayer, candidate.time_seconds);
    const relativeTime = candidate.time_seconds - view.trajectory_replay_info.source_time_origin;
    const selectedReplay = await seek(page, replayPlayer, relativeTime);
    assert(Math.abs(selectedReplay.current_time + view.trajectory_replay_info.source_time_origin - candidate.time_seconds) < .07);
    const replayScreenshot = `${out}/desktop-replay-${camera}.png`;
    await card.locator('.trajectory-replay').screenshot({path: replayScreenshot});
    screenshots.push({path: replayScreenshot, sha256: fileHash(replayScreenshot), viewport: [1440, 1050], camera,
      displayed_replay_time: selectedReplay.current_time, capture: 'selected replay frame presented onscreen'});
    const screenshot = `${out}/desktop-view-${camera}.png`;
    await card.screenshot({path: screenshot});
    screenshots.push({path: screenshot, sha256: fileHash(screenshot), viewport: [1440, 1050], camera,
      displayed_source_time: selectedSource.current_time, displayed_replay_time: selectedReplay.current_time});
    checks.push({camera, source_result_sha256: source.result_sha256, source_sha256: source.source_sha256,
      selected: source.selected, public_candidate: candidate, candidate_text: await card.locator('.landing-candidate').innerText(),
      seek_button_target: target, seek_button_actual: buttonTime, source_playback: sourcePlayback, replay_playback: replayPlayback,
      selected_source: selectedSource, selected_replay: selectedReplay, replay_source_time_origin: view.trajectory_replay_info.source_time_origin,
      replay_timeline_note: note, selected_artifact_metadata_agreement: true});
  }
  await page.reload();
  await page.locator('#final-verdict').waitFor();
  assert.equal(await page.locator('#final-verdict').innerText(), 'REVIEW');
  assert.equal(await page.locator('.analysis-card[data-state="completed"]').count(), 2);
  const mobile = [];
  const mobileContext = await browser.newContext({viewport: {width: 375, height: 844}, isMobile: true, hasTouch: true,
    deviceScaleFactor: 1, locale: 'ko-KR', timezoneId: 'UTC'});
  const mobilePage = await mobileContext.newPage(); observe(mobilePage);
  for (const width of [375, 390]) {
    await mobilePage.setViewportSize({width, height: 844});
    await mobilePage.goto(`${origin}/pair?job=${jobId}`);
    await mobilePage.locator('#final-verdict').waitFor();
    assert.equal(await mobilePage.locator('#final-verdict').innerText(), 'REVIEW');
    assert(await mobilePage.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `overflow at${width}`);
    const mobileVideoChecks = [];
    for (const camera of ['a', 'b']) {
      const candidate = mobilePage.locator(`#result-view-${camera} .landing-candidate`);
      await candidate.scrollIntoViewIfNeeded();
      const bounds = await candidate.boundingBox();
      assert(bounds.x >= 0 && bounds.x + bounds.width <= width + 1);
      assert.equal(await candidate.locator('.candidate-badge').innerText(), '추정 · 접촉 미확인');
      const view = job.views[camera.toUpperCase()];
      const sourceSelector = `#result-view-${camera} .view-video`;
      const replaySelector = `#result-view-${camera} .trajectory-replay video`;
      const sourcePlayback = await play(mobilePage, sourceSelector);
      const replayPlayback = await play(mobilePage, replaySelector);
      const sourceFrame = await seek(mobilePage, sourceSelector, view.landing_candidate.time_seconds);
      const replayFrame = await seek(mobilePage, replaySelector, view.landing_candidate.time_seconds - view.trajectory_replay_info.source_time_origin);
      mobileVideoChecks.push({camera, source_playback: sourcePlayback, replay_playback: replayPlayback,
        selected_source: sourceFrame, selected_replay: replayFrame});
    }
    await mobilePage.locator('#result-view-a .landing-candidate').scrollIntoViewIfNeeded();
    const screenshot = `${out}/mobile-${width}.png`;
    await mobilePage.screenshot({path: screenshot});
    screenshots.push({path: screenshot, sha256: fileHash(screenshot), viewport: [width, 844], is_mobile_emulation: true});
    mobile.push({width, height: 844, horizontal_overflow: false, both_candidates_fit: true, review_wording_preserved: true, video_checks: mobileVideoChecks});
  }
  await mobileContext.close();
  await context.tracing.stop({path: `${out}/trace.zip`});
  assert.equal(mutations.length, 0, JSON.stringify(mutations));
  assert.equal(jsErrors.length, 0, jsErrors.join('\n'));
  assert.equal(consoleErrors.filter(row => !row.location.url?.endsWith('/favicon.ico')).length, 0, JSON.stringify(consoleErrors));
  assert.equal(failures.length, 0, JSON.stringify(failures));
  const relevantHttpErrors = badResponses.filter(row => !row.url.endsWith('/favicon.ico'));
  assert.equal(relevantHttpErrors.length, 0, JSON.stringify(relevantHttpErrors));
  const report = {status: 'passed', job_id: jobId, scope: provenance.scope, core_commit: provenance.core_commit,
    implementation_fingerprint: provenance.implementation_fingerprint, browser: {engine: 'Chromium', version: browserVersion,
      playwright_version: require('playwright/package.json').version, executable: '/usr/bin/chromium', headless: true, host: 'Linux'},
    provenance, checks, http_assets: httpAssets, reload_restoration: true, mobile, screenshots,
    javascript_errors: jsErrors, console_errors: consoleErrors, failed_requests: failures, http_errors: badResponses,
    expected_video_seek_cancellations: expectedCancellations, browser_mutation_requests: mutations,
    new_holdout_inputs_or_references_opened: false,
    limits: ['Completed CLI results replayed; browser upload/inference/live progress and browser-job timing not tested.',
      'Desktop Chromium and mobile viewport/touch emulation, not a physical phone or Windows codec test.',
      'REVIEW output remains estimated image-space evidence; rendering/playback checks do not validate identity, floor contact, target court or accuracy.',
      'Independent ordinary4A/12A display slots do not establish same-rally correspondence; no cross-view mapping is claimed.']};
  fs.writeFileSync(`${out}/browser-validation.json`, JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({job: jobId, status: report.status, checks: checks.map(row => ({camera: row.camera, selected: row.selected,
    source_decoded: row.source_playback.decoded_frames, replay_decoded: row.replay_playback.decoded_frames})), js_errors: jsErrors,
    http_errors: badResponses, console_errors: consoleErrors, mobile}));
  await browser.close();
})().catch(error => {console.error(error); process.exit(1);});
