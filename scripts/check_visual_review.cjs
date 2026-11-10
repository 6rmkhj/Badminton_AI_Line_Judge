// Read-only browser verification of the portable AI visual review report.
// Usage: node scripts/check_visual_review.cjs [run-directory] [server-URL]
const {chromium} = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');

(async () => {
  const root = path.resolve(process.argv[2] || 'runs/ai-reviewed-poc');
  const url = process.argv[3] || 'http://127.0.0.1:8766';
  const reportBytes = fs.readFileSync(path.join(root, 'visual-review.json'));
  const expected = JSON.parse(reportBytes.toString('utf8'));
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium',
    args: ['--no-sandbox'],
  });
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = [];
  const failedRequests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => { if (response.status() >= 400) failedRequests.push({url: response.url(), status: response.status()}); });
  const checks = {passed: false, expectedVideoCount: expected.video_count,
    reportSha256: crypto.createHash('sha256').update(reportBytes).digest('hex'),
    htmlSha256: crypto.createHash('sha256').update(fs.readFileSync(path.join(root, 'index.html'))).digest('hex'),
    filters: {}, errors, failedRequests};
  try {
    await page.goto(url, {waitUntil: 'domcontentloaded'});
    await page.waitForSelector('article[data-decision]');
    const count = await page.locator('article[data-decision]').count();
    checks.videoCount = count;
    assert.equal(count, expected.video_count, 'Every report record must render one card');
    assert.equal(count, 74, 'The project acceptance report must contain all 74 videos');
    checks.observationCountsVisible = await page.locator('#observation-counts').isVisible();
    assert(checks.observationCountsVisible, 'Observation status summary must be visible');
    for (const label of ['IN', 'OUT', 'REVIEW']) {
      await page.locator('#filter').selectOption(label);
      const state = await page.evaluate(() => {
        const cards = [...document.querySelectorAll('article[data-decision]')].filter(card => !card.hidden);
        return {count: cards.length, labels: [...new Set(cards.map(card => card.dataset.decision))]};
      });
      checks.filters[label] = state;
      assert.equal(state.count, expected.assisted_counts[label], `Wrong ${label} filter count`);
      assert(state.labels.every(value => value === label), 'Filter shows another verdict');
    }
    await page.locator('#filter').selectOption('all');

    async function checkVideo(videoId) {
      const selector = `#${videoId} video`;
      const locator = page.locator(selector);
      assert.equal(await locator.count(), 1, `Missing preview for ${videoId}`);
      await locator.scrollIntoViewIfNeeded();
      await locator.evaluate(video => { video.muted = true; video.load(); });
      await page.waitForFunction(selector => {
        const video = document.querySelector(selector);
        return video && video.readyState >= 2 && video.videoWidth > 0;
      }, selector, {timeout: 30000});
      await locator.evaluate(async video => { video.currentTime = 0; await video.play(); });
      await page.waitForFunction(selector => document.querySelector(selector).currentTime > 0.05, selector);
      await locator.evaluate(video => video.pause());
      const state = await locator.evaluate(video => ({
        width: video.videoWidth, height: video.videoHeight,
        duration: video.duration, playedTo: video.currentTime,
        error: video.error?.message || null,
      }));
      assert.equal(state.error, null);
      assert(state.duration > 0 && state.playedTo > 0);
      return {videoId, ...state};
    }
    checks.firstPreview = await checkVideo(expected.videos[0].video_id);
    checks.lastPreview = await checkVideo(expected.videos.at(-1).video_id);

    const seeks = await page.locator('button[data-seek-video]').count();
    checks.seekControlCount = seeks;
    if (seeks) {
      const control = page.locator('button[data-seek-video]').first();
      const videoId = await control.getAttribute('data-seek-video');
      const time = Number(await control.getAttribute('data-seek-time'));
      await control.click();
      await page.waitForFunction(({videoId, time}) => {
        const video = document.querySelector(`#${videoId} video`);
        return video && !video.seeking && Math.abs(video.currentTime - Math.min(time, video.duration - 0.001)) < 0.06;
      }, {videoId, time});
      checks.contactSeek = {videoId, requested: time, actual: await page.locator(`#${videoId} video`).evaluate(video => video.currentTime)};
    }

    const donorLinks = page.locator('a.donor-link');
    checks.donorLinkCount = await donorLinks.count();
    checks.donorNavigation = [];
    assert.equal(checks.donorLinkCount, expected.cross_view_transferred_count);
    for (let i = 0; i < checks.donorLinkCount; i++) {
      await page.locator('#filter').selectOption('all');
      const link = donorLinks.nth(i);
      const target = (await link.getAttribute('href')).slice(1);
      await link.click();
      const state = await page.locator(`#${target}`).evaluate(article => ({exists: true, hidden: article.hidden, top: article.getBoundingClientRect().top}));
      assert(!state.hidden && Math.abs(state.top) < 5, 'Donor link must reveal and navigate to the source card');
      checks.donorNavigation.push({target, ...state});
    }

    // Decode each image one at a time to avoid retaining hundreds of full-resolution sheets.
    const images = await page.locator('article .evidence img').evaluateAll(elements => elements.map(image => image.src));
    const expectedImages = expected.videos.reduce((sum, row) => sum + row.assets.evidence.length, 0);
    assert.equal(images.length, expectedImages);
    checks.evidenceImageCount = images.length;
    checks.evidenceImagesDecoded = 0;
    for (const src of [...new Set(images)]) {
      const result = await page.evaluate(src => new Promise(resolve => {
        const image = new Image();
        const timer = setTimeout(() => { image.src = ''; resolve({ok: false, error: 'timeout'}); }, 30000);
        image.onload = () => { clearTimeout(timer); const value = {ok: image.naturalWidth > 0, width: image.naturalWidth, height: image.naturalHeight}; image.src = ''; resolve(value); };
        image.onerror = () => { clearTimeout(timer); resolve({ok: false, error: 'decode failed'}); };
        image.src = src;
      }), src);
      assert(result.ok, `Evidence image failed to decode: ${src}`);
      checks.evidenceImagesDecoded++;
    }
    const crossAssets = expected.videos.flatMap(row => row.assets.evidence.filter(asset => asset.kind === 'cross_view_correspondence'));
    checks.crossViewCorrespondenceAssets = crossAssets.length;
    for (const asset of crossAssets) {
      const response = await page.request.get(new URL(asset.path, url.endsWith('/') ? url : url + '/').href);
      assert.equal(response.status(), 200);
    }

    await page.locator('#filter').selectOption('all');
    await page.evaluate(() => scrollTo(0,0));
    checks.desktop = await page.evaluate(() => ({horizontalOverflow: document.documentElement.scrollWidth > innerWidth, width: innerWidth}));
    assert.equal(checks.desktop.horizontalOverflow, false);
    await page.screenshot({path: path.join(root, 'review-desktop.png')});
    await page.setViewportSize({width: 390, height: 844});
    checks.mobile = await page.evaluate(() => ({horizontalOverflow: document.documentElement.scrollWidth > innerWidth, width: innerWidth}));
    assert.equal(checks.mobile.horizontalOverflow, false);
    await page.screenshot({path: path.join(root, 'review-mobile.png')});
    // The last card is also checked at mobile width, since its media may differ.
    await page.locator('article[data-decision]').last().scrollIntoViewIfNeeded();
    checks.mobileLastCardOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
    assert.equal(checks.mobileLastCardOverflow, false);
    assert.equal(errors.length, 0, 'Unexpected browser script error');
    assert.equal(failedRequests.length, 0, 'A referenced report asset returned an HTTP error');
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(root, 'visual-review.json'))).digest('hex'), checks.reportSha256, 'Report changed during browser verification');
    assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(root, 'index.html'))).digest('hex'), checks.htmlSha256, 'HTML changed during browser verification');
    checks.passed = true;
    console.log(JSON.stringify(checks, null, 2));
  } finally {
    fs.writeFileSync(path.join(root, 'browser-check.json'), JSON.stringify(checks, null, 2) + '\n');
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
