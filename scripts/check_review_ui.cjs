// Read-only browser check. Start badminton serve for the desired run first.
const {chromium} = require('playwright');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const root = path.resolve(process.argv[2] || 'runs/integrated-poc');
  const url = process.argv[3] || 'http://127.0.0.1:8765';
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_PATH || '/usr/bin/chromium', args:['--no-sandbox']});
  const page = await browser.newPage({viewport:{width:1440,height:1000}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(url);
    await page.waitForSelector('#workspace:not([hidden])');
    await page.waitForFunction(() => document.querySelector('#player').readyState >= 2);
    const first = Number(await page.locator('#timeline').inputValue());
    await page.locator('#next-frame').click();
    await page.waitForFunction(() => Number(document.querySelector('#timeline').value) === 1);
    const advanced = await page.locator('#frame-readout').innerText();
    await page.locator('#evidence-gallery').scrollIntoViewIfNeeded();
    await page.evaluate(() => document.querySelectorAll('#evidence-gallery img').forEach(image => image.loading = 'eager'));
    await page.waitForFunction(() => [...document.querySelectorAll('#evidence-gallery img')].every(image => image.complete && image.naturalWidth > 0));
    const clip = page.locator('#evidence-gallery video');
    if (await clip.count()) {
      await clip.evaluate(video => video.load());
      await page.waitForFunction(() => document.querySelector('#evidence-gallery video').readyState >= 2);
      await clip.evaluate(async video => { await video.play(); });
      await page.waitForFunction(() => document.querySelector('#evidence-gallery video').currentTime > 0);
      await clip.evaluate(video => video.pause());
    }
    const desktop = await page.evaluate(() => ({
      videoCount: Number(document.querySelector('#video-count').textContent),
      playerSize: [document.querySelector('#player').videoWidth, document.querySelector('#player').videoHeight],
      nativeSequenceVisible: [...document.querySelectorAll('#evidence-gallery figcaption')].some(item => item.textContent.includes('원본 연속 프레임')),
      evidenceImages: [...document.querySelectorAll('#evidence-gallery img')].map(image => ({width:image.naturalWidth,height:image.naturalHeight})),
      annotatedClip: [...document.querySelectorAll('#evidence-gallery video')].map(video => ({width:video.videoWidth,height:video.videoHeight,duration:video.duration,error:video.error?.message || null})),
      horizontalOverflow: document.documentElement.scrollWidth > innerWidth,
    }));
    await page.evaluate(() => window.scrollTo(0,0));
    await page.screenshot({path:path.join(root,'review-ui.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    const mobile = await page.evaluate(() => ({horizontalOverflow:document.documentElement.scrollWidth > innerWidth}));
    await page.screenshot({path:path.join(root,'review-mobile.png'),fullPage:true});
    await page.setViewportSize({width:1440,height:1000});
    const last = page.locator('#video-list .video-item').last();
    const lastId = await last.getAttribute('data-video-id');
    await last.click();
    await page.waitForFunction(id => document.querySelector('#result-link').getAttribute('href').includes(id), lastId);
    await page.waitForFunction(() => document.querySelector('#player').readyState >= 2);
    await page.locator('#timeline').evaluate(input => { input.value = input.max; input.dispatchEvent(new Event('input', {bubbles:true})); });
    await page.waitForFunction(() => !document.querySelector('#player').seeking);
    const finalFrame = await page.evaluate(() => ({
      selected:Number(document.querySelector('#timeline').value),
      lastIndex:Number(document.querySelector('#timeline').max),
      readout:document.querySelector('#frame-readout').textContent,
      videoError:document.querySelector('#player').error?.message || null,
    }));
    const checks = {firstFrame:first,advancedFrame:advanced,desktop,mobile,lastVideo:{id:lastId,...finalFrame},errors};
    fs.writeFileSync(path.join(root,'browser-check.json'),JSON.stringify(checks,null,2)+'\n');
    if (first !== 0 || !desktop.nativeSequenceVisible || !desktop.annotatedClip.length || desktop.horizontalOverflow || mobile.horizontalOverflow || errors.length || finalFrame.selected !== finalFrame.lastIndex || finalFrame.videoError) {
      throw new Error('Review browser assertions failed: '+JSON.stringify(checks));
    }
    console.log(JSON.stringify(checks,null,2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });
