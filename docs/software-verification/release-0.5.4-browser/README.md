# 0.5.4 completed-result browser replay

Actual completed CLI outputs from core `7306efe267e40f4f8539c0439711ec54c2afcd41` were replayed in the frozen UI. Each result, source video, displayed artifact and UI file is hash-bound in the JSON reports. No production source was edited.

| Case / display slot | Selected frame | Source timeline | Image point (960×540) | UI state |
|---|---:|---:|---|---|
| release054controls / A | 49 | 1.633267s | [360.869, 447.182] | REVIEW · estimated · contact unverified |
| release054controls / B | 42 | 1.792200s | [396.667, 200.333] | REVIEW · estimated · contact unverified |
| release054gains / A | 188 | 3.171144s | [602.7496043657602, 240.64656956344658] | REVIEW · estimated · contact unverified |
| release054gains / B | 151 | 2.547044s | [481.934, 210.755] | REVIEW · estimated · contact unverified |

The checks compare serialized candidate/frame/point, displayed labels, selected evidence metadata and delivered artifact bytes. Source and trajectory videos decode, play and seek to matching source times. Reload restores completed results. Desktop 1440×1050 and mobile 375/390×844 viewport/touch emulation show both candidates without horizontal overflow. Screenshots are included; full traces with video bodies remain in the workspace with their hashes recorded.

The two ordinary development clips occupy independent display slots; their fixture explicitly treats them as different events. The original-control pair also receives no unverified cross-view verdict transfer. All outputs remain REVIEW.

No JavaScript exceptions or failed application/media requests were observed. Chromium emitted one non-JavaScript resource warning per run for the missing `/favicon.ico`; it is retained in the logs.

This verifies completed-result display and media consistency. It does not test browser upload/analysis, live processing progress, pair-job or stage durations, physical mobile devices, Windows codecs, landing accuracy or validated contact. No new holdout input, reference or result was opened for this UI review.

[Machine-readable summary](browser-verification-summary.json)

Selected-frame screenshots wait for Chromium’s presented video frame callback, rather than only `currentTime`. The capture harness uses a recorded0.0005s PTS-boundary epsilon; the actually presented frame timestamps match each selected source timestamp. Separate onscreen replay captures are included. This corrected an offscreen headless capture artifact without changing application code.
