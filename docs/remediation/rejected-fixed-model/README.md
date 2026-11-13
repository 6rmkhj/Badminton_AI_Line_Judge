# Rejected fixed-model development comparison

The fixed TrackNet adapter and checkpoint were tested on two already released cases. CV-only, model-only and fusion each selected **0/2** first-floor-vicinity matches. Fusion followed part of the real approach in one case but switched to a shoe; in the second case it selected an unrelated distant trajectory. The model added 432.04 seconds of inference work across these two clips. This is not a complete-program latency measurement.

[FINAL_FIXED_FUSION_REVIEW.json](FINAL_FIXED_FUSION_REVIEW.json) binds the unchanged parameters, checkpoint, actual source-cache checks, stage results and source review. These are post-release development experiments, excluded from the frozen 7306 cohort metrics. The checkpoint, adapter and model behavior are not shipped or enabled. No prospective cohort was activated by this comparison.
