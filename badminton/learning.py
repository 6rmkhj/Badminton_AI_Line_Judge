"""Optional supervised patch scorer; no bundled or automatically invented weights.

This scorer refines already-generated candidates. It is not a complete shuttle
detector, contact classifier, or evidence of line-judging accuracy. Training uses
explicitly reviewed shuttle/background labels, with group-disjoint holdouts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


FORMAT_VERSION = 1
ARCHITECTURE = "candidate_patch_cnn_v1"
PREPROCESSING = "opencv_source_bgr_area_v1"
LABELS = {"background": 0, "shuttle": 1}


def _load_manifest(manifest):
    return json.loads(Path(manifest).read_text()) if isinstance(manifest, (str, Path)) else manifest


def _collect_labels(manifest, annotations_dir, label_source='human_review'):
    if label_source not in {'human_review', 'ai_visual_review'}:
        raise ValueError('label_source must be human_review or ai_visual_review')
    records, excluded = [], Counter()
    videos = manifest.get("videos", [])
    errors, seen_ids, content_points = [], set(), {}
    for video in videos:
        video_id = video.get("id")
        if not video_id or video_id in seen_ids:
            errors.append("Manifest video IDs must be present and unique")
            continue
        seen_ids.add(video_id)
        path = Path(annotations_dir) / f"{video_id}.json"
        if not path.exists():
            excluded["missing_annotation"] += 1
            continue
        try:
            annotation = json.loads(path.read_text())
        except (ValueError, OSError):
            excluded["invalid_annotation"] += 1
            continue
        reviewed = (annotation.get("source") == label_source
                    and bool(str(annotation.get("reviewer", "")).strip()))
        if label_source == 'ai_visual_review':
            reviewed = (reviewed and annotation.get('status') == 'reviewed_by_ai'
                        and annotation.get('human_verified') is False
                        and annotation.get('detection_labels_verified_by_ai') is True)
        else:
            reviewed = reviewed and annotation.get("detection_labels_verified") is True
        if not reviewed or annotation.get("video_id", video_id) != video_id:
            excluded["unverified_annotation"] += 1
            continue
        if annotation.get("source_sha256") != video.get("sha256"):
            excluded["annotation_source_hash_missing_or_mismatched"] += 1
            continue
        labels = annotation.get("detection_labels", [])
        if not isinstance(labels, list):
            excluded["invalid_detection_labels"] += 1
            continue
        if not labels:
            excluded["no_explicit_detection_labels"] += 1
        size = annotation.get("analysis_size", {})
        if not isinstance(size, dict):
            size = {}
        width, height = size.get("width"), size.get("height")
        if labels and (not isinstance(width, int) or not isinstance(height, int)
                       or width < 2 or height < 2 or width > 16384 or height > 16384):
            excluded["missing_analysis_size"] += len(labels)
            continue
        seen_points = {}
        for label in labels:
            if not isinstance(label, dict) or label.get("label") not in LABELS:
                excluded["invalid_label"] += 1
                continue
            frame_index, x, y = label.get("frame_index"), label.get("x"), label.get("y")
            if (not isinstance(frame_index, int) or isinstance(frame_index, bool) or frame_index < 0
                    or not isinstance(x, (int, float)) or not isinstance(y, (int, float))
                    or not np.isfinite([x, y]).all() or not 0 <= x < width or not 0 <= y < height):
                excluded["invalid_label_location"] += 1
                continue
            identity = (frame_index, round(x, 3), round(y, 3))
            if identity in seen_points:
                if seen_points[identity] != label["label"]:
                    errors.append(f"Contradictory labels at the same point in {video_id}")
                excluded["duplicate_patch_label"] += 1
                continue
            seen_points[identity] = label["label"]
            content_identity = (video['sha256'], *identity)
            if content_identity in content_points:
                if content_points[content_identity] != label['label']:
                    errors.append('Contradictory labels for duplicate video content')
                excluded['duplicate_content_patch_label'] += 1
                continue
            content_points[content_identity] = label['label']
            records.append({"video_id": video_id, "file": video.get("file"),
                            "path": video.get("path"), "sha256": video.get("sha256"),
                            "group_id": video.get("group_id"), "split": video.get("split"),
                            "frame_index": frame_index, "x": float(x), "y": float(y),
                            "analysis_width": width, "analysis_height": height,
                            "label": LABELS[label["label"]], "label_name": label["label"],
                            "reviewer": annotation["reviewer"], "label_source": label_source})
    return records, dict(excluded), errors


def training_readiness(manifest, annotations_dir, min_train_per_class=4, min_eval_per_class=2,
                       label_source='human_review'):
    """Report missing data honestly; draft suggestions never become labels."""
    manifest = _load_manifest(manifest)
    records, excluded, errors = _collect_labels(manifest, annotations_dir, label_source)
    if min_train_per_class < 1 or min_eval_per_class < 1:
        raise ValueError("Minimum class counts must be positive")
    counts = {split: {name: 0 for name in LABELS} for split in ("train", "val", "test")}
    group_splits, content_splits, pair_splits = defaultdict(set), defaultdict(set), defaultdict(set)
    # Validate the whole manifest, including clips without labels, so an unlabeled
    # duplicate cannot silently cross an evaluation boundary later.
    for video in manifest.get("videos", []):
        split, group, digest = video.get("split"), video.get("group_id"), video.get("sha256")
        if split not in counts or not group or not digest:
            errors.append("Every manifest video requires a train/val/test split, group_id and sha256")
            continue
        group_splits[group].add(split)
        content_splits[digest].add(split)
        if video.get("pair_key"):
            pair_splits[video["pair_key"]].add(split)
    if any(len(splits) > 1 for splits in group_splits.values()):
        errors.append("Same-rally group leakage across dataset splits")
    if any(len(splits) > 1 for splits in content_splits.values()):
        errors.append("Duplicate video content leakage across dataset splits")
    if any(len(splits) > 1 for splits in pair_splits.values()):
        errors.append("Camera-pair rally leakage across dataset splits")
    for record in records:
        if record["split"] in counts:
            counts[record["split"]][record["label_name"]] += 1
    for split, classes in counts.items():
        minimum = min_train_per_class if split == "train" else min_eval_per_class
        for name, count in classes.items():
            if count < minimum:
                errors.append(f"{split} needs at least {minimum} verified {name} patches; found {count}")
    return {"ready": not errors, "dataset_kind": manifest.get("dataset_kind", "real_video"),
            "label_source": label_source, "independent_human_ground_truth": label_source == 'human_review',
            "model_available": False, "trained": False, "verified_patch_count": len(records),
            "counts": counts, "excluded": excluded, "reasons": sorted(set(errors)),
            "requirements": {"train_min_per_class": min_train_per_class,
                             "heldout_min_per_class": min_eval_per_class},
            "scope": "Candidate patch classification only; not shuttle recall or IN/OUT accuracy",
            "notes": ["Explicit reviewed shuttle/background labels of the selected source are required.",
                      "AI-reviewed labels are provisional supervision; agreement with them is not human ground-truth accuracy.",
                      "Verdicts, automated candidate suggestions and unreviewed points are not training labels.",
                      "Validation selects the checkpoint; test data is evaluated once after selection.",
                      "These minimum counts are execution prerequisites, not statistical adequacy."]}


def _patch(frame, x, y, size):
    # Constant padding avoids an off-image reflection introducing artificial
    # mirrored shuttles; labels outside the image are rejected before training.
    half = size // 2
    padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    cx, cy = int(round(float(x))), int(round(float(y)))
    crop = padded[max(0, cy):max(0, cy) + size, max(0, cx):max(0, cx) + size]
    if crop.shape[:2] != (size, size):
        crop = cv2.resize(crop, (size, size))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def prepare_training_samples(manifest, annotations_dir, video_dir=None, patch_size=32,
                             min_train_per_class=4, min_eval_per_class=2, label_source='human_review'):
    """Extract reviewed patches by sequential frame index after MP4 edit lists.

    Returned ``patches`` is an N,H,W,3 uint8 array; ``report`` and ``samples`` are
    JSON serializable. Coordinate dimensions must accompany reviewed labels.
    """
    if patch_size < 16 or patch_size > 128 or patch_size % 2:
        raise ValueError("patch_size must be an even number from 16 to 128")
    manifest = _load_manifest(manifest)
    report = training_readiness(manifest, annotations_dir, min_train_per_class, min_eval_per_class, label_source)
    if not report["ready"]:
        raise ValueError("Training data not ready: " + "; ".join(report["reasons"]))
    records, _, _ = _collect_labels(manifest, annotations_dir, label_source)
    by_video = defaultdict(list)
    for index, record in enumerate(records):
        by_video[record["video_id"]].append((index, record))
    patches = [None] * len(records)
    for video_id, indexed in by_video.items():
        record = indexed[0][1]
        path = Path(record["path"]) if record.get("path") else Path(video_dir or ".") / record["file"]
        if not path.is_file() and video_dir is not None:
            local_root = Path(video_dir).resolve()
            local = (local_root / record['file']).resolve()
            if not local.is_relative_to(local_root):
                raise ValueError('Manifest relative video path escapes video_dir')
            path = local
        if not path.is_file():
            raise ValueError(f"Missing video for {video_id}: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Video content changed since manifest: {video_id}")
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        requested = defaultdict(list)
        for index, sample in indexed:
            requested[sample["frame_index"]].append((index, sample))
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot decode video: {video_id}")
            for frame_index in range(max(requested) + 1):
                # Decode sequentially to preserve edit-list/frame identity, but
                # convert native 4K/8K pixels to BGR only for reviewed samples.
                if not capture.grab():
                    raise ValueError(f"Reviewed frame {frame_index} unavailable in {video_id}")
                if frame_index not in requested:
                    continue
                ok, frame = capture.retrieve()
                if not ok:
                    raise ValueError(f"Reviewed frame {frame_index} unavailable in {video_id}")
                cached_resizes = {}
                for index, sample in requested[frame_index]:
                    dimensions = (sample["analysis_width"], sample["analysis_height"])
                    if dimensions not in cached_resizes:
                        cached_resizes[dimensions] = cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)
                    patches[index] = _patch(cached_resizes[dimensions], sample["x"], sample["y"], patch_size)
        finally:
            capture.release()
    return {"report": report, "samples": records, "patches": np.stack(patches), "patch_size": patch_size}


def _network():
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 24, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((2, 2)),
                         nn.Flatten(), nn.Linear(96, 1))


def _metrics(labels, probabilities):
    labels, probabilities = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    predicted = probabilities >= 0.5
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    return {"samples": int(len(labels)), "threshold": 0.5,
            "accuracy": float((tp + tn) / len(labels)) if len(labels) else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "brier_score": float(np.mean((probabilities - labels) ** 2)) if len(labels) else None,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def train_candidate_scorer(manifest, annotations_dir, video_dir, output_path, epochs=20,
                           learning_rate=0.002, patch_size=32, min_train_per_class=4,
                           min_eval_per_class=2, seed=42, label_source='human_review'):
    """Train on verified patches; never manufacture weights when labels are absent."""
    import torch
    if not isinstance(epochs, int) or not 1 <= epochs <= 1000:
        raise ValueError("epochs must be an integer from 1 to 1000")
    if not 0 < learning_rate <= 0.1:
        raise ValueError("learning_rate must be positive and no greater than 0.1")
    prepared = prepare_training_samples(manifest, annotations_dir, video_dir, patch_size,
                                         min_train_per_class, min_eval_per_class, label_source)
    torch.manual_seed(seed)
    model = _network()
    inputs = torch.from_numpy(prepared["patches"].transpose(0, 3, 1, 2).copy()).float() / 255
    targets = torch.tensor([r["label"] for r in prepared["samples"]], dtype=torch.float32)
    indices = {split: torch.tensor([i for i, r in enumerate(prepared["samples"]) if r["split"] == split])
               for split in ("train", "val", "test")}
    train_labels = targets[indices["train"]]
    ratio = float((train_labels == 0).sum() / (train_labels == 1).sum())
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio))
    validation_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state, best_loss, best_epoch, history = None, float("inf"), None, []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        order = indices["train"][torch.randperm(len(indices["train"]), generator=generator)]
        losses = []
        for batch in order.split(32):
            optimizer.zero_grad()
            loss = criterion(model(inputs[batch]).squeeze(1), targets[batch])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(validation_criterion(model(inputs[indices["val"]]).squeeze(1), targets[indices["val"]]))
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch + 1
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss})
    model.load_state_dict(best_state)
    model.eval()
    # Test data is read for metrics only after checkpoint selection is complete.
    with torch.no_grad():
        metrics = {split: _metrics(targets[index].numpy(), torch.sigmoid(model(inputs[index]).squeeze(1)).numpy())
                   for split, index in indices.items()}
    metadata = {"architecture": ARCHITECTURE, "patch_size": patch_size,
                "preprocessing": PREPROCESSING,
                "analysis_sizes": sorted({(r["analysis_width"], r["analysis_height"]) for r in prepared["samples"]}),
                "dataset_kind": prepared["report"]["dataset_kind"],
                "label_source": 'explicit_human_review' if label_source == 'human_review' else 'ai_visual_review',
                "independent_human_ground_truth": label_source == 'human_review',
                "metric_interpretation": 'heldout human-label patch agreement' if label_source == 'human_review' else 'heldout AI-label patch agreement, not verified accuracy',
                "dataset_fingerprint": _load_manifest(manifest).get('dataset_fingerprint'),
                "label_manifest_sha256": hashlib.sha256(json.dumps(
                    [{key: r[key] for key in ('sha256', 'group_id', 'split', 'frame_index', 'x', 'y', 'label', 'label_source')}
                     for r in prepared['samples']], sort_keys=True).encode()).hexdigest(),
                "scope": "Candidate patch scorer; not full-frame detection or line-judging accuracy",
                "probabilities_calibrated": False, "seed": seed, "epochs": epochs,
                "selected_epoch": best_epoch, "selection_metric": "validation_binary_cross_entropy",
                "counts": prepared["report"]["counts"], "metrics": metrics,
                "split_groups": {split: sorted({r["group_id"] for r in prepared["samples"] if r["split"] == split})
                                 for split in indices},
                "split_video_sha256": {split: sorted({r["sha256"] for r in prepared["samples"] if r["split"] == split})
                                       for split in indices}}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": FORMAT_VERSION, "metadata": metadata, "state_dict": best_state}, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    report = {"trained": True, "model_path": str(destination.resolve()), "weights_sha256": digest,
              **metadata, "history": history,
              "warning": "Heldout patch metrics do not establish full-video or IN/OUT performance; independent evaluation is still required."}
    destination.with_suffix(destination.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


class CandidateScorer:
    def __init__(self, model, metadata, weights_sha256):
        self.model = model.eval()
        self.patch_size = metadata["patch_size"]
        self.provenance = {"method": "supervised_candidate_patch_classifier", "weights_sha256": weights_sha256,
                           **metadata}

    def predict(self, patches):
        """Return uncalibrated class scores for N,H,W,3 uint8 RGB patches."""
        import torch
        patches = np.asarray(patches)
        if patches.ndim != 4 or patches.shape[-1] != 3 or patches.dtype != np.uint8:
            raise ValueError("patches must be an N,H,W,3 uint8 RGB array")
        if len(patches) == 0:
            return []
        if patches.shape[1:3] != (self.patch_size, self.patch_size):
            raise ValueError("Patch dimensions differ from the trained model")
        inputs = torch.from_numpy(patches.transpose(0, 3, 1, 2).copy()).float() / 255
        with torch.no_grad():
            return torch.sigmoid(self.model(inputs).squeeze(1)).numpy().astype(float).tolist()

    def score(self, frame, detections, preprocessing=None):
        """Score only frames matching the reviewed training preprocessing domain."""
        if not detections:
            return []
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Candidate scorer requires uint8 BGR source frames")
        height, width = frame.shape[:2]
        if self.provenance.get("preprocessing") != PREPROCESSING or preprocessing != PREPROCESSING:
            raise ValueError("Candidate scorer preprocessing signature differs from its training domain")
        if [width, height] not in [list(size) for size in self.provenance.get("analysis_sizes", [])]:
            raise ValueError("Candidate scorer frame size differs from its training domain")
        for detection in detections:
            x, y = detection.get("x"), detection.get("y")
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)) or not np.isfinite([x, y]).all() or not 0 <= x < width or not 0 <= y < height:
                raise ValueError("Candidate coordinates must lie inside the supplied frame")
        patches = np.stack([_patch(frame, d["x"], d["y"], self.patch_size) for d in detections])
        scores = self.predict(patches)
        return [{**d, "learned_confidence": score, "learned_score_calibrated": False,
                 "scorer_weights_sha256": self.provenance["weights_sha256"]}
                for d, score in zip(detections, scores)]


def load_candidate_scorer(path):
    """Load only this versioned tensor checkpoint using safe weights-only loading."""
    import torch
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metadata = checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
    if (not isinstance(checkpoint, dict) or checkpoint.get("format_version") != FORMAT_VERSION
            or metadata.get("architecture") != ARCHITECTURE
            or not isinstance(metadata.get("patch_size"), int)
            or not 16 <= metadata["patch_size"] <= 128
            or metadata.get("preprocessing") != PREPROCESSING
            or not isinstance(metadata.get("analysis_sizes"), list) or not metadata["analysis_sizes"]):
        raise ValueError("Unsupported candidate scorer checkpoint")
    model = _network()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return CandidateScorer(model, metadata, hashlib.sha256(path.read_bytes()).hexdigest())
