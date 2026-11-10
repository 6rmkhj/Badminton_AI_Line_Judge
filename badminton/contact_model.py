"""Optional supervised temporal first-contact recognition and localization.

No weights are bundled. A motion heuristic or unreviewed pseudo-label cannot
train this module. Small-data models can propose contacts, but can certify one
only after recorded independent held-out precision/localization checks pass.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .learning import PREPROCESSING, _load_manifest, _patch

CLASSES = ("airborne", "first_contact", "after_contact")
ARCHITECTURE = "temporal_contact_cnn_v1"


def _records(manifest, annotations_dir):
    records, excluded = [], Counter()
    for video in manifest.get("videos", []):
        path = Path(annotations_dir) / f"{video['id']}.json"
        if not path.exists():
            excluded["missing_annotation"] += 1
            continue
        annotation = json.loads(path.read_text())
        if (annotation.get("source") != "human_review" or annotation.get("video_id", video["id"]) != video["id"]
                or not str(annotation.get("reviewer", "")).strip()
                or annotation.get("contact_sequences_verified") is not True):
            excluded["unverified_contact_sequences"] += 1
            continue
        if annotation.get("source_sha256") != video.get("sha256"):
            excluded["annotation_source_hash_missing_or_mismatched"] += 1
            continue
        size = annotation.get("analysis_size", {})
        width, height = size.get("width"), size.get("height")
        if not isinstance(width, int) or not isinstance(height, int) or not 2 <= width <= 16384 or not 2 <= height <= 16384:
            excluded["invalid_analysis_dimensions"] += 1
            continue
        seen = set()
        for label in annotation.get("contact_sequences", []):
            if not isinstance(label, dict) or label.get("label") not in CLASSES:
                excluded["invalid_contact_class"] += 1
                continue
            index, x, y = label.get("frame_index"), label.get("x"), label.get("y")
            if (not isinstance(index, int) or isinstance(index, bool) or index < 2
                    or not isinstance(x, (float, int)) or not isinstance(y, (float, int))
                    or not np.isfinite([x, y]).all() or not 0 <= x < width or not 0 <= y < height):
                excluded["invalid_contact_location"] += 1
                continue
            point = label.get("contact_px")
            if label["label"] == "first_contact":
                if (label.get("contact_observed") is not True or label.get("contact_point_verified") is not True
                        or not isinstance(point, list) or len(point) != 2
                        or not all(isinstance(v, (int, float)) for v in point) or not np.isfinite(point).all()
                        or not 0 <= point[0] < width or not 0 <= point[1] < height):
                    excluded["first_contact_requires_observed_cork_contact_point"] += 1
                    continue
            identity = (index, round(float(x), 3), round(float(y), 3))
            if identity in seen:
                raise ValueError(f"Repeated or contradictory temporal labels at {video['id']} frame {index}")
            seen.add(identity)
            independent = label.get("independent_evidence", "")
            records.append({**video, "frame_index": index, "x": float(x), "y": float(y),
                            "width": width, "height": height, "label": CLASSES.index(label["label"]),
                            "label_name": label["label"], "contact_px": point if label["label"] == "first_contact" else None,
                            "independent_evidence": independent.strip() if isinstance(independent, str) else ""})
    return records, dict(excluded)


def contact_training_readiness(manifest, annotations_dir, minimum_per_class=2):
    manifest = _load_manifest(manifest)
    if minimum_per_class < 1:
        raise ValueError("minimum_per_class must be positive")
    records, excluded = _records(manifest, annotations_dir)
    counts = {split: {name: 0 for name in CLASSES} for split in ("train", "val", "test")}
    reasons = []
    ids = [video.get("id") for video in manifest.get("videos", [])]
    if not all(ids) or len(set(ids)) != len(ids):
        reasons.append("Manifest video IDs must be present and unique")
    for key in ("group_id", "sha256", "pair_key"):
        splits = defaultdict(set)
        for video in manifest.get("videos", []):
            if video.get("split") not in counts or not video.get("group_id") or not video.get("sha256"):
                reasons.append("Every video requires a group, content SHA256, and train/val/test split")
            if video.get(key):
                splits[video[key]].add(video.get("split"))
        if any(len(value) > 1 for value in splits.values()):
            reasons.append(f"Dataset leakage across {key}")
    for record in records:
        if record.get("split") in counts:
            counts[record["split"]][record["label_name"]] += 1
    for split, classes in counts.items():
        for name, count in classes.items():
            if count < minimum_per_class:
                reasons.append(f"{split}: need {minimum_per_class} reviewed {name} sequences; found {count}")
    return {"ready": not reasons, "trained": False, "model_available": False,
            "reviewed_sequences": len(records), "counts": counts, "excluded": excluded,
            "reasons": sorted(set(reasons)), "minimum_per_class": minimum_per_class,
            "scope": "Temporal visible-first-contact proposals and cork-point localization, not IN/OUT decisions",
            "qualification_note": "Training prerequisites do not imply qualification for automatic contact certification."}


def build_contact_examples(manifest, annotations_dir, video_dir=None, patch_size=48, minimum_per_class=2):
    from .media import probe
    manifest = _load_manifest(manifest)
    if not isinstance(patch_size, int) or patch_size < 24 or patch_size > 128 or patch_size % 2:
        raise ValueError("patch_size must be even and between 24 and 128")
    readiness = contact_training_readiness(manifest, annotations_dir, minimum_per_class)
    if not readiness["ready"]:
        raise ValueError("Contact training data not ready: " + "; ".join(readiness["reasons"]))
    records, _ = _records(manifest, annotations_dir)
    by_video = defaultdict(list)
    for i, record in enumerate(records):
        by_video[record["id"]].append((i, record))
    clips, offsets, intervals = [None] * len(records), np.zeros((len(records), 2), dtype=np.float32), []
    for video_id, indexed in by_video.items():
        first = indexed[0][1]
        path = Path(first["path"]) if first.get("path") else Path(video_dir or ".") / first["file"]
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != first["sha256"]:
            raise ValueError(f"Changed source video: {video_id}")
        times = probe(path)["timestamps"]
        requested = defaultdict(list)
        for i, record in indexed:
            center = record["frame_index"]
            if center + 2 >= len(times):
                raise ValueError(f"Incomplete five-frame window for {video_id} frame {center}")
            steps = np.diff(times[center - 2:center + 3])
            if np.any(steps <= 0) or max(steps) > np.median(steps) * 1.75:
                raise ValueError("Contact training window contains a timestamp discontinuity")
            record["frame_interval_seconds"] = float(np.median(steps))
            intervals.append(record["frame_interval_seconds"])
            if record["contact_px"] is not None:
                offset = np.asarray(record["contact_px"]) - [record["x"], record["y"]]
                if np.any(np.abs(offset) > patch_size / 2 - 2):
                    raise ValueError("Confirmed contact point lies outside candidate crop")
                offsets[i] = offset / (patch_size / 2)
            clips[i] = [None] * 5
            for window_index, frame_index in enumerate(range(center - 2, center + 3)):
                requested[frame_index].append((i, window_index, record))
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        try:
            if not capture.isOpened():
                raise ValueError(f"Cannot decode source {video_id}")
            for frame_index in range(max(requested) + 1):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Source frame missing in {video_id}")
                for i, window_index, record in requested.get(frame_index, []):
                    resized = cv2.resize(frame, (record["width"], record["height"]), interpolation=cv2.INTER_AREA)
                    clips[i][window_index] = _patch(resized, record["x"], record["y"], patch_size)
        finally:
            capture.release()
    return {"report": readiness, "samples": records, "clips": np.asarray(clips, dtype=np.uint8),
            "offsets": offsets, "patch_size": patch_size,
            "frame_interval_range_seconds": [min(intervals), max(intervals)]}


def _network():
    import torch
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(torch.nn.Conv3d(3, 8, 3, padding=1), torch.nn.ReLU(),
                torch.nn.MaxPool3d((1, 2, 2)), torch.nn.Conv3d(8, 12, 3, padding=1), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool3d((3, 2, 2)), torch.nn.Flatten())
            self.classes = torch.nn.Linear(144, 3)
            self.offset = torch.nn.Linear(144, 2)
        def forward(self, clips):
            features = self.features(clips)
            return self.classes(features), torch.tanh(self.offset(features))
    return Network()


def _wilson_lower(success, total):
    if not total:
        return 0.0
    p, z = success / total, 1.959963984540054
    return (p + z*z/(2*total) - z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1+z*z/total)


def _contact_precision(labels, probabilities, threshold, groups=None):
    selected = (probabilities[:, 1] >= threshold) & (probabilities.argmax(axis=1) == 1)
    if groups is None:
        groups = [str(i) for i in range(len(labels))]
    selected_groups = defaultdict(list)
    for i, group in enumerate(groups):
        if selected[i]:
            selected_groups[group].append(bool(labels[i] == 1))
    # Correlated frames from the same rally count once; any false contact in a
    # selected group fails that group. Never inflate confidence with frames.
    successes = sum(all(values) for values in selected_groups.values())
    total = len(selected_groups)
    return {"correct": successes, "selected": total, "precision": successes / total if total else None,
            "wilson_95_lower": _wilson_lower(successes, total)}


def _qualified(metadata):
    evidence = metadata.get("qualification_evidence", {})
    validation, test = evidence.get("validation", {}), evidence.get("test", {})
    spatial = evidence.get("localization", {})
    split_groups = metadata.get("split_groups", {})
    validation_groups, test_groups = set(split_groups.get("val", [])), set(split_groups.get("test", []))
    def count_valid(values):
        correct, selected = values.get("correct"), values.get("selected")
        return (isinstance(correct, int) and isinstance(selected, int) and 0 <= correct <= selected
                and _wilson_lower(correct, selected) >= 0.95)
    total, covered = spatial.get("test_count", 0), spatial.get("test_covered", 0)
    bound = spatial.get("uncertainty_px")
    return bool(metadata.get("dataset_kind") == "real_video"
                and metadata.get("preprocessing") == PREPROCESSING
                and evidence.get("statistical_unit") == "rally_group"
                and validation_groups and test_groups and validation_groups.isdisjoint(test_groups)
                and validation_groups.isdisjoint(split_groups.get("train", []))
                and test_groups.isdisjoint(split_groups.get("train", []))
                and validation.get("selected", 0) <= len(validation_groups)
                and test.get("selected", 0) <= len(test_groups)
                and spatial.get("test_count", 0) <= len(test_groups)
                and evidence.get("all_holdout_labels_have_independent_evidence") is True
                and count_valid(validation) and count_valid(test)
                and isinstance(total, int) and isinstance(covered, int) and 0 <= covered <= total
                and _wilson_lower(covered, total) >= 0.90
                and isinstance(bound, (int, float)) and math.isfinite(bound) and 1 <= bound <= 4
                and 0.90 <= metadata.get("threshold", 0) <= 1)


def train_contact_model(manifest, annotations_dir, video_dir, output_path, epochs=20,
                        learning_rate=0.002, patch_size=48, minimum_per_class=2, seed=42):
    import torch
    if not isinstance(epochs, int) or not 1 <= epochs <= 1000 or not 0 < learning_rate <= 0.1:
        raise ValueError("Invalid training epochs or learning_rate")
    manifest = _load_manifest(manifest)
    prepared = build_contact_examples(manifest, annotations_dir, video_dir, patch_size, minimum_per_class)
    torch.manual_seed(seed)
    model = _network()
    inputs = torch.from_numpy(prepared["clips"].transpose(0, 4, 1, 2, 3).copy()).float() / 255
    labels = torch.tensor([r["label"] for r in prepared["samples"]], dtype=torch.long)
    offsets = torch.from_numpy(prepared["offsets"])
    indices = {split: torch.tensor([i for i, r in enumerate(prepared["samples"]) if r["split"] == split]) for split in ("train", "val", "test")}
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best, best_loss, best_epoch, history = None, float("inf"), None, []
    generator = torch.Generator().manual_seed(seed)
    def loss_for(index):
        logits, predicted = model(inputs[index])
        classification = torch.nn.functional.cross_entropy(logits, labels[index])
        positive = labels[index] == 1
        localization = torch.nn.functional.smooth_l1_loss(predicted[positive], offsets[index][positive]) if positive.any() else logits.sum() * 0
        return classification + 2 * localization
    for epoch in range(epochs):
        model.train()
        order = indices["train"][torch.randperm(len(indices["train"]), generator=generator)]
        losses = []
        for index in order.split(16):
            optimizer.zero_grad()
            loss = loss_for(index)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_for(indices["val"]))
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch + 1
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": validation_loss})
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        logits, predicted_offsets = model(inputs)
        probabilities = torch.softmax(logits, 1).numpy()
        position_errors = np.linalg.norm((predicted_offsets.numpy() - prepared["offsets"]) * patch_size / 2, axis=1)
    y = labels.numpy()
    val_index, test_index = indices["val"].numpy(), indices["test"].numpy()
    # Threshold is selected using validation only, never optimized on the test set.
    groups = np.asarray([r["group_id"] for r in prepared["samples"]])
    thresholds = np.arange(0.90, 1.0, 0.01)
    threshold = next((float(t) for t in thresholds if _contact_precision(y[val_index], probabilities[val_index], t, groups[val_index])["wilson_95_lower"] >= 0.95), 0.99)
    def group_errors(index):
        errors = defaultdict(list)
        for i in index:
            if y[i] == 1:
                errors[groups[i]].append(position_errors[i])
        return np.asarray([max(values) for values in errors.values()])
    val_errors = group_errors(val_index)
    uncertainty = max(1.0, float(np.quantile(val_errors, 0.95)))
    test_errors = group_errors(test_index)
    evidence = {"statistical_unit": "rally_group",
                "validation": _contact_precision(y[val_index], probabilities[val_index], threshold, groups[val_index]),
                "test": _contact_precision(y[test_index], probabilities[test_index], threshold, groups[test_index]),
                "all_holdout_labels_have_independent_evidence": all(r["independent_evidence"] for r in prepared["samples"] if r["split"] in ("val", "test")),
                "localization": {"uncertainty_px": uncertainty, "validation_count": int(len(val_errors)),
                                 "test_count": int(len(test_errors)), "test_covered": int(np.sum(test_errors <= uncertainty)),
                                 "test_mean_error_px": float(np.mean(test_errors))}}
    metrics = {}
    for split, index in indices.items():
        array = index.numpy()
        confusion = np.zeros((3, 3), dtype=int)
        for actual, predicted in zip(y[array], probabilities[array].argmax(axis=1)):
            confusion[actual, predicted] += 1
        metrics[split] = {"count": len(array), "accuracy": float(np.trace(confusion) / len(array)), "confusion": confusion.tolist()}
    metadata = {"architecture": ARCHITECTURE, "dataset_kind": manifest.get("dataset_kind", "real_video"),
                "preprocessing": PREPROCESSING,
                "classes": list(CLASSES), "window_frames": 5, "patch_size": patch_size,
                "threshold": threshold, "qualification_evidence": evidence,
                "frame_interval_range_seconds": prepared["frame_interval_range_seconds"],
                "analysis_sizes": sorted({(r["width"], r["height"]) for r in prepared["samples"]}),
                "selected_epoch": best_epoch, "epochs": epochs, "seed": seed,
                "label_source": "explicit_reviewed_temporal_classes_and_cork_points",
                "metrics": metrics, "probabilities_calibrated": False,
                "split_groups": {split: sorted({r["group_id"] for r in prepared["samples"] if r["split"] == split}) for split in indices}}
    metadata["automatic_contact_qualified"] = _qualified(metadata)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "metadata": metadata, "state_dict": best}, destination)
    report = {"trained": True, "model_path": str(destination.resolve()), **metadata,
              "weights_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "history": history,
              "note": "Temporal patch metrics do not establish event recall, target identity, court calibration or IN/OUT accuracy."}
    destination.with_suffix(destination.suffix + ".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


@lru_cache(maxsize=4)
def _load_checkpoint_cached(path_string, mtime_ns, size, inode, ctime_ns):
    import torch
    path = Path(path_string)
    stored = torch.load(path, map_location="cpu", weights_only=True)
    metadata = stored.get("metadata", {})
    if stored.get("format_version") != 1 or metadata.get("architecture") != ARCHITECTURE or metadata.get("classes") != list(CLASSES):
        raise ValueError("Unsupported temporal contact checkpoint")
    patch_size = metadata.get("patch_size")
    if not isinstance(patch_size, int) or not 24 <= patch_size <= 128 or patch_size % 2:
        raise ValueError("Invalid contact checkpoint patch size")
    model = _network()
    model.load_state_dict(stored["state_dict"], strict=True)
    model.eval()
    return model, metadata, hashlib.sha256(path.read_bytes()).hexdigest()


def infer_contact(frames, timestamps, candidate, checkpoint, preprocessing=None):
    """Return a proposed cork point; certify it only with qualified heldout evidence."""
    import torch
    path = Path(checkpoint).resolve()
    stat = path.stat()
    model, metadata, digest = _load_checkpoint_cached(str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino, stat.st_ctime_ns)
    patch_size = metadata["patch_size"]
    result = {"contact_observed": False, "contact_px": None, "probabilities": None,
              "model_origin": "learned_temporal_contact", "reason_codes": [],
              "provenance": {"weights_sha256": digest,
                             "architecture": ARCHITECTURE, "dataset_kind": metadata.get("dataset_kind"),
                             "preprocessing": metadata.get("preprocessing"),
                             "runtime_preprocessing": preprocessing,
                             "automatic_contact_qualified": _qualified(metadata),
                             "qualification_evidence": copy.deepcopy(metadata.get("qualification_evidence"))}}
    index, point = candidate.get("frame_index"), candidate.get("candidate_px")
    if point is None:
        point = candidate.get("contact_px")
    if (not isinstance(index, int) or index < 2 or index + 2 >= len(frames) or len(timestamps) != len(frames)
            or not isinstance(point, (list, tuple)) or len(point) != 2
            or not all(isinstance(v, (int, float)) for v in point) or not np.isfinite(point).all()):
        result["reason_codes"].append("CONTACT_MODEL_INCOMPLETE_WINDOW")
        return result
    window = frames[index - 2:index + 3]
    if any(frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 for frame in window):
        raise ValueError("Contact model frames must be uint8 BGR images")
    height, width = window[0].shape[:2]
    if any(frame.shape != window[0].shape for frame in window) or not 0 <= point[0] < width or not 0 <= point[1] < height:
        result["reason_codes"].append("CONTACT_MODEL_INVALID_CROP")
        return result
    steps = np.diff(np.asarray(timestamps[index - 2:index + 3], dtype=float))
    if not np.isfinite(steps).all() or np.any(steps <= 0) or max(steps) > np.median(steps) * 1.75:
        result["reason_codes"].append("CONTACT_MODEL_TIMESTAMP_GAP")
        return result
    trained_sizes = metadata.get("analysis_sizes", [])
    interval_range = metadata.get("frame_interval_range_seconds", [])
    compatible = ([width, height] in [list(size) for size in trained_sizes]
                  and len(interval_range) == 2 and interval_range[0] * 0.8 <= np.median(steps) <= interval_range[1] * 1.2)
    clips = np.stack([_patch(frame, point[0], point[1], patch_size) for frame in window])
    tensor = torch.from_numpy(clips.transpose(3, 0, 1, 2).copy()[None]).float() / 255
    with torch.no_grad():
        logits, offset = model(tensor)
        probabilities = torch.softmax(logits, 1)[0].numpy()
        proposed = np.asarray(point) + offset[0].numpy() * patch_size / 2
    result.update(probabilities={name: float(probabilities[i]) for i, name in enumerate(CLASSES)},
                  proposed_contact_px=proposed.astype(float).tolist(), frame_index=index,
                  time=float(timestamps[index]), confidence=float(probabilities[1]),
                  confidence_kind="uncalibrated_softmax_with_heldout_threshold_evidence")
    if metadata.get("preprocessing") != PREPROCESSING or preprocessing != PREPROCESSING:
        result["reason_codes"].append("CONTACT_MODEL_PREPROCESSING_MISMATCH")
    if not compatible:
        result["reason_codes"].append("CONTACT_MODEL_CAPTURE_DOMAIN_MISMATCH")
    if not _qualified(metadata):
        result["reason_codes"].append("CONTACT_MODEL_UNQUALIFIED_HOLDOUT_EVIDENCE")
    if probabilities.argmax() != 1 or probabilities[1] < metadata.get("threshold", 1):
        result["reason_codes"].append("CONTACT_MODEL_FIRST_CONTACT_NOT_CONFIRMED")
    if not 0 <= proposed[0] < width or not 0 <= proposed[1] < height:
        result["reason_codes"].append("CONTACT_MODEL_POINT_OUTSIDE_IMAGE")
    if not result["reason_codes"]:
        uncertainty = metadata["qualification_evidence"]["localization"]["uncertainty_px"]
        result.update(contact_observed=True, contact_px=result["proposed_contact_px"],
                      contact_radius_px=0.0, uncertainty_px=float(uncertainty) + 1.0,
                      contact_evidence_source="qualified_learned_contact_localization")
    return result
