"""Source manifests and explicitly unverified annotation drafts.

Filename pairs are *candidate* shared rallies, not a synchronization assertion.
They are nevertheless kept together for a conservative leakage-safe split.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import re


def video_id(relative_path: str) -> str:
    return "v_" + hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_identity(name: str):
    stem = Path(name).stem
    match = re.search(r"(?<![A-Za-z])([ab])(?=시점)", stem, re.I)
    if match is None:
        match = re.search(r"^([ab])(?=-\d+시점)", stem, re.I)
    if match is None:
        return None, None
    return stem[:match.start()] + "{view}" + stem[match.end():], match[1].lower()


def build_manifest(video_dir, seed: int = 42) -> dict:
    """Hash MP4s and split connected rally/duplicate groups, never frames.

    ``videos[].id`` is based on the relative path so identical copies still have
    separate source records. ``sha256`` identifies content. ``group_id`` joins
    candidate A/B names and all same-content files transitively.
    """
    video_dir = Path(video_dir).resolve()
    paths = sorted(p for p in video_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".mp4")
    if not paths:
        raise ValueError(f"No MP4 inputs found in {video_dir}")
    rows = []
    for path in paths:
        relative = path.relative_to(video_dir).as_posix()
        pair_key, view = pair_identity(path.name)
        # Identically named clips in different folders are not assumed paired.
        if pair_key is not None:
            pair_key = str(Path(relative).parent / pair_key)
        rows.append({"id": video_id(relative), "file": relative, "path": str(path),
                     "sha256": file_hash(path), "bytes": path.stat().st_size,
                     "pair_key": pair_key, "view": view})
    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        parents[find(right)] = find(left)

    pair_first, hash_first = {}, {}
    for index, row in enumerate(rows):
        for key, lookup in ((row["pair_key"], pair_first), (row["sha256"], hash_first)):
            if key is None:
                continue
            if key in lookup:
                union(lookup[key], index)
            else:
                lookup[key] = index
    components = defaultdict(list)
    for index, row in enumerate(rows):
        components[find(index)].append(row)
    groups = []
    for members in components.values():
        files = sorted(row["file"] for row in members)
        group_id = "g_" + hashlib.sha256("\n".join(files).encode("utf-8")).hexdigest()[:12]
        group = {"group_id": group_id, "video_ids": [row["id"] for row in members],
                 "files": files, "pair_keys": sorted({row["pair_key"] for row in members if row["pair_key"]}),
                 "content_hashes": sorted({row["sha256"] for row in members})}
        group["duplicate_merge"] = len(group["content_hashes"]) < len(members)
        groups.append(group)
        for row in members:
            row["group_id"] = group_id
    ordered = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group['group_id']}".encode()).hexdigest())
    count = len(ordered)
    if count >= 3:
        validation_count = max(1, int(round(count * 0.15)))
        test_count = max(1, int(round(count * 0.15)))
        train_count = count - validation_count - test_count
    else:
        train_count, validation_count, test_count = count, 0, 0
    split_by_group = {}
    for index, group in enumerate(ordered):
        split = "train" if index < train_count else "val" if index < train_count + validation_count else "test"
        group["split"] = split
        split_by_group[group["group_id"]] = split
    for row in rows:
        row["split"] = split_by_group[row["group_id"]]
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
    duplicate_groups = [{"sha256": digest, "video_ids": [r["id"] for r in members],
                         "files": [r["file"] for r in members], "group_id": members[0]["group_id"]}
                        for digest, members in sorted(by_hash.items()) if len(members) > 1]
    fingerprint = hashlib.sha256("\n".join(f"{row['file']}\0{row['sha256']}" for row in rows).encode("utf-8")).hexdigest()
    return {"schema_version": 1, "video_directory": str(video_dir), "seed": seed,
            "dataset_fingerprint": fingerprint,
            "file_count": len(rows), "unique_content_count": len(by_hash), "group_count": count,
            "split_method": "deterministic connected rally-and-content groups; approximately 70/15/15 percent of groups",
            "split_group_counts": {"train": train_count, "val": validation_count, "test": test_count},
            "grouping_caveat": "A/B filename grouping is provisional and prevents likely leakage; it does not verify the same rally or temporal synchronization.",
            "generalization_evidence": {"independent_venues_verified": False,
                                        "independent_recording_dates_verified": False,
                                        "reason": "No independently reviewed venue/date/device metadata supplied; group holdout alone cannot establish cross-venue generalization."},
            "warnings": (["Fewer than three independent groups: no validation/test holdout can be formed."] if count < 3 else []),
            "videos": rows, "groups": sorted(groups, key=lambda group: group["group_id"]),
            "duplicate_groups": duplicate_groups}


def draft_annotations(results, output_dir) -> dict:
    """Write UNKNOWN machine drafts without overwriting any existing annotation.

    Pass the run's ``annotation-drafts`` directory, distinct from reviewed
    ``annotations``. Suggestions are copied separately and never become truth.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created, preserved = [], []
    if isinstance(results, dict):
        results = results.get("results", [results])
    for result in results:
        identifier = result.get("video_id") or result.get("id")
        if not identifier or not re.fullmatch(r"[A-Za-z0-9_-]+", str(identifier)):
            raise ValueError("Result requires a safe video_id")
        path = output_dir / f"{identifier}.json"
        if path.exists():
            preserved.append(str(path))
            continue
        suggestions = []
        for landing in result.get("landings", []):
            suggestions.append({key: landing.get(key) for key in
                                ("track_id", "frame_index", "time", "candidate_px", "confidence", "reason_codes")})
        draft = {"schema_version": 1, "video_id": identifier, "file": result.get("file"),
                 "source_sha256": result.get("source_sha256"),
                 "source": "machine_draft", "status": "needs_review", "reviewed": False,
                 "label": "UNKNOWN", "contact_observed": False, "time": None,
                 "contact_px": None, "court_id": None, "reviewer": None,
                 "independent_evidence": "", "near_line_distance_m": None,
                 "notes": "자동 생성된 검수 초안입니다. 후보를 확인하기 전에는 정답 라벨로 사용할 수 없습니다.",
                 "analysis": result.get("analysis", {}), "model_suggestions": suggestions,
                 "model_decision": result.get("decision"),
                 "created_at_utc": datetime.now(timezone.utc).isoformat()}
        # Exclusive creation also protects a reviewer saving concurrently.
        try:
            with path.open("x", encoding="utf-8") as handle:
                json.dump(draft, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
            created.append(str(path))
        except FileExistsError:
            preserved.append(str(path))
    return {"created_count": len(created), "preserved_count": len(preserved),
            "created": created, "preserved": preserved, "ground_truth_created": False}
