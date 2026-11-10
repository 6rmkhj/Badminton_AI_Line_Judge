#!/usr/bin/env python3
"""Validate real video inputs, not AI inference or line-judging accuracy.

Only Python's standard library and ffmpeg/ffprobe are required. Every primary
video stream is decoded in full; audio streams are outside this check's scope.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


SCOPE = (
    "Real-video input integrity and metadata validation only. No trained model, "
    "court calibration, shuttle tracking, landing detection, IN/OUT accuracy, "
    "or camera synchronization is evaluated."
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def numeric(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def frame_rate(value):
    try:
        return float(Fraction(value))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def inferred_pair(path):
    """Replace only a known A/B view marker, preserving the recording family."""
    stem = path.stem
    # Examples: a시점 1, a시점(uhd30 hd30) 1, 8k 30 4k 24 a시점 1.
    match = re.search(r"(?<![A-Za-z])([ab])(?=시점)", stem, re.IGNORECASE)
    if match is None:
        # Example: a-1시점(uhd30 4k30).
        match = re.search(r"^([ab])(?=-\d+시점)", stem, re.IGNORECASE)
    if match is None:
        return None, None
    return stem[:match.start()] + "{view}" + stem[match.end():], match[1].lower()


def hashes(path):
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1()
    git_blob.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            sha256.update(chunk)
            git_blob.update(chunk)
    return sha256.hexdigest(), git_blob.hexdigest()


def apply_packet_accounting(row, packet_count, discard_count, source, ignored_editlist_packet_count=None, editlist_excluded_packets=None):
    """Reconcile MP4 edit-list preroll for H.264/HEVC, retaining raw counts.

    A packet is not universally a frame. This reconciliation is restricted to
    these codecs and requires matching metadata/packet counts, or an independent
    ignore_editlist probe explaining the difference. It never changes a failed
    decoder result into a successful one.
    """
    if not isinstance(packet_count, int) or not isinstance(discard_count, int) or not 0 <= discard_count <= packet_count:
        raise ValueError("Invalid packet/discard counts")
    for flag in (
        "metadata_and_decoded_frame_count_mismatch", "metadata_frame_count_includes_discard_preroll",
        "metadata_and_packet_count_mismatch", "unexpected_presented_frame_count_mismatch",
        "metadata_frame_count_includes_edit_list_exclusions",
    ):
        row["flags"] = [existing for existing in row["flags"] if existing != flag]
    row.update({
        "packet_count": packet_count,
        "discard_flagged_packets": discard_count,
        "expected_presented_packet_count": packet_count - discard_count,
        "packet_accounting_source": source,
        "metadata_and_packet_count_match": None,
        "expected_presented_frame_count": None,
        "presented_frame_count_matches": None,
        "frame_count_accounting_status": "unavailable",
        "ignored_editlist_packet_count": ignored_editlist_packet_count,
        "editlist_excluded_packets": editlist_excluded_packets,
        "editlist_exclusion_verified": False,
    })
    metadata_count, decoded_count = row.get("metadata_frame_count"), row.get("decoded_frames")
    editlist_verified = (
        row.get("codec") in ("h264", "hevc")
        and isinstance(metadata_count, int)
        and isinstance(ignored_editlist_packet_count, int)
        and ignored_editlist_packet_count == metadata_count
        and packet_count < metadata_count
        and editlist_excluded_packets == ignored_editlist_packet_count - packet_count
    )
    row["editlist_exclusion_verified"] = editlist_verified
    if metadata_count is not None:
        row["metadata_and_packet_count_match"] = metadata_count == packet_count
        if metadata_count != packet_count:
            row["flags"].append("metadata_and_packet_count_mismatch")
    row["frame_count_matches"] = metadata_count == decoded_count if metadata_count is not None and decoded_count is not None else None
    if row["frame_count_matches"] is False:
        row["flags"].append("metadata_and_decoded_frame_count_mismatch")
    if row.get("codec") in ("h264", "hevc"):
        base_count = packet_count if editlist_verified or metadata_count is None else metadata_count
        row["expected_presented_frame_count"] = base_count - discard_count
        if decoded_count is not None:
            row["presented_frame_count_matches"] = decoded_count == row["expected_presented_frame_count"] and (row["metadata_and_packet_count_match"] is not False or editlist_verified)
    else:
        # Do not infer one packet per frame for arbitrary video codecs.
        row["expected_presented_frame_count"] = metadata_count
        row["presented_frame_count_matches"] = row["frame_count_matches"]
    if row["presented_frame_count_matches"] is True:
        if editlist_verified:
            row["frame_count_accounting_status"] = "explained_by_edit_list"
            row["flags"].append("metadata_frame_count_includes_edit_list_exclusions")
            if discard_count:
                row["flags"].append("metadata_frame_count_includes_discard_preroll")
        elif row["frame_count_matches"] is False and discard_count:
            row["frame_count_accounting_status"] = "explained_by_discard_preroll"
            row["flags"].append("metadata_frame_count_includes_discard_preroll")
        else:
            row["frame_count_accounting_status"] = "matched"
    elif row["presented_frame_count_matches"] is False:
        row["frame_count_accounting_status"] = "unexpected_mismatch"
        row["flags"].append("unexpected_presented_frame_count_mismatch")
    return row


def timing_index(timing_audit):
    rows = timing_audit.get("files", [])
    result = {row["file"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Timing audit contains duplicate filenames")
    return result


def reconcile_row(row, audit_row, source):
    if audit_row.get("codec") != row.get("codec") or audit_row.get("metadata_frame_count") != row.get("metadata_frame_count"):
        raise ValueError(f"Timing audit metadata does not match {row['file']}")
    return apply_packet_accounting(row, audit_row["packet_count"], audit_row["discard_flagged_packets"], source, audit_row.get("ignored_editlist_packet_count"), audit_row.get("editlist_excluded_packets"))


def count_summary(rows):
    failed = [row["file"] for row in rows if row["probe_status"] != "passed" or row["decode_status"] != "passed" or row.get("error") or row.get("presented_frame_count_matches", row["frame_count_matches"]) is False]
    return {
        "metadata_frame_count_matches": dict(Counter(str(row["frame_count_matches"]) for row in rows)),
        "presented_frame_count_matches": dict(Counter(str(row.get("presented_frame_count_matches")) for row in rows)),
        "frame_count_accounting_status_counts": dict(Counter(row.get("frame_count_accounting_status", "unavailable") for row in rows)),
        "total_video_packets": sum(row.get("packet_count") or 0 for row in rows),
        "total_discard_flagged_packets": sum(row.get("discard_flagged_packets") or 0 for row in rows),
        "total_verified_editlist_excluded_packets": sum(row.get("editlist_excluded_packets") or 0 for row in rows if row.get("editlist_exclusion_verified")),
        "total_expected_presented_packets": sum(row.get("expected_presented_packet_count") or 0 for row in rows),
        "flag_counts": dict(Counter(flag for row in rows for flag in row["flags"])),
        "all_input_integrity_checks_passed": bool(rows) and not failed,
        "failed_files": failed,
    }


def reconcile_summary(summary, timing_audit, source="timing_audit.json"):
    """Return a copied summary reclassified from packet evidence; no decoding.

    Caller should ensure the audit is from the same source files. Filenames,
    codecs and metadata frame counts are checked here; content hashes from the
    original validation remain intact for independent provenance checking.
    """
    result = copy.deepcopy(summary)
    indexed = timing_index(timing_audit)
    for row in result["files"]:
        if row["probe_status"] == "passed":
            if row["file"] not in indexed:
                raise ValueError(f"Timing audit missing {row['file']}")
            reconcile_row(row, indexed[row["file"]], source)
    result.update(count_summary(result["files"]))
    result["frame_accounting_reconciled_at_utc"] = utc_now()
    result["frame_accounting_audit_source"] = source
    result["frame_accounting_caveat"] = "Raw MP4 nb_frames can include discard/preroll packets. H.264/HEVC output counts are compared to presented packets excluding D-flagged preroll. Metadata/packet differences are accepted only when an ignore_editlist probe confirms the original metadata count and edit-list exclusion count. Decoder success is checked separately."
    return result


def run_file(path, index, video_dir, output_dir, ffmpeg, ffprobe, timeout, audit_row=None):
    started = time.monotonic()
    relative_path = str(path.relative_to(video_dir))
    artifact_name = f"{index:03d}-{hashlib.sha256(relative_path.encode()).hexdigest()[:12]}"
    probe_path = output_dir / "metadata" / f"{artifact_name}.json"
    decode_log_path = output_dir / "logs" / f"{artifact_name}.decode.log"
    progress_path = output_dir / "logs" / f"{artifact_name}.ffmpeg-progress.log"
    pair_key, view = inferred_pair(path)
    row = {
        "file": relative_path,
        "bytes": path.stat().st_size,
        "sha256": None,
        "git_blob_sha1": None,
        "probe_status": "not_run",
        "decode_status": "not_run",
        "decoded_frames": None,
        "metadata_frame_count": None,
        "frame_count_matches": None,
        "codec": None,
        "width": None,
        "height": None,
        "pixel_format": None,
        "avg_fps": None,
        "nominal_fps": None,
        "duration_seconds": None,
        "video_stream_count": None,
        "inferred_pair_key": pair_key,
        "inferred_view": view,
        "flags": [],
        "error": None,
        "metadata_path": str(probe_path.relative_to(output_dir)),
        "decode_log_path": str(decode_log_path.relative_to(output_dir)),
        "decode_progress_path": str(progress_path.relative_to(output_dir)),
    }
    try:
        row["sha256"], row["git_blob_sha1"] = hashes(path)
        probe_command = [
            ffprobe, "-v", "error", "-show_streams", "-show_format",
            "-print_format", "json", str(path),
        ]
        if audit_row is None:
            probe_command[1:1] = ["-show_packets"]
        probe = subprocess.run(probe_command, capture_output=True, text=True, timeout=60)
        if probe.returncode:
            row["probe_status"] = "failed"
            row["error"] = probe.stderr.strip() or f"ffprobe exit {probe.returncode}"
            probe_path.write_text(json.dumps({"error": row["error"]}, ensure_ascii=False, indent=2) + "\n")
            return row
        metadata = json.loads(probe.stdout)
        probe_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        streams = [stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"]
        row["video_stream_count"] = len(streams)
        if not streams:
            row["probe_status"] = "failed"
            row["error"] = "No video stream found"
            return row
        stream = streams[0]
        row["probe_status"] = "passed"
        row.update({
            "codec": stream.get("codec_name"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "pixel_format": stream.get("pix_fmt"),
            "avg_fps": frame_rate(stream.get("avg_frame_rate")),
            "nominal_fps": frame_rate(stream.get("r_frame_rate")),
            "duration_seconds": numeric(stream.get("duration")) or numeric(metadata.get("format", {}).get("duration")),
        })
        if str(stream.get("nb_frames", "")).isdigit():
            row["metadata_frame_count"] = int(stream["nb_frames"])
        if audit_row is None:
            # ffprobe includes all streams; use exactly the primary video stream
            # decoded by -map 0:v:0, excluding audio and any secondary videos.
            packets = [packet for packet in metadata.get("packets", []) if packet.get("stream_index") == stream["index"]]
            packet_count = len(packets)
            discard_count = sum("D" in packet.get("flags", "") for packet in packets)
            ignored_editlist_packet_count = editlist_excluded_packets = None
            if row["codec"] in ("h264", "hevc") and row["metadata_frame_count"] is not None and packet_count < row["metadata_frame_count"]:
                unedited_command = [ffprobe, "-v", "error", "-ignore_editlist", "1", "-select_streams", "v:0", "-show_packets", "-show_entries", "packet=stream_index,flags", "-print_format", "json", str(path)]
                unedited = subprocess.run(unedited_command, capture_output=True, text=True, timeout=60)
                evidence = {"command": unedited_command, "return_code": unedited.returncode, "stderr": unedited.stderr}
                if unedited.returncode == 0 and not unedited.stderr.strip():
                    evidence["probe"] = json.loads(unedited.stdout)
                    ignored_editlist_packet_count = len(evidence["probe"].get("packets", []))
                    editlist_excluded_packets = ignored_editlist_packet_count - packet_count
                metadata["editlist_accounting_evidence"] = evidence
                probe_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        else:
            packet_count, discard_count = audit_row["packet_count"], audit_row["discard_flagged_packets"]
        if len(streams) > 1:
            row["flags"].append("multiple_video_streams_primary_only_decoded")
        if not row["avg_fps"]:
            row["flags"].append("average_frame_rate_unavailable")
        elif row["avg_fps"] < 59:
            row["flags"].append("below_60fps_reference_not_a_failure")
        if row["avg_fps"] and row["nominal_fps"] and abs(row["avg_fps"] / row["nominal_fps"] - 1) > 0.01:
            row["flags"].append("average_and_nominal_frame_rates_differ_not_proof_of_vfr")
        if not row["duration_seconds"]:
            row["flags"].append("duration_missing_or_zero")
        if row["width"] and row["height"] and max(row["width"], row["height"]) < 3840:
            row["flags"].append("below_4k_reference_not_a_failure")
        # -xerror and explode turn decode errors into a failed process. Disabling
        # output frame-rate conversion preserves the actual decoded frame count.
        decode_command = [
            ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-xerror",
            "-err_detect", "explode", "-threads", "2", "-filter_threads", "1",
            "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
            "-fps_mode", "passthrough", "-enc_time_base", "demux", "-threads", "2", "-f", "null",
            "-progress", "pipe:1", "-nostats", "-",
        ]
        row["decode_command"] = decode_command
        try:
            with decode_log_path.open("w") as error_log, progress_path.open("w") as progress_log:
                process = subprocess.run(decode_command, stdout=progress_log, stderr=error_log, timeout=timeout)
            progress = progress_path.read_text(errors="replace")
            frame_counts = re.findall(r"^frame=(\d+)\s*$", progress, flags=re.MULTILINE)
            row["decoded_frames"] = int(frame_counts[-1]) if frame_counts else None
            errors = decode_log_path.read_text(errors="replace").strip()
            row["decode_return_code"] = process.returncode
            row["decode_progress_finished"] = bool(re.search(r"^progress=end\s*$", progress, flags=re.MULTILINE))
            if process.returncode != 0 or errors or not row["decode_progress_finished"] or not row["decoded_frames"]:
                row["decode_status"] = "failed"
                row["error"] = errors[-4000:] or f"Decode incomplete; exit={process.returncode}, frames={row['decoded_frames']}"
            else:
                row["decode_status"] = "passed"
            if audit_row is None:
                apply_packet_accounting(row, packet_count, discard_count, "ffprobe_primary_video_packets", ignored_editlist_packet_count, editlist_excluded_packets)
            else:
                reconcile_row(row, audit_row, "supplied_timing_audit")
            if row["decoded_frames"] and row["duration_seconds"] and row["avg_fps"]:
                derived_duration = row["decoded_frames"] / row["avg_fps"]
                row["frame_count_duration_delta_seconds"] = derived_duration - row["duration_seconds"]
                if abs(derived_duration - row["duration_seconds"]) > max(0.1, 2 / row["avg_fps"]):
                    row["flags"].append("frame_count_and_metadata_duration_differ")
        except subprocess.TimeoutExpired:
            row["decode_status"] = "timeout"
            row["error"] = f"Full decode exceeded {timeout} seconds"
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        if row["probe_status"] == "not_run":
            row["probe_status"] = "failed"
        elif row["decode_status"] == "not_run":
            row["decode_status"] = "failed"
    finally:
        row["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return row


def summarize(rows, args, started_at, elapsed, versions):
    by_sha = defaultdict(list)
    by_blob = defaultdict(list)
    by_pair = defaultdict(list)
    for row in rows:
        if row["sha256"]:
            by_sha[row["sha256"]].append(row["file"])
        if row["git_blob_sha1"]:
            by_blob[row["git_blob_sha1"]].append(row["file"])
        if row["inferred_pair_key"]:
            by_pair[row["inferred_pair_key"]].append(row)
    pairs = []
    for key, members in sorted(by_pair.items()):
        a = [row for row in members if row["inferred_view"] == "a"]
        b = [row for row in members if row["inferred_view"] == "b"]
        entry = {"key": key, "files": [row["file"] for row in members], "status": "one_a_one_b" if len(a) == len(b) == 1 else "missing_or_ambiguous_view"}
        if len(a) == len(b) == 1:
            if a[0]["duration_seconds"] is not None and b[0]["duration_seconds"] is not None:
                entry["duration_difference_seconds"] = round(abs(a[0]["duration_seconds"] - b[0]["duration_seconds"]), 6)
            entry["identical_bytes"] = bool(a[0]["sha256"] and a[0]["sha256"] == b[0]["sha256"])
        pairs.append(entry)
    return {
        "scope": SCOPE,
        "started_at_utc": started_at,
        "finished_at_utc": utc_now(),
        "elapsed_seconds": round(elapsed, 3),
        "video_directory": str(args.video_dir),
        "workers": args.workers,
        "ffmpeg_decode_threads_per_worker": 2,
        "versions": versions,
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "total_duration_seconds": round(sum(row["duration_seconds"] or 0 for row in rows), 6),
        "total_decoded_frames": sum(row["decoded_frames"] or 0 for row in rows),
        "probe_status_counts": dict(Counter(row["probe_status"] for row in rows)),
        "decode_status_counts": dict(Counter(row["decode_status"] for row in rows)),
        "resolution_counts": dict(Counter(f"{row['width']}x{row['height']}" for row in rows)),
        "average_fps_counts": dict(Counter(str(round(row["avg_fps"], 6)) if row["avg_fps"] else "unknown" for row in rows)),
        "codec_counts": dict(Counter(row["codec"] or "unknown" for row in rows)),
        **count_summary(rows),
        "frame_accounting_caveat": "Raw MP4 nb_frames can include discard/preroll packets. H.264/HEVC output counts are compared to presented packets excluding D-flagged preroll. Metadata/packet differences are accepted only when an ignore_editlist probe confirms the original metadata count and edit-list exclusion count. Decoder success is checked separately.",
        "sha256_duplicate_groups": [{"sha256": key, "files": value} for key, value in by_sha.items() if len(value) > 1],
        "git_blob_duplicate_groups": [{"git_blob_sha1": key, "files": value} for key, value in by_blob.items() if len(value) > 1],
        "unique_content_count": len(by_sha),
        "pairing_caveat": "Groups are inferred solely from A/B view markers in filenames. Matching names, durations or frame counts do not establish the same rally, alignment, or synchronization.",
        "inferred_pair_count": sum(pair["status"] == "one_a_one_b" for pair in pairs),
        "inferred_pairs": pairs,
        "files": rows,
    }


def write_reports(summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    fields = ["file", "bytes", "sha256", "git_blob_sha1", "codec", "width", "height", "pixel_format", "avg_fps", "nominal_fps", "duration_seconds", "metadata_frame_count", "decoded_frames", "frame_count_matches", "packet_count", "discard_flagged_packets", "expected_presented_packet_count", "expected_presented_frame_count", "presented_frame_count_matches", "frame_count_accounting_status", "ignored_editlist_packet_count", "editlist_excluded_packets", "editlist_exclusion_verified", "probe_status", "decode_status", "inferred_pair_key", "inferred_view", "elapsed_seconds", "flags", "error", "metadata_path", "decode_log_path", "decode_progress_path"]
    with (output_dir / "files.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in summary["files"]:
            writer.writerow({**row, "flags": ";".join(row["flags"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timing-audit", type=Path, help="Reuse an independent packet audit for the same source files")
    parser.add_argument("--reconcile-summary", type=Path, help="Reclassify this existing summary with --timing-audit, without decoding or overwriting progress logs")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--decode-timeout", type=float, default=1800, help="Seconds allowed per full primary-video-stream decode (default: 1800)")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    audit = json.loads(args.timing_audit.read_text()) if args.timing_audit else None
    if args.reconcile_summary:
        if audit is None:
            parser.error("--reconcile-summary requires --timing-audit")
        summary = reconcile_summary(json.loads(args.reconcile_summary.read_text()), audit, str(args.timing_audit.resolve()))
        write_reports(summary, args.output_dir)
        print(json.dumps({"event": "reconciled_without_decoding", **count_summary(summary["files"])}, ensure_ascii=False), flush=True)
        return 0 if summary["all_input_integrity_checks_passed"] else 1
    if args.video_dir is None:
        parser.error("--video-dir is required unless --reconcile-summary is used")
    args.video_dir = args.video_dir.resolve()
    audit_rows = timing_index(audit) if audit else {}
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")
    if args.decode_timeout <= 0:
        parser.error("--decode-timeout must be positive")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        parser.error("ffmpeg and ffprobe must be installed and on PATH")
    files = sorted(path for path in args.video_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4")
    if not files:
        parser.error(f"No MP4 files found in {args.video_dir}")
    for folder in (args.output_dir, args.output_dir / "logs", args.output_dir / "metadata"):
        folder.mkdir(parents=True, exist_ok=True)
    versions = {name: subprocess.check_output([binary, "-version"], text=True).splitlines()[0] for name, binary in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe))}
    started_at, started = utc_now(), time.monotonic()
    rows = []
    with (args.output_dir / "progress.jsonl").open("w") as progress:
        def emit(event):
            event["timestamp_utc"] = utc_now()
            encoded = json.dumps(event, ensure_ascii=False)
            progress.write(encoded + "\n")
            progress.flush()
            print(encoded, flush=True)
        emit({"event": "started", "files": len(files), "workers": args.workers, "scope": SCOPE})
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_file, path, index, args.video_dir, args.output_dir, ffmpeg, ffprobe, args.decode_timeout, audit_rows.get(str(path.relative_to(args.video_dir)))): path for index, path in enumerate(files, 1)}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                emit({"event": "file_completed", "completed": len(rows), "total": len(files), **row})
        rows.sort(key=lambda row: row["file"])
        summary = summarize(rows, args, started_at, time.monotonic() - started, versions)
        write_reports(summary, args.output_dir)
        emit({"event": "finished", "file_count": len(rows), "passed": summary["all_input_integrity_checks_passed"], "elapsed_seconds": summary["elapsed_seconds"], "summary_path": str(args.output_dir / "summary.json")})
    return 0 if summary["all_input_integrity_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
