"""Inspect stream metadata and packet timestamp cadence without decoding frames."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
from pathlib import Path
import re
import statistics
import subprocess


def ratio(value):
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def inspect(path):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_packets",
               "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,start_time,time_base:packet=pts_time,dts_time,duration_time,flags",
               "-of", "json", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    raw = json.loads(result.stdout)
    stream = raw["streams"][0]
    packets = raw.get("packets", [])
    pts = [float(p["pts_time"]) for p in packets if "pts_time" in p]
    dts = [float(p["dts_time"]) for p in packets if "dts_time" in p]
    ordered = sorted(pts)
    deltas = [b - a for a, b in zip(ordered, ordered[1:])]
    positive = [delta for delta in deltas if delta > 0]
    median = statistics.median(positive) if positive else None
    nominal_fps = ratio(stream.get("r_frame_rate"))
    average_fps = ratio(stream.get("avg_frame_rate"))
    nominal_period = 1 / nominal_fps if nominal_fps else None
    presented_pts = sorted(float(p["pts_time"]) for p in packets if "pts_time" in p and "D" not in p.get("flags", ""))
    presented_deltas = [b - a for a, b in zip(presented_pts, presented_pts[1:])]
    presented_median = statistics.median(d for d in presented_deltas if d > 0) if presented_deltas else None
    presented_gaps = [{"previous_pts_seconds": a, "next_pts_seconds": b, "interval_ms": round((b-a)*1000, 4)}
                      for a, b in zip(presented_pts, presented_pts[1:])
                      if presented_median and (b-a) > presented_median * 1.5]
    gaps = [{"previous_pts_seconds": a, "next_pts_seconds": b, "interval_ms": round((b-a)*1000, 4)}
            for a, b in zip(ordered, ordered[1:]) if median and (b-a) > median * 1.5]
    metadata_frame_count = int(stream["nb_frames"]) if stream.get("nb_frames", "").isdigit() else None
    editlist_evidence = {"ignored_editlist_probe_performed": False,
                         "ignored_editlist_packet_count": None,
                         "editlist_excluded_packets": 0,
                         "packet_count_difference_explained_by_edit_list": False}
    # An MP4 edit list can suppress packets entirely rather than return them
    # with D flags. Probe the same file without applying edit lists only when
    # its raw frame metadata and default packet count disagree.
    if metadata_frame_count is not None and metadata_frame_count != len(packets):
        ignored_command = command[:-1] + ["-ignore_editlist", "1", str(path)]
        ignored_result = subprocess.run(ignored_command, capture_output=True, text=True, check=True)
        ignored_raw = json.loads(ignored_result.stdout)
        ignored_packets = ignored_raw.get("packets", [])
        ignored_stream = ignored_raw["streams"][0]
        ignored_pts = [float(p["pts_time"]) for p in ignored_packets if "pts_time" in p]
        difference = len(ignored_packets) - len(packets)
        editlist_evidence.update({
            "ignored_editlist_probe_performed": True,
            "ignored_editlist_packet_count": len(ignored_packets),
            "ignored_editlist_metadata_frame_count": int(ignored_stream["nb_frames"]) if ignored_stream.get("nb_frames", "").isdigit() else None,
            "ignored_editlist_stream_duration_seconds": float(ignored_stream["duration"]) if "duration" in ignored_stream else None,
            "ignored_editlist_first_pts_seconds": min(ignored_pts) if ignored_pts else None,
            "ignored_editlist_last_pts_seconds": max(ignored_pts) if ignored_pts else None,
            "ignored_editlist_discard_flagged_packets": sum("D" in p.get("flags", "") for p in ignored_packets),
            "ignored_editlist_ffprobe_stderr": ignored_result.stderr.strip(),
            "editlist_excluded_packets": difference,
            "packet_count_difference_explained_by_edit_list": difference > 0 and len(ignored_packets) == metadata_frame_count,
        })
    return {
        "file": path.name, "codec": stream.get("codec_name"),
        "width": stream.get("width"), "height": stream.get("height"),
        "nominal_fps": nominal_fps, "average_fps": average_fps,
        "nominal_fps_fraction": stream.get("r_frame_rate"),
        "average_fps_fraction": stream.get("avg_frame_rate"),
        "stream_duration_seconds": float(stream["duration"]) if "duration" in stream else None,
        "metadata_frame_count": metadata_frame_count,
        **editlist_evidence,
        "packet_count": len(packets), "pts_count": len(pts), "dts_count": len(dts),
        "discard_flagged_packets": sum("D" in p.get("flags", "") for p in packets),
        "presentable_packet_count": len(presented_pts),
        "presentable_interval_median_ms": round(presented_median * 1000, 5) if presented_median else None,
        "presentable_intervals_over_1_5x_median": presented_gaps,
        "first_pts_seconds": min(pts) if pts else None, "last_pts_seconds": max(pts) if pts else None,
        "duplicate_pts_count": len(pts) - len(set(pts)),
        "packet_order_pts_backsteps": sum(b < a for a, b in zip(pts, pts[1:])),
        "packet_order_dts_backsteps": sum(b < a for a, b in zip(dts, dts[1:])),
        "presentation_interval_median_ms": round(median * 1000, 5) if median else None,
        "presentation_interval_min_ms": round(min(deltas) * 1000, 5) if deltas else None,
        "presentation_interval_max_ms": round(max(deltas) * 1000, 5) if deltas else None,
        "intervals_over_1_5x_median": gaps,
        "intervals_outside_20pct_nominal": sum(abs(d - nominal_period) > nominal_period * 0.2 for d in deltas) if nominal_period else None,
        "ffprobe_stderr": result.stderr.strip(),
    }


def pair_key(name):
    match = re.search(r"([ab])(?=(?:-\d+)?시점)", name)
    if not match:
        return None, None
    return name[:match.start()] + "{camera}" + name[match.end():], match.group(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.video_dir.glob("*.mp4"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(inspect, paths))
    groups = Counter((r["codec"], r["width"], r["height"], r["nominal_fps_fraction"]) for r in rows)
    pairmap = {}
    for row in rows:
        key, camera = pair_key(row["file"])
        if key is not None:
            pairmap.setdefault(key, {})[camera] = row
    pairs = []
    for key, pair in sorted(pairmap.items()):
        if set(pair) != {"a", "b"}:
            continue
        a, b = pair["a"], pair["b"]
        difference = a["stream_duration_seconds"] - b["stream_duration_seconds"]
        pairs.append({"key": key, "a_file": a["file"], "b_file": b["file"],
                      "a_duration_seconds": a["stream_duration_seconds"],
                      "b_duration_seconds": b["stream_duration_seconds"],
                      "a_minus_b_duration_seconds": round(difference, 6),
                      "a_nominal_fps": a["nominal_fps"], "b_nominal_fps": b["nominal_fps"],
                      "duration_difference_gt_0_5_seconds": abs(difference) > 0.5})
    output = {
        "scope": "Independent ffprobe packet timestamp and metadata audit; no video decoding or synchronization verification",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.video_dir.parent, text=True).strip(),
        "notes": ["Presentation cadence uses sorted packet PTS to avoid mistaking B-frame packet ordering for faulty display order.",
                  "Packet PTS backsteps can be normal for B-frame streams; DTS backsteps are separately reported.",
                  "Packets marked D by ffprobe are discard/preroll data used for decoding; presentable cadence excludes those packets.",
                  "nb_frames and packet counts may include discard/preroll frames and therefore need not equal output frame counts after MP4 edit lists.",
                  "When nb_frames differs from default packet count, a second metadata-only probe with -ignore_editlist 1 checks whether the edit list suppressed packets entirely. This is separate from D-flagged preroll.",
                  "presentable_packet_count is packet-level evidence for reconciling decoder output; packets and frames are not universally one-to-one across codecs.",
                  "r_frame_rate is a base frame rate guess; some clips report 48 or 120 despite median/average rates near 24 or 60. Prefer presented timestamps for alignment.",
                  "Cadence alerts use >1.5x median and >20% nominal interval heuristics, not labeled pass/fail accuracy thresholds.",
                  "Same filename number and family identify candidate camera pairs only; equal duration would not prove temporal alignment.",
                  "Different nominal FPS requires timestamp-based alignment rather than same frame-index pairing.",
                  "Packet-level checks do not establish a physical camera's capture-time precision or rolling-shutter timing."],
        "summary": {"files": len(rows), "packets": sum(r["packet_count"] for r in rows),
                    "groups": [{"codec": k[0], "width": k[1], "height": k[2], "nominal_fps": k[3], "files": v} for k, v in sorted(groups.items())],
                    "files_with_duplicate_pts": [r["file"] for r in rows if r["duplicate_pts_count"]],
                    "files_with_dts_backsteps": [r["file"] for r in rows if r["packet_order_dts_backsteps"]],
                    "files_with_intervals_over_1_5x_median": [r["file"] for r in rows if r["intervals_over_1_5x_median"]],
                    "discard_flagged_packets": sum(r["discard_flagged_packets"] for r in rows),
                    "ignored_editlist_probe_files": sum(r["ignored_editlist_probe_performed"] for r in rows),
                    "editlist_excluded_packets": sum(r["editlist_excluded_packets"] for r in rows),
                    "files_with_editlist_excluded_packets": [r["file"] for r in rows if r["editlist_excluded_packets"]],
                    "presentable_packets": sum(r["presentable_packet_count"] for r in rows),
                    "files_with_discard_flagged_packets": [r["file"] for r in rows if r["discard_flagged_packets"]],
                    "files_with_presentable_intervals_over_1_5x_median": [r["file"] for r in rows if r["presentable_intervals_over_1_5x_median"]],
                    "files_with_intervals_outside_20pct_nominal": [r["file"] for r in rows if r["intervals_outside_20pct_nominal"]],
                    "candidate_pairs": len(pairs),
                    "pairs_with_different_nominal_fps": sum(p["a_nominal_fps"] != p["b_nominal_fps"] for p in pairs),
                    "pairs_duration_difference_gt_0_5_seconds": sum(p["duration_difference_gt_0_5_seconds"] for p in pairs),
                    "max_absolute_pair_duration_difference_seconds": max((abs(p["a_minus_b_duration_seconds"]) for p in pairs), default=None)},
        "files": rows, "candidate_pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
