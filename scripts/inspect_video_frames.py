"""Sample real clips and exercise OpenCV preprocessing; no AI accuracy claims."""
import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def group_name(name):
    camera = re.search(r"([ab])(?:-\d+)?시점", name)
    camera = camera.group(1) if camera else "unknown"
    family = ("8k30_4k24" if name.startswith("8k") else
              "uhd30_4k30" if "uhd30 4k30" in name else
              "uhd30_hd30" if "uhd30 hd30" in name else "base")
    return f"{family}_{camera}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    rows, representatives, failures = [], {}, []
    paths = sorted(args.video_dir.glob("*.mp4"))
    for index, path in enumerate(paths, 1):
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_N_THREADS, 2])
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        group = group_name(path.name)
        for fraction in (0.25, 0.5, 0.75):
            target = max(0, min(count - 1, int(count * fraction)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = capture.read()
            if not ok:
                failures.append({"file": path.name, "target_frame": target})
                continue
            height, width = frame.shape[:2]
            scale = min(1.0, 1280 / width)
            sample = cv2.resize(frame, (round(width * scale), round(height * scale)),
                                interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,
                                    minLineLength=120, maxLineGap=20)
            candidates = [] if lines is None else lines[:, 0].tolist()
            overlay = sample.copy()
            for x1, y1, x2, y2 in candidates:
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 210, 255), 2)
            sample_id = f"{index:03d}_{int(fraction * 100):02d}"
            original_file = samples_dir / f"{sample_id}.jpg"
            overlay_file = samples_dir / f"{sample_id}_lines.jpg"
            if not cv2.imwrite(str(original_file), sample) or not cv2.imwrite(str(overlay_file), overlay):
                raise RuntimeError(f"Could not save {sample_id}")
            row = {"file": path.name, "group": group, "fraction": fraction,
                   "requested_frame": target, "reported_frame_after_read": capture.get(cv2.CAP_PROP_POS_FRAMES),
                   "reported_time_ms": capture.get(cv2.CAP_PROP_POS_MSEC),
                   "nominal_time_seconds": target / fps if fps else None,
                   "display_width": width, "display_height": height,
                   "analysis_width": sample.shape[1], "analysis_height": sample.shape[0],
                   "gray_mean": round(float(gray.mean()), 3),
                   "laplacian_variance": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 3),
                   "dark_fraction_le_5": round(float((gray <= 5).mean()), 5),
                   "bright_fraction_ge_250": round(float((gray >= 250).mean()), 5),
                   "hough_line_candidate_count": len(candidates),
                   "sample": str(original_file.relative_to(args.output_dir)),
                   "overlay": str(overlay_file.relative_to(args.output_dir))}
            rows.append(row)
            if fraction == 0.5 and group not in representatives:
                representatives[group] = row
        capture.release()
        print(json.dumps({"completed": index, "total": len(paths), "file": path.name}, ensure_ascii=False), flush=True)
    if rows:
        with (args.output_dir / "frame_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    summary = {"scope": "OpenCV sampling and generic Canny/Hough line candidates; not court calibration, shuttle detection, or IN/OUT validation",
               "parameters": {"sample_fractions": [0.25, 0.5, 0.75], "max_analysis_width": 1280,
                              "canny": [50, 150], "hough_votes": 80, "min_line_length": 120, "max_line_gap": 20},
               "videos": len(paths), "expected_samples": len(paths) * 3, "successful_samples": len(rows),
               "failed_samples": failures,
               "samples_with_line_candidates": sum(row["hough_line_candidate_count"] > 0 for row in rows),
               "representatives": representatives,
               "notes": ["Line candidates include court lines, other courts, walls and equipment; counts are not detection accuracy.",
                         "Brightness and Laplacian values are descriptive; no labeled quality threshold is applied.",
                         "Sampled frames cannot prove first landing contact visibility or complete frame-time regularity."]}
    (args.output_dir / "frame_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    # One original/overlay row per capture family and camera, with ASCII labels.
    if representatives:
        sheet = Image.new("RGB", (1280, 390 * len(representatives)), "#111827")
        draw = ImageDraw.Draw(sheet)
        for i, (group, row) in enumerate(sorted(representatives.items())):
            draw.text((12, i * 390 + 8), f"{group} | mid-clip sample | original / generic line candidates", fill="white")
            for column, key in enumerate(("sample", "overlay")):
                picture = Image.open(args.output_dir / row[key])
                picture.thumbnail((640, 350))
                sheet.paste(picture, (column * 640 + (640 - picture.width) // 2,
                                      i * 390 + 32 + (350 - picture.height) // 2))
        sheet.save(args.output_dir / "representative_frames.jpg", quality=88)
    print(json.dumps({"successful_samples": len(rows), "failures": len(failures)}, ensure_ascii=False), flush=True)
    return 1 if failures or not paths else 0


if __name__ == "__main__":
    raise SystemExit(main())
