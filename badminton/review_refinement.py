"""Bounded source-bound AI review hints for explicitly assisted reanalysis."""
from __future__ import annotations

import math


def review_regions(annotation, analysis_width=None, frame_count=None):
    """Keep labels as search hints, never contact/identity confirmations.

    Coordinates remain in the pipeline's analysis pixel space. Native ROI
    extraction maps them back to source pixels. No final resting-point inference
    or guessed region is created for unobserved contacts.
    """
    if not isinstance(annotation, dict):
        return []
    size = annotation.get('analysis_size') or {}
    if not isinstance(size, dict):
        return []
    width, height = size.get('width'), size.get('height')
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
               for v in (width, height, analysis_width if analysis_width is not None else width)):
        return []
    if frame_count is not None and (not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0):
        return []
    inspected_values = annotation.get('inspected_frame_indices')
    inspected = {f for f in inspected_values if isinstance(f, int) and not isinstance(f, bool) and f >= 0
                 and (frame_count is None or f < frame_count)} if isinstance(inspected_values, list) else set()
    scale = (analysis_width or width) / width
    points = []
    labels = annotation.get('detection_labels')
    labels = labels if isinstance(labels, list) and annotation.get('detection_labels_verified_by_ai') is True else []
    for item in labels:
        if isinstance(item, dict) and item.get('label') == 'shuttle':
            points.append((item.get('frame_index'), item.get('x'), item.get('y')))
    contact = annotation.get('contact_px')
    if (isinstance(contact, list) and len(contact) == 2
            and (annotation.get('contact_observed') is True or annotation.get('contact_interval_observed') is True)):
        points.append((annotation.get('contact_frame_index'), *contact))
    valid = sorted(set((frame, float(x), float(y)) for frame, x, y in points
                      if isinstance(frame, int) and not isinstance(frame, bool) and frame in inspected
                      and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (x,y))
                      and 0 <= x < width and 0 <= y < height))
    groups = []
    for point in valid:
        # Separate remote objects/time windows rather than spanning a whole frame.
        if groups and point[0]-groups[-1][-1][0] <= 16 and math.hypot(point[1]-groups[-1][-1][1], point[2]-groups[-1][-1][2]) <= 96:
            groups[-1].append(point)
        else:
            groups.append([point])
    regions = []
    for group in groups[:16]:
        xs, ys = [p[1] for p in group], [p[2] for p in group]
        left, top = max(0, min(xs)-24), max(0, min(ys)-24)
        right, bottom = min(width, max(xs)+24), min(height, max(ys)+24)
        end = min(group[-1][0]+8, frame_count-1) if frame_count is not None else group[-1][0]+8
        regions.append({'frame_start': max(0, group[0][0]-8), 'frame_end': end,
                        'bbox': [round(v*scale, 3) for v in (left,top,right-left,bottom-top)]})
    return regions
