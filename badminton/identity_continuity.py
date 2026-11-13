"""Appearance-based vetoes for unverified free-flight review support.

Motion coherence does not establish which physical object produced a sequence.
The combined cue below can veto a flight-support claim after an occluder entry;
absence of that veto never confirms object identity or physical contact.
"""
from __future__ import annotations

from collections import Counter
import math
from statistics import median


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _polarity(observation):
    value = str(observation.get('appearance_polarity', '')).lower()
    if value in {'bright', 'dark'}:
        return value
    method = str(observation.get('method', '')).lower()
    if 'bright_component' in method:
        return 'bright'
    if 'dark_component' in method:
        return 'dark'
    return None


def _appearance(observations):
    radii = [o['radius'] for o in observations if _finite(o.get('radius')) and o['radius'] > 0]
    polarities = Counter(filter(None, (_polarity(o) for o in observations)))
    known = sum(polarities.values())
    dominant = max(polarities, key=polarities.get) if known else None
    return {'observation_count': len(observations), 'radius_count': len(radii),
            'median_radius_px': median(radii) if radii else None,
            'known_polarity_count': known, 'polarity_counts': dict(polarities),
            'dominant_polarity': dominant,
            'dominant_polarity_fraction': polarities[dominant] / known if known else None,
            'possible_occluder_overlap_count': sum(o.get('possible_occluder_overlap') is True for o in observations)}


def assess_identity_continuity(observations, candidate_frame_index, *, max_lookback=18, max_lookahead=6,
                               nominal_step=None):
    """Assess a bounded local association window without relabeling identity.

    After a previously clear moving approach enters a possible occluder, either
    a combined polarity/size break or an extreme sustained size collapse can
    withhold flight support. The collapse-only case additionally requires the
    event and every measured post-event sample to remain inside the occluder;
    a flight emerging clearly cannot trigger it. A transient size change or a
    blur-size increase alone is retained. Passing never confirms identity.
    """
    result = {'method': 'occluder_entry_appearance_continuity_v2',
              'flight_support_veto': False, 'flight_support_eligible': True,
              'identity_confirmed': False, 'physical_contact_confirmed': False,
              'status': 'no_combined_discontinuity_evidence', 'reason_codes': [], 'evidence': [],
              'scope': 'A local review-support veto only; passing this check does not verify shuttle identity or contact.'}
    if not _finite(candidate_frame_index):
        result['status'] = 'insufficient_appearance_evidence'
        return result
    actual = [o for o in observations if o.get('observed') is not False
              and _finite(o.get('frame_index')) and _finite(o.get('x')) and _finite(o.get('y'))]
    actual.sort(key=lambda o: o['frame_index'])
    if not actual:
        result['status'] = 'insufficient_appearance_evidence'
        return result
    candidate_index = min(range(len(actual)), key=lambda index: abs(actual[index]['frame_index'] - candidate_frame_index))
    if abs(actual[candidate_index]['frame_index'] - candidate_frame_index) > 2:
        result['status'] = 'insufficient_appearance_evidence'
        return result
    before_limit = min(64, max(6, int(max_lookback)))
    after_limit = min(12, max(3, int(max_lookahead)))
    window = actual[max(0, candidate_index - before_limit):candidate_index + after_limit + 1]
    candidate_index = next(i for i, o in enumerate(window) if o is actual[candidate_index])
    event_observation = window[candidate_index]
    event_radius = event_observation.get('radius')
    post_observations = []
    previous = event_observation
    for observation in window[candidate_index + 1:candidate_index + after_limit + 1]:
        # Later reacquisitions cannot establish a sustained local appearance
        # break. Match the event estimator's measured-segment boundary, with a
        # source-frame fallback for standalone assessments without timestamps.
        if (observation['frame_index'] - previous['frame_index'] > 2.5
                or (_finite(nominal_step) and nominal_step > 0
                    and _finite(observation.get('time')) and _finite(previous.get('time'))
                    and observation['time'] - previous['time'] > nominal_step * 2.5)):
            break
        post_observations.append(observation)
        previous = observation
    post_event = _appearance(post_observations)
    for index in range(3, candidate_index + 1):
        entry = window[index]
        if entry.get('possible_occluder_overlap') is not True or window[index - 1].get('possible_occluder_overlap') is True:
            continue
        clear = [o for o in window[max(0, index - 6):index] if o.get('possible_occluder_overlap') is False]
        if len(clear) < 3:
            continue
        before = _appearance(clear)
        # Use observations through and after the proposed event, so an entry
        # point that is still genuinely bright does not hide a later switch.
        after = _appearance(window[index:candidate_index + 1] + post_observations)
        baseline = before['median_radius_px']
        if baseline is None or before['radius_count'] < 3:
            continue
        polarity_change = (before['known_polarity_count'] >= 3 and after['known_polarity_count'] >= 3
            and before['dominant_polarity_fraction'] >= .75 and after['dominant_polarity_fraction'] >= .67
            and before['dominant_polarity'] != after['dominant_polarity'])
        sustained_overlap = after['possible_occluder_overlap_count'] >= max(3, after['observation_count'] * .6)
        event_ratio = event_radius / baseline if _finite(event_radius) and event_radius > 0 else None
        post_ratio = post_event['median_radius_px'] / baseline if post_event['radius_count'] >= 3 else None
        size_discontinuity = ((event_ratio is not None and (event_ratio <= .55 or event_ratio >= 1.8))
                              or (post_ratio is not None and (post_ratio <= .55 or post_ratio >= 1.8)))
        span = math.hypot(max(o['x'] for o in clear) - min(o['x'] for o in clear),
                          max(o['y'] for o in clear) - min(o['y'] for o in clear))
        approach_motion = span >= max(12., baseline * 4)
        appearance_veto = bool(polarity_change and sustained_overlap and size_discontinuity and approach_motion)
        # A hand/racket highlight can retain the incoming bright polarity while
        # replacing a resolved shuttle with a tiny fragment. Require a large
        # collapse at the event AND a collapsed median over at least three
        # distinct post-event samples, with no clear/unknown post sample.
        # Radius <= .35 at the event means at most ~12% of incoming component
        # area; the post median must stay at most half-radius (~25% area).
        # A median allows a noisy component without making the rule depend on
        # whether a three- or four-sample post tail happens to be available.
        # Isolated blur or centroid changes do not suffice. This is an
        # association veto, not an object classifier.
        distinct_post = list({o['frame_index']: o for o in post_observations}.values())
        post_radii = [o['radius'] for o in distinct_post if _finite(o.get('radius')) and o['radius'] > 0]
        collapsed_count = sum(radius <= baseline * .5 for radius in post_radii)
        wholly_overlapping_post = (len(distinct_post) >= 3
            and event_observation.get('possible_occluder_overlap') is True
            and all(o.get('possible_occluder_overlap') is True for o in distinct_post))
        distinct_post_ratio = median(post_radii) / baseline if len(post_radii) >= 3 else None
        sustained_size_collapse = (event_ratio is not None and event_ratio <= .35
            and len(post_radii) == len(distinct_post) and len(post_radii) >= 3
            and distinct_post_ratio is not None and distinct_post_ratio <= .5)
        collapse_veto = bool(approach_motion and sustained_overlap and wholly_overlapping_post
                             and sustained_size_collapse)
        veto = appearance_veto or collapse_veto
        result['evidence'].append({'entry_frame': entry['frame_index'], 'candidate_frame': event_observation['frame_index'],
            'before': before, 'after': after, 'post_event': post_event,
            'event_radius_ratio': event_ratio, 'post_event_radius_ratio': post_ratio,
            'polarity_change': polarity_change, 'size_discontinuity': size_discontinuity,
            'sustained_overlap': sustained_overlap, 'clear_approach_motion': approach_motion,
            'clear_approach_span_px': span, 'post_wholly_overlaps_possible_occluder': wholly_overlapping_post,
            'post_distinct_observation_count': len(distinct_post), 'post_collapsed_radius_count': collapsed_count,
            'post_collapse_sample_rule': 'contiguous_measured_post_event_observations',
            'event_radius_collapse_limit': .35, 'post_radius_collapse_limit': .5,
            'post_distinct_median_radius_ratio': distinct_post_ratio,
            'sustained_size_collapse': sustained_size_collapse, 'appearance_change_veto': appearance_veto,
            'size_collapse_veto': collapse_veto, 'veto': veto})
        if veto:
            reasons = (['OCCLUDER_ENTRY_AND_APPEARANCE_DISCONTINUITY'] if appearance_veto else [])
            if collapse_veto:
                reasons.append('OCCLUDER_ENTRY_AND_SUSTAINED_SIZE_COLLAPSE')
            result.update(flight_support_veto=True, flight_support_eligible=False,
                          status=('occluder_entry_with_sustained_size_collapse' if collapse_veto
                                  else 'occluder_entry_with_appearance_discontinuity'),
                          reason_codes=list(dict.fromkeys([*result['reason_codes'], *reasons])))
    if not result['flight_support_veto'] and all(_polarity(o) is None for o in window):
        result['status'] = 'insufficient_appearance_evidence'
    return result
