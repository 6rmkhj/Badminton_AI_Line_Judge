"""Protect frozen denominators, coordinate/timing metrics and input provenance."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('landing_engine_evaluator', Path(__file__).parents[1] / 'scripts/evaluate_landing_engine.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def reference(**extra):
    return dict(contact_visibility='visible', coordinate_size=[960, 540],
                landing_frame_interval=[100, 102], landing_point_px=[200, 300], tolerance_px=6, **extra)


def candidate(frame=101, xy=None, kind='observed_object_center'):
    return {'frame_index':frame, 'candidate_px':xy or [400, 600], 'track_id':1,
            'image_motion_event':{'point_kind':kind}, 'reason_codes':['CONTACT_UNOBSERVED']}


def result(selected=None, candidates=None):
    selected = candidate() if selected is None else selected
    return {'source_sha256':'source', 'analysis':{'width':1920, 'height':1080},
            'selected_landing':selected, 'landings':candidates or [selected],
            'decision':{'decision':'REVIEW', 'reason_codes':['CONTACT_UNOBSERVED','CONTACT_UNOBSERVED','CALIBRATION_UNCONFIRMED']}}


def row(ref=None, key='case'):
    return {'key':key, 'cohort':'test', 'group_id':'pair', 'source_sha256':'source',
            'reference_sha256':'frozen', 'reference':reference() if ref is None else ref}


def test_coordinate_scaling_and_exact_tolerance_boundary():
    comparison = evaluator.compare(candidate(102, [412,600]), result(), reference())
    assert comparison['distance_px'] == 6
    assert comparison['frame_error'] == 0
    assert comparison['matches_first_vicinity']
    assert not evaluator.compare(candidate(103), result(), reference())['matches_first_vicinity']
    assert not evaluator.compare(candidate(102, [412.001,600]), result(), reference())['matches_first_vicinity']


def test_provided_extra_frame_tolerance_only():
    assert evaluator.compare(candidate(103), result(), reference(frame_tolerance=1))['matches_first_vicinity']
    with pytest.raises(ValueError, match='tolerance'):
        evaluator.compare(candidate(), result(), reference(frame_tolerance=-1))


@pytest.mark.parametrize('rank,at6,at12', [(1,True,True),(6,True,True),(7,False,True),(12,False,True),(13,False,False)])
def test_recall_ranks_preserve_candidate_budget(rank, at6, at12):
    candidates = [candidate(90) for _ in range(13)]
    candidates[rank-1] = candidate()
    record = evaluator.evaluate_case(row(), result(candidate(90), candidates))
    assert record['top1_match'] is False
    assert record['recall_at_6'] is at6
    assert record['recall_at_12'] is at12
    assert record['matching_candidate_ranks'] == [rank]


def test_recall_uses_saved_final_presentation_order():
    data = result(candidate(90), [candidate()])
    data['benchmark_ranked_landings'] = [candidate(90)] * 6 + [candidate()]
    record = evaluator.evaluate_case(row(), data)
    assert record['recall_at_6'] is False
    assert record['recall_at_12'] is True


def test_unscorable_and_abstentions_keep_different_denominators():
    ref = reference(); ref.update(contact_visibility='occluded', landing_point_px=None, landing_frame_interval=None)
    excluded = evaluator.evaluate_case(row(ref), result())
    missing = result(); missing.update(selected_landing=None, landings=[])
    abstained = evaluator.evaluate_case(row(key='missing'), missing)
    summary = evaluator.summarize([excluded, abstained])
    assert summary['video_count'] == 2
    assert summary['scorable_count'] == 1
    assert summary['excluded_count'] == 1
    assert summary['top1_fraction'] == 0
    assert summary['candidate_abstention_count'] == 1
    assert summary['median_center_error_px'] is None
    assert summary['review_reason_counts'] == {'CALIBRATION_UNCONFIRMED':2,'CONTACT_UNOBSERVED':2}
    assert summary['independent_contact_accuracy'] is None
    assert summary['independent_line_judgment_accuracy'] is None


def test_matching_inferred_center_does_not_become_observed_contact():
    data = result(candidate(kind='inferred_object_center'))
    record = evaluator.evaluate_case(row(), data)
    summary = evaluator.summarize([record])
    assert summary['top1_match_count'] == 1
    assert summary['matched_candidate_kind_counts'] == {'inferred_object_center':1}
    assert summary['physical_contact_observed_count'] == 0


def test_invalid_prediction_geometry_is_miss_not_excluded():
    data = result(); data['analysis']['width'] = 0
    record = evaluator.evaluate_case(row(), data)
    assert record['localization_scorable'] is True
    assert record['selection_category'] == 'prediction_geometry_unavailable'
    assert record['top1_match'] is False


@pytest.mark.parametrize('update', [{'source_sha256':'wrong'}, {'assistance_used':True}, {'automatic_evaluation_eligible':False}])
def test_unusable_result_rejected(update):
    data = result(); data.update(update)
    with pytest.raises(ValueError):
        evaluator.evaluate_case(row(), data)


def test_reference_and_source_hashes_required(tmp_path):
    video = tmp_path/'clip.mp4'; video.write_bytes(b'original')
    ref = reference(source_sha256=evaluator.sha(video))
    refpath = tmp_path/'ref.json'; refpath.write_text(json.dumps(ref))
    case = {'key':'one','file':'clip.mp4','source_sha256':evaluator.sha(video),
            'reference_path':'ref.json','reference_sha256':evaluator.sha(refpath)}
    protocol = {'videos':[case]}
    assert len(evaluator.validate_inputs(protocol,tmp_path,tmp_path)) == 1
    refpath.write_text(json.dumps({**ref,'tolerance_px':999}))
    with pytest.raises(ValueError, match='reference SHA'):
        evaluator.validate_inputs(protocol,tmp_path,tmp_path)
    refpath.write_text(json.dumps(ref)); video.write_bytes(b'altered')
    with pytest.raises(ValueError, match='Source missing or SHA'):
        evaluator.validate_inputs(protocol,tmp_path,tmp_path)


def test_unknown_key_rejected_and_unassisted_only():
    with pytest.raises(ValueError, match='Unknown keys'):
        evaluator.validate_inputs({'videos':[]},Path('.'),Path('.'),['missing'])
    for cfg in [{'events':{'event_window_seconds':[1,2]}}, {'detector':{'refinement_regions':[{}]}}, {'identity':{'hit_anchors':[{}]}}]:
        with pytest.raises(ValueError, match='unassisted'):
            evaluator.unassisted(cfg)
    evaluator.unassisted({'events':{'max_candidates':12}})
