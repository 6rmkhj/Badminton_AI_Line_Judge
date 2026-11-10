from copy import deepcopy

from badminton.pair_workspace import combine_views
from badminton.review_diagnostics import pair_diagnostics, view_diagnostics
from badminton.review_refinement import review_regions


def view(label='REVIEW', status=None, digest='a'):
    return {'status': 'completed', 'direct_decision': label, 'qualified': label != 'REVIEW',
            'source_sha256': digest*64, 'review_source': 'ai_visual_review',
            'review_details': {'observation_status': status, 'focus_time_seconds': 2.7}}


def test_boundary_candidate_can_be_located_without_claiming_contact_or_binary_label():
    source = view(status='line_ambiguous')
    detail = view_diagnostics(source)
    assert detail['focus_time_seconds'] == 2.7
    assert not detail['focus_is_confirmed_contact']
    assert detail['missing_requirements'] == ['접촉과 라인 관계 불명확']
    assert source['direct_decision'] == 'REVIEW'


def test_pair_diagnosis_keeps_matching_and_contact_as_distinct_blockers():
    sources = {'A': view(status='line_ambiguous'), 'B': view(status='contact_occluded', digest='b')}
    before = deepcopy(sources)
    result = combine_views(sources)
    assert result['decision'] == 'REVIEW' and sources == before
    issues = result['review_diagnostics']['issues']
    assert issues[0]['code'] == 'SAME_EVENT_UNCONFIRMED'
    assert [issue['camera_ids'] for issue in issues[1:]] == [['A'], ['B']]
    confirmed = combine_views(sources, same_event_confirmed=True)
    assert confirmed['decision'] == 'REVIEW'
    assert all(i['code'] != 'SAME_EVENT_UNCONFIRMED' for i in confirmed['review_diagnostics']['issues'])


def test_valid_other_camera_resolves_pair_without_a_false_pair_blocker():
    result = combine_views({'A': view('IN'), 'B': view(status='contact_occluded', digest='b')},
                           same_event_confirmed=True)
    assert result['decision'] == 'IN'
    assert result['review_diagnostics']['issues'] == []
    assert result['review_diagnostics']['views']['B']['missing_requirements']


def test_fresh_diagnosis_does_not_infer_occlusion_or_truncation_from_missing_detection():
    detail = view_diagnostics({'reason_codes': ['FIRST_CONTACT_NOT_OBSERVED', 'CALIBRATION_NOT_ACCEPTED',
                                               'TARGET_SHUTTLE_IDENTITY_UNCERTAIN'],
                              'review_source': 'automatic'})
    assert detail['observation_status'] is None
    assert len(detail['missing_requirements']) == 3
    assert '접촉 전 클립 종료' not in detail['missing_requirements']


def test_clipped_contact_requires_additional_source_evidence():
    result = combine_views({'A': view(status='no_contact_in_clip'), 'B': view(status='contact_occluded', digest='b')},
                           same_event_confirmed=True)
    assert result['review_diagnostics']['issues'][0]['resolvable_with_existing_inputs'] is False


def test_refinement_only_uses_bounded_actual_shuttle_hints_not_background_or_invalid_points():
    annotation = {'analysis_size': {'width':960,'height':540}, 'detection_labels_verified_by_ai':True,
                  'inspected_frame_indices':[80,81,82], 'detection_labels': [
        {'frame_index':80,'x':719,'y':314,'label':'shuttle'},
        {'frame_index':82,'x':724,'y':311,'label':'shuttle'},
        {'frame_index':81,'x':780,'y':350,'label':'background'},
        {'frame_index':True,'x':100,'y':100,'label':'shuttle'},
        {'frame_index':81,'x':float('nan'),'y':20,'label':'shuttle'},
        {'frame_index':81,'x':1000,'y':20,'label':'shuttle'}]}
    original = deepcopy(annotation)
    hints = review_regions(annotation, 480)
    assert len(hints) == 1
    assert hints[0]['frame_start'] == 72 and hints[0]['frame_end'] == 90
    assert hints[0]['bbox'] == [347.5,143.5,26.5,25.5]
    assert annotation['detection_labels'] == original['detection_labels']
    assert review_regions({'analysis_size':annotation['analysis_size']}) == []


def test_unverified_uninspected_out_of_timeline_hints_are_ignored():
    a = {'analysis_size':{'width':960,'height':540},'inspected_frame_indices':[1,99999],
         'detection_labels_verified_by_ai':False,
         'detection_labels':[None, 'invalid', {'label':'shuttle','frame_index':1,'x':100,'y':100},
                             {'label':'shuttle','frame_index':2,'x':100,'y':100},
                             {'label':'shuttle','frame_index':99999,'x':100,'y':100}]}
    assert review_regions(a, frame_count=20) == []
    a['detection_labels_verified_by_ai'] = True
    regions = review_regions(a, frame_count=20)
    assert len(regions) == 1 and regions[0]['frame_end'] == 9
    assert review_regions(a, frame_count=1) == []
    a.update(detection_labels=[],contact_frame_index=1,contact_px=[100,100])
    assert review_regions(a) == []
