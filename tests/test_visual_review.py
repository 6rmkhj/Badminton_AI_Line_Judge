from copy import deepcopy
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from badminton.visual_review import (
    build_visual_review_report, judge_visual_geometry, validate_visual_annotation,
)


@pytest.fixture
def context(tmp_path):
    image = tmp_path / 'evidence' / 'contact.jpg'
    image.parent.mkdir()
    cv2.imwrite(str(image), np.zeros((48, 80, 3), np.uint8))
    result = {'video_id': 'v_1', 'file': 'one.mp4', 'source_sha256': hashlib.sha256(b'video').hexdigest(),
              'analysis': {'width': 100, 'height': 100},
              'metadata': {'width': 400, 'height': 400},
              'timestamps': [0., .1, .2, .3, .4], 'decision': {'decision': 'REVIEW'}, 'artifacts': {}}
    annotation = {'source': 'ai_visual_review', 'status': 'reviewed_by_ai', 'human_verified': False,
                  'video_id': 'v_1', 'file': 'one.mp4', 'source_sha256': hashlib.sha256(b'video').hexdigest(),
                  'analysis_size': {'width': 100, 'height': 100},
                  'source_size': {'width': 400, 'height': 400}, 'label': 'IN',
                  'target_contact_identity_confirmed': True, 'observation_status': 'contact_observed',
                  'reviewer': 'AI visual reviewer', 'rationale': 'The contact is visible inside all court edges.',
                  'rules': {'game': 'doubles', 'phase': 'rally'}, 'inspected_frame_indices': [0,1,2,3,4],
                  'evidence_paths': ['evidence/contact.jpg'], 'contact_observed': True,
                  'contact_frame_index': 2, 'time': .2, 'contact_px': [50, 50],
                  'boundary_evidence': {'kind': 'court_polygon', 'polygon_px': [[10,10],[90,10],[90,90],[10,90]],
                                        'polygon_region': 'full_court', 'contact_uncertainty_px': 2,
                                        'boundary_uncertainty_px': 1, 'explanation': 'Four outer painted edges reviewed.'}}
    return tmp_path, result, annotation


def test_assisted_in_keeps_ai_provenance_and_automatic_exclusion(context):
    root, result, annotation = context
    outcome = validate_visual_annotation(annotation, result, root)
    assert outcome['valid'] and outcome['qualified']
    assert outcome['assisted_decision'] == 'IN'
    assert not outcome['evaluation_eligible'] and not outcome['ground_truth_eligible']
    assert not outcome['human_verified']
    assert outcome['geometry']['uncertainty_px'] == 3
    assert result['decision']['decision'] == 'REVIEW'


@pytest.mark.parametrize('field,value,reason', [
    ('source', 'human_review', 'AI_REVIEW_PROVENANCE_REQUIRED'),
    ('human_verified', True, 'AI_REVIEW_MUST_NOT_CLAIM_HUMAN_VERIFICATION'),
    ('source_sha256', 'b'*64, 'ANNOTATION_SOURCE_HASH_MISMATCH'),
    ('analysis_size', {'width':200,'height':200}, 'ANNOTATION_ANALYSIS_SIZE_MISMATCH'),
    ('source_size', {'width':800,'height':800}, 'ANNOTATION_SOURCE_SIZE_MISMATCH'),
    ('time', .25, 'CONTACT_TIME_DOES_NOT_MATCH_SOURCE_FRAME'),
    ('contact_frame_index', True, 'CONTACT_FRAME_INVALID'),
    ('contact_px', [100,50], 'CONTACT_POINT_OUTSIDE_ANALYSIS_FRAME'),
])
def test_invalid_evidence_never_becomes_binary(context, field, value, reason):
    root, result, annotation = context
    annotation[field] = value
    outcome = validate_visual_annotation(annotation, result, root)
    assert outcome['assisted_decision'] == 'REVIEW'
    assert reason in outcome['errors']


def test_final_resting_position_is_not_contact(context):
    root, result, annotation = context
    annotation['contact_observed'] = False
    annotation['observation_status'] = 'contact_occluded'
    outcome = validate_visual_annotation(annotation, result, root)
    assert outcome['valid'] and not outcome['qualified']
    assert outcome['assisted_decision'] == 'REVIEW'
    assert outcome['suggested_label'] == 'IN'
    assert 'GROUND_CONTACT_NOT_OBSERVED' in outcome['reason_codes']


def test_frame_before_and_after_contact_must_be_visually_inspected(context):
    root, result, annotation = context
    annotation['inspected_frame_indices'] = [2, 4]
    outcome = validate_visual_annotation(annotation, result, root)
    assert not outcome['qualified']
    assert 'CONTACT_ADJACENT_FRAMES_NOT_INSPECTED' in outcome['reason_codes']


def test_interval_keeps_multiple_possible_contact_positions(context):
    root, result, annotation = context
    annotation.update(contact_observed=False, contact_interval_observed=True, observation_status='contact_interval',
                      contact_interval_frames=[1,2], contact_support_px=[[40,40],[60,60]])
    assert validate_visual_annotation(annotation, result, root)['assisted_decision'] == 'IN'
    annotation['contact_support_px'] = [[40,40],[91,60]]
    outcome = validate_visual_annotation(annotation, result, root)
    assert outcome['assisted_decision'] == 'REVIEW'
    assert outcome['geometry']['decision'] == 'REVIEW'


def test_local_outer_line_suffices_for_out_but_not_in(context):
    root, result, annotation = context
    annotation['boundary_evidence'] = {'kind':'local_boundary',
        'boundaries':[{'role':'left','line_px':[[40,10],[40,90]],
                       'inside_point_px':[70,50],'role_confirmed':True}],
        'contact_uncertainty_px':2,'boundary_uncertainty_px':1,'explanation':'Left outside painted edge.'}
    annotation.update(label='OUT',contact_px=[30,50])
    assert validate_visual_annotation(annotation,result,root)['assisted_decision'] == 'OUT'
    annotation.update(label='IN',contact_px=[50,50])
    outcome = validate_visual_annotation(annotation,result,root)
    assert outcome['assisted_decision'] == 'REVIEW'
    assert 'INSUFFICIENT_BOUNDARIES_FOR_IN' in outcome['reason_codes']


def test_uncertainty_and_support_cannot_be_replaced_by_center(context):
    root, result, annotation = context
    annotation['boundary_evidence']['contact_uncertainty_px'] = 4
    annotation['contact_px'] = [13,50]
    outcome = validate_visual_annotation(annotation,result,root)
    assert outcome['assisted_decision'] == 'REVIEW'
    annotation.update(label='OUT',contact_support_px=[[5,50],[15,50]])
    assert validate_visual_annotation(annotation,result,root)['assisted_decision'] == 'REVIEW'


def test_half_court_never_makes_net_an_out_boundary(context):
    root, result, annotation = context
    geometry = annotation['boundary_evidence']
    geometry.update(polygon_region='near_half', target_contact_half='near',
                    edge_roles=['net','right','near','left'])
    annotation.update(label='OUT',contact_px=[50,5])
    assert validate_visual_annotation(annotation,result,root)['assisted_decision'] == 'REVIEW'
    annotation['contact_px'] = [5,50]
    assert validate_visual_annotation(annotation,result,root)['assisted_decision'] == 'OUT'
    geometry['target_contact_half'] = 'far'
    assert validate_visual_annotation(annotation,result,root)['assisted_decision'] == 'REVIEW'


def test_geometry_does_not_accept_nonconvex_polygon_or_airborne_point(context):
    root, result, annotation = context
    annotation['boundary_evidence']['polygon_px'] = [[10,10],[90,10],[30,30],[90,90],[10,90]]
    outcome = validate_visual_annotation(annotation,result,root)
    assert outcome['assisted_decision'] == 'REVIEW'
    assert 'BOUNDARY_GEOMETRY_INVALID' in outcome['reason_codes']


def test_label_geometry_disagreement_is_visible(context):
    root,result,annotation=context
    annotation['label']='OUT'
    outcome=validate_visual_annotation(annotation,result,root)
    assert outcome['assisted_decision']=='REVIEW'
    assert 'VISUAL_LABEL_DISAGREES_WITH_COMPUTED_GEOMETRY' in outcome['reason_codes']


@pytest.mark.parametrize('path', ['../contact.jpg','C:\\secret.jpg','/etc/passwd','evidence/missing.jpg'])
def test_evidence_paths_are_confined(context,path):
    root,result,annotation=context
    annotation['evidence_paths']=[path]
    outcome=validate_visual_annotation(annotation,result,root)
    assert outcome['assisted_decision']=='REVIEW'
    assert 'VISUAL_EVIDENCE_PATH_OR_IMAGE_INVALID' in outcome['errors']


def test_evidence_symlink_escape_is_rejected(context,tmp_path_factory):
    root,result,annotation=context
    outside=tmp_path_factory.mktemp('outside')/'image.jpg'
    cv2.imwrite(str(outside),np.zeros((5,5,3),np.uint8))
    (root/'evidence/link.jpg').symlink_to(outside)
    annotation['evidence_paths']=['evidence/link.jpg']
    assert not validate_visual_annotation(annotation,result,root)['valid']


def _write_run(root,results,annotations):
    run,labels,out=root/'run',root/'labels',root/'review'
    run.mkdir();labels.mkdir()
    (root/'video').mkdir(exist_ok=True)
    entries=[]
    for result,annotation in zip(results,annotations):
        identifier=result['video_id']
        (root/'video'/result['file']).write_bytes(b'video')
        path=run/'videos'/identifier/'result.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(result))
        (labels/f'{identifier}.json').write_text(json.dumps(annotation))
        entries.append({'id':identifier,'file':result['file'],'result_path':f'videos/{identifier}/result.json'})
    (run/'index.json').write_text(json.dumps({'videos':entries}))
    return run,labels,out


def test_portable_report_escapes_markup_preserves_original_and_counts_review(context):
    root,result,annotation=context
    annotation['rationale']='<script>alert(1)</script>'
    run,labels,out=_write_run(root,[result],[annotation])
    before=(run/'videos/v_1/result.json').read_bytes()
    report=build_visual_review_report(run,labels,out,project_root=root)
    assert report['assisted_counts']=={'IN':1,'OUT':0,'REVIEW':0}
    assert report['automatic_counts']=={'REVIEW':1}
    assert report['automatic_accuracy'] is None
    assert (run/'videos/v_1/result.json').read_bytes()==before
    assert (out/'assets/v_1/evidence-000.jpg').exists()
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in (out/'index.html').read_text()
    assert (out/'visual-review.json').exists()


def test_conflicting_duplicate_content_is_not_two_independent_labels(context):
    root,result,annotation=context
    result2,annotation2=deepcopy(result),deepcopy(annotation)
    result2.update(video_id='v_2',file='two.mp4')
    annotation2.update(video_id='v_2',file='two.mp4',label='OUT',contact_px=[5,50])
    run,labels,out=_write_run(root,[result,result2],[annotation,annotation2])
    report=build_visual_review_report(run,labels,out,project_root=root)
    assert report['assisted_counts']=={'IN':0,'OUT':0,'REVIEW':2}
    assert len(report['duplicate_conflicts'])==1
    assert all('DUPLICATE_CONTENT_LABEL_CONFLICT' in row['review']['reason_codes'] for row in report['videos'])


def test_output_cannot_mutate_analysis_run(context):
    root,result,annotation=context
    run,labels,out=_write_run(root,[result],[annotation])
    with pytest.raises(ValueError,match='separate'):
        build_visual_review_report(run,labels,run/'review',project_root=root)


def test_visible_post_rally_toss_does_not_identify_target_contact(context):
    root,result,annotation=context
    annotation['target_contact_identity_confirmed']=False
    outcome=validate_visual_annotation(annotation,result,root)
    assert outcome['assisted_decision']=='REVIEW'
    assert 'TARGET_CONTACT_IDENTITY_NOT_CONFIRMED' in outcome['reason_codes']


def test_changed_original_bytes_invalidate_even_matching_annotation_hash(context):
    root,result,annotation=context
    run,labels,out=_write_run(root,[result],[annotation])
    (root/'video'/result['file']).write_bytes(b'changed')
    report=build_visual_review_report(run,labels,out,project_root=root)
    assert report['source_hash_verified_count']==0
    assert report['assisted_counts']['REVIEW']==1
    assert 'ORIGINAL_SOURCE_CHANGED_SINCE_ANALYSIS' in report['videos'][0]['review']['errors']


def _cross_view_context(context, donor_label='IN'):
    root,result,annotation=context
    donor,donor_annotation=deepcopy(result),deepcopy(annotation)
    donor.update(video_id='v_2',file='two.mp4',source_sha256=hashlib.sha256(b'video2').hexdigest())
    donor_annotation.update(video_id='v_2',file='two.mp4',source_sha256=donor['source_sha256'],label=donor_label)
    if donor_label=='OUT':
        donor_annotation['contact_px']=[5,50]
    annotation.update(label='UNKNOWN',contact_observed=False,observation_status='contact_occluded',
                      target_contact_identity_confirmed=False)
    annotation['cross_view_match']={'other_video_id':'v_2','same_event_confirmed_by_ai':True,
        'matched_actions':[{'frame_index':0,'other_frame_index':0,'description':'Receiver prepares for the shot.'},
                           {'frame_index':1,'other_frame_index':1,'description':'Racket swing starts.'}],
        'rationale':'The same players and two distinct actions are visible.',
        'evidence_paths':['evidence/contact.jpg']}
    run,labels,out=_write_run(root,[result,donor],[annotation,donor_annotation])
    (root/'video/two.mp4').write_bytes(b'video2')
    return root,run,labels,out,annotation,donor_annotation


def test_cross_view_transfer_keeps_donor_coordinates_and_local_unknown(context):
    root,run,labels,out,annotation,donor=_cross_view_context(context)
    report=build_visual_review_report(run,labels,out,project_root=root)
    receiver=report['videos'][0]
    assert receiver['assisted_decision']=='IN'
    assert receiver['direct_assisted_decision']=='REVIEW'
    assert receiver['assistance_mode']=='cross_view_ai_review'
    assert receiver['originating_video_id']=='v_2'
    assert receiver['donor_contact']['coordinate_view']=='v_2'
    assert receiver['annotation']['contact_observed'] is False
    assert receiver['annotation']['label']=='UNKNOWN'
    assert report['cross_view_transferred_count']==1
    assert report['automatic_accuracy'] is None
    assert not receiver['evaluation_eligible']


@pytest.mark.parametrize('mutation,reason', [
    ('uninspected','CROSS_VIEW_ACTION_NOT_IN_INSPECTED_SOURCE_FRAMES'),
    ('one_action','CROSS_VIEW_NEEDS_TWO_DISTINCT_REVIEWED_ACTIONS'),
    ('not_confirmed','CROSS_VIEW_MATCH_NOT_EXPLICITLY_REVIEWED'),
    ('reversed','CROSS_VIEW_ACTIONS_NOT_DISTINCT_AND_TEMPORALLY_ORDERED'),
])
def test_cross_view_requires_actual_reviewed_event_correspondence(context,mutation,reason):
    root,run,labels,out,annotation,donor=_cross_view_context(context)
    match=annotation['cross_view_match']
    if mutation=='uninspected': annotation['inspected_frame_indices']=[2,3,4]
    if mutation=='one_action': match['matched_actions']=match['matched_actions'][:1]
    if mutation=='not_confirmed': match['same_event_confirmed_by_ai']=False
    if mutation=='reversed':
        match['matched_actions'][0]['other_frame_index']=1
        match['matched_actions'][1]['other_frame_index']=0
    (labels/'v_1.json').write_text(json.dumps(annotation))
    report=build_visual_review_report(run,labels,out,project_root=root)
    receiver=report['videos'][0]
    assert receiver['assisted_decision']=='REVIEW'
    assert reason in receiver['cross_view_review']['reason_codes']


def test_cross_view_cannot_bootstrap_two_unknowns_through_cycle(context):
    root,run,labels,out,annotation,donor=_cross_view_context(context)
    donor.update(label='UNKNOWN',contact_observed=False,observation_status='contact_occluded')
    donor['cross_view_match']=deepcopy(annotation['cross_view_match'])
    donor['cross_view_match']['other_video_id']='v_1'
    (labels/'v_2.json').write_text(json.dumps(donor))
    report=build_visual_review_report(run,labels,out,project_root=root)
    assert report['assisted_counts']['REVIEW']==2
    assert report['cross_view_transferred_count']==0


def test_cross_view_conflicting_direct_contacts_veto_both(context):
    root,run,labels,out,annotation,donor=_cross_view_context(context,donor_label='OUT')
    annotation.update(label='IN',contact_observed=True,observation_status='contact_observed',
                      target_contact_identity_confirmed=True)
    (labels/'v_1.json').write_text(json.dumps(annotation))
    report=build_visual_review_report(run,labels,out,project_root=root)
    assert report['assisted_counts']['REVIEW']==2
    assert all('CROSS_VIEW_QUALIFIED_CONTACTS_CONFLICT' in r['review']['reason_codes'] for r in report['videos'])


def test_malformed_review_fields_are_reported_without_crashing(context):
    root,result,annotation=context
    annotation.update(label=['IN'], suggested_label={'bad':'value'}, observation_status=['contact_observed'],
                      rules={'game':['doubles'],'phase':'rally'})
    outcome=validate_visual_annotation(annotation,result,root)
    assert outcome['assisted_decision']=='REVIEW'
    assert not outcome['valid']
    assert outcome['suggested_label'] is None
    run,labels,out=_write_run(root,[result],[annotation])
    report=build_visual_review_report(run,labels,out,project_root=root)
    assert report['observation_status_counts']=={'missing':1}
    assert (out/'index.html').exists()


def test_missing_or_non_object_annotation_keeps_video_in_report(context):
    root,result,annotation=context
    run,labels,out=_write_run(root,[result],[annotation])
    (labels/'v_1.json').write_text('[1, 2, 3]')
    report=build_visual_review_report(run,labels,out,project_root=root)
    assert report['video_count']==1 and report['assisted_counts']['REVIEW']==1
    assert 'ANNOTATION_NOT_AN_OBJECT' in report['videos'][0]['review']['errors']
    (labels/'v_1.json').unlink()
    report=build_visual_review_report(run,labels,out,project_root=root)
    assert report['reviewed_count']==0
    assert 'ANNOTATION_MISSING' in report['videos'][0]['review']['errors']


def test_uninformative_reciprocal_view_does_not_erase_direct_observation(context):
    root,run,labels,out,annotation,donor=_cross_view_context(context)
    donor['cross_view_match']=deepcopy(annotation['cross_view_match'])
    donor['cross_view_match']['other_video_id']='v_1'
    (labels/'v_2.json').write_text(json.dumps(donor))
    report=build_visual_review_report(run,labels,out,project_root=root)
    assert report['assisted_counts']=={'IN':2,'OUT':0,'REVIEW':0}
    assert report['videos'][0]['assistance_mode']=='cross_view_ai_review'
    assert report['videos'][1]['direct_assisted_decision']=='IN'
    assert report['videos'][1]['review']['qualified'] is True
    assert report['videos'][1]['cross_view_review']['optional_correspondence_ignored'] is True


def test_per_edge_uncertainty_preserves_local_evidence_without_dropping_far_error(context):
    root,result,annotation=context
    annotation['contact_px']=[14,50]
    geometry=annotation['boundary_evidence']
    geometry.update(contact_uncertainty_px=1,boundary_uncertainty_px=8)
    assert validate_visual_annotation(annotation,result,root)['assisted_decision']=='REVIEW'
    geometry['boundary_uncertainty_px_by_edge']=[8,8,8,1]
    outcome=validate_visual_annotation(annotation,result,root)
    assert outcome['assisted_decision']=='IN'
    boundaries=outcome['geometry']['boundaries']
    assert [row['uncertainty_px'] for row in boundaries]==[9,9,9,2]
    assert outcome['geometry']['uncertainty_px']==9
    geometry['boundary_uncertainty_px_by_edge']=[1,1,1,-1]
    assert validate_visual_annotation(annotation,result,root)['assisted_decision']=='REVIEW'
    geometry['boundary_uncertainty_px_by_edge']=[1]
    assert validate_visual_annotation(annotation,result,root)['assisted_decision']=='REVIEW'


def test_local_boundary_uncertainty_override_applies_only_to_its_edge(context):
    root,result,annotation=context
    annotation.update(label='OUT',contact_px=[36,50])
    annotation['boundary_evidence']={'kind':'local_boundary',
        'boundaries':[{'role':'left','line_px':[[40,10],[40,90]],'inside_point_px':[70,50],
                       'role_confirmed':True,'boundary_uncertainty_px':1}],
        'contact_uncertainty_px':1,'boundary_uncertainty_px':8,'explanation':'Nearby left edge visible precisely.'}
    outcome=validate_visual_annotation(annotation,result,root)
    assert outcome['assisted_decision']=='OUT'
    assert outcome['geometry']['boundaries'][0]['uncertainty_px']==2


def test_reviewed_court_interior_only_proves_robust_in(context):
    root,result,annotation=context
    geometry=annotation['boundary_evidence']
    geometry.update(kind='court_interior',interior_confirmed=True)
    geometry.pop('polygon_region')
    outcome=validate_visual_annotation(annotation,result,root)
    assert outcome['assisted_decision']=='IN'
    assert all(row['role']=='interior' and not row['is_outer_boundary'] for row in outcome['geometry']['boundaries'])
    annotation.update(label='OUT',contact_px=[5,50])
    assert validate_visual_annotation(annotation,result,root)['assisted_decision']=='REVIEW'
    annotation.update(label='IN',contact_px=[12,50])
    assert validate_visual_annotation(annotation,result,root)['assisted_decision']=='REVIEW'
    annotation['contact_px']=[50,50]
    geometry['interior_confirmed']=False
    assert validate_visual_annotation(annotation,result,root)['assisted_decision']=='REVIEW'


def _expand_cross_view_timeline(run, labels, annotation, donor_annotation):
    timestamps=[index*.05 for index in range(20)]
    for identifier, record in [('v_1',annotation),('v_2',donor_annotation)]:
        path=run/'videos'/identifier/'result.json'
        result=json.loads(path.read_text());result['timestamps']=timestamps
        path.write_text(json.dumps(result))
        record.update(inspected_frame_indices=list(range(20)),time=.1)
        (labels/f'{identifier}.json').write_text(json.dumps(record))


def test_cross_view_rejects_inconsistent_action_phase_offsets(context):
    root,run,labels,out,annotation,donor=_cross_view_context(context)
    _expand_cross_view_timeline(run,labels,annotation,donor)
    annotation['cross_view_match']['matched_actions'][1]['other_frame_index']=11
    (labels/'v_1.json').write_text(json.dumps(annotation))
    report=build_visual_review_report(run,labels,out,project_root=root)
    receiver=report['videos'][0]
    assert receiver['assisted_decision']=='REVIEW'
    assert 'CROSS_VIEW_ACTION_TIMING_INCONSISTENT' in receiver['cross_view_review']['reason_codes']
    check=receiver['cross_view_review']['action_timing_check']
    assert check['offset_spread_seconds']==pytest.approx(.5)
    assert check['allowed_spread_seconds']==pytest.approx(.2)
    assert not check['clock_synchronization_claimed']
    assert report['videos'][1]['assisted_decision']=='IN'


def test_cross_view_accepts_unknown_constant_offset_without_claiming_clock_sync(context):
    root,run,labels,out,annotation,donor=_cross_view_context(context)
    _expand_cross_view_timeline(run,labels,annotation,donor)
    annotation['cross_view_match']['matched_actions'][0]['other_frame_index']=10
    annotation['cross_view_match']['matched_actions'][1]['other_frame_index']=11
    (labels/'v_1.json').write_text(json.dumps(annotation))
    report=build_visual_review_report(run,labels,out,project_root=root)
    receiver=report['videos'][0]
    assert receiver['assisted_decision']=='IN'
    check=receiver['cross_view_review']['action_timing_check']
    assert check['action_offsets_seconds']==pytest.approx([.5,.5])
    assert check['offset_spread_seconds']==pytest.approx(0)
    assert check['clock_synchronization_claimed'] is False


def _pair_record(identifier, filename, decision, digest=None, annotation=None):
    return {'video_id':identifier,'file':filename,'assisted_decision':decision,
            'direct_assisted_decision':decision,'source_sha256':digest or identifier,
            'annotation':annotation or {}}


def test_filename_pair_audit_flags_missing_transfer_candidate_without_propagation():
    from badminton.visual_review import audit_filename_pairs
    records=[_pair_record('a1','a시점 1.mp4','REVIEW'),_pair_record('b1','b시점 1.mp4','IN')]
    original=deepcopy(records)
    audit=audit_filename_pairs(records)
    assert audit['counts']['one_review_with_decided_partner']==1
    item=audit['items'][0]
    assert item['review_video_id']=='a1' and item['donor_candidate']['video_id']=='b1'
    assert item['receiver_correspondence']['status']=='missing'
    assert not item['event_correspondence_validated']
    assert audit['filename_only_propagation'] is False
    assert records==original


def test_filename_pair_audit_counts_both_review_and_conflicting_labels():
    from badminton.visual_review import audit_filename_pairs
    records=[_pair_record('a1','a시점 1.mp4','REVIEW'),_pair_record('b1','b시점 1.mp4','REVIEW'),
             _pair_record('a2','a시점 2.mp4','IN'),_pair_record('b2','b시점 2.mp4','OUT'),
             _pair_record('a3','a시점 3.mp4','OUT'),_pair_record('b3','b시점 3.mp4','OUT')]
    audit=audit_filename_pairs(records)
    assert audit['counts']=={'both_resolved_same_label':1,'both_review':1,
                             'one_review_with_decided_partner':0,'conflicting_resolved_labels':1}
    assert audit['both_review_video_count']==2
    assert audit['event_correspondence_validated_pair_count']==0
    assert records[2]['assisted_decision']=='IN' and records[3]['assisted_decision']=='OUT'


def test_duplicate_content_across_filename_pairs_does_not_override_reviewed_mismatch():
    from badminton.visual_review import audit_filename_pairs
    mismatch={'cross_view_candidates':[{'other_video_id':'b18','status':'different_rally',
        'same_event_confirmed_by_ai':False,'rationale':'B18 repeats B17; A18 has a different shot sequence.'}]}
    records=[_pair_record('a17','a시점 17.mp4','IN'),_pair_record('b17','b시점 17.mp4','IN','duplicate'),
             _pair_record('a18','a시점 18.mp4','REVIEW',annotation=mismatch),
             _pair_record('b18','b시점 18.mp4','IN','duplicate')]
    original=deepcopy(records);audit=audit_filename_pairs(records)
    assert audit['duplicate_content_across_pair_count']==2
    item=next(pair for pair in audit['items'] if pair['review_video_id']=='a18')
    assert item['duplicate_content_across_pairs']
    assert item['receiver_correspondence']['status']=='rejected_different_event'
    assert item['cross_pair_duplicates'][0]['duplicate_video_id']=='b17'
    assert not item['event_correspondence_validated']
    assert records==original


def test_pair_audit_acceptance_requires_validated_action_correspondence():
    from badminton.visual_review import audit_filename_pairs
    a=_pair_record('a1','a시점 1.mp4','IN',annotation={'cross_view_match':{
        'other_video_id':'b1','same_event_confirmed_by_ai':True,'rationale':'Two reviewed actions.'}})
    b=_pair_record('b1','b시점 1.mp4','IN')
    audit=audit_filename_pairs([a,b])
    assert not audit['items'][0]['event_correspondence_validated']
    a['cross_view_review']={'qualified':False,'originating_video_id':'b1',
                            'reason_codes':['CROSS_VIEW_ACTION_TIMING_INCONSISTENT']}
    assert audit_filename_pairs([a,b])['items'][0]['correspondence'][0]['status']=='rejected'
    a['cross_view_review'].update(qualified=True,reason_codes=[])
    assert audit_filename_pairs([a,b])['event_correspondence_validated_pair_count']==1


def test_pair_audit_keeps_different_directories_and_unpaired_files_separate():
    from badminton.visual_review import audit_filename_pairs
    records=[_pair_record('a1','day1/a시점 1.mp4','IN'),_pair_record('b1','day1/b시점 1.mp4','IN'),
             _pair_record('a2','day2/a시점 1.mp4','REVIEW'),_pair_record('b2','day2/b시점 1.mp4','REVIEW'),
             _pair_record('unknown','unpaired.mp4','REVIEW')]
    audit=audit_filename_pairs(records)
    assert audit['candidate_pair_count']==2
    assert audit['unpaired_video_ids']==['unknown']


def test_report_prominently_exposes_missing_opposite_view_review(context):
    root,result,annotation=context
    result.update(file='a시점 1.mp4')
    annotation.update(file='a시점 1.mp4',label='UNKNOWN',contact_observed=False,
                      observation_status='contact_out_of_view',target_contact_identity_confirmed=False)
    donor,donor_annotation=deepcopy(result),deepcopy(annotation)
    donor.update(video_id='v_2',file='b시점 1.mp4',source_sha256=hashlib.sha256(b'video2').hexdigest())
    donor_annotation.update(video_id='v_2',file='b시점 1.mp4',source_sha256=donor['source_sha256'],
                            label='IN',contact_observed=True,observation_status='contact_observed',
                            target_contact_identity_confirmed=True)
    run,labels,out=_write_run(root,[result,donor],[annotation,donor_annotation])
    (root/'video/b시점 1.mp4').write_bytes(b'video2')
    report=build_visual_review_report(run,labels,out,project_root=root)
    assert report['assisted_counts']=={'IN':1,'OUT':0,'REVIEW':1}
    assert report['filename_pair_audit']['counts']['one_review_with_decided_partner']==1
    text=(out/'index.html').read_text()
    assert 'id="filename-pair-audit"' in text and 'data-mixed-review="v_1"' in text
    assert '사건 대응 검수 누락' in text and '같은 번호는 같은 사건의 증거가 아닙니다' in text
