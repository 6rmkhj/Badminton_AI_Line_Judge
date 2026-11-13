"""Pixel-support safeguards; synthetic images are not shuttle accuracy labels."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from badminton.observation_support import (assess_observation_support, rejects_moving_association,
                                           observation_support_status, annotate_observations, remap_native_support)


def texture():
    rng = np.random.default_rng(42)
    gray = cv2.GaussianBlur(rng.integers(35, 145, (128, 128), dtype=np.uint8), (3, 3), 0)
    return np.repeat(gray[:, :, None], 3, axis=2)


def observation(**extra):
    return {'x':64.,'y':64.,'bbox':[61,61,7,7],'appearance_polarity':'bright',
            'frame_index':4,'observed':True,**extra}


def static_frames():
    frame=texture()
    cv2.circle(frame,(64,64),2,(235,235,235),-1)
    return [frame.copy() for _ in range(9)]


def test_persistent_feature_is_only_negative_moving_evidence_and_does_not_mutate():
    frames=static_frames();before=np.stack(frames);candidate=observation();original=deepcopy(candidate)
    evidence=assess_observation_support(frames,4,candidate)
    assert evidence['status']=='persistent_background' and evidence['evidence_valid']
    assert evidence['reject_for_moving_association']
    assert evidence['window_frame_indices']==[0,2,3,4,5,6,8]
    assert evidence['point_px']==[64,64] and evidence['coordinate_size']==[128,128]
    assert not evidence['confirms_identity'] and not evidence['confirms_contact']
    assert candidate==original and np.array_equal(np.stack(frames),before)
    assert rejects_moving_association({**candidate,'observation_support':evidence})


def test_moving_blurred_shuttle_is_preserved():
    base=texture();frames=[]
    for index in range(9):
        frame=base.copy();x=64+(index-4)*7
        cv2.line(frame,(x-6,64),(x+5,64),(220,220,220),2)
        frames.append(frame)
    evidence=assess_observation_support(frames,4,observation(bbox=[57,62,14,5]))
    assert evidence['status']=='transient' and evidence['evidence_valid']
    assert not evidence['reject_for_moving_association']


def test_static_background_with_small_camera_translation_is_registered():
    source=static_frames()[0]
    frames=[cv2.warpAffine(source,np.array([[1,0,(i-4)*.5],[0,1,(i-4)*.25]],np.float32),
                          (128,128),borderMode=cv2.BORDER_REFLECT) for i in range(9)]
    evidence=assess_observation_support(frames,4,observation())
    assert evidence['status']=='persistent_background'
    assert any(abs(row['translation_px'][0])>.2 for row in evidence['alignment'])


def test_moving_blur_with_camera_translation_remains_transient():
    base=texture();frames=[]
    for index in range(9):
        frame=base.copy();x=64+(index-4)*6
        cv2.line(frame,(x-5,64),(x+4,64),(230,230,230),2)
        frames.append(cv2.warpAffine(frame,np.array([[1,0,(index-4)*.4],[0,1,0]],np.float32),
                                     (128,128),borderMode=cv2.BORDER_REFLECT))
    evidence=assess_observation_support(frames,4,observation(bbox=[58,62,12,5]))
    assert evidence['status']=='transient' and not evidence['reject_for_moving_association']


def test_transient_racket_or_hand_is_not_identity_or_contact_proof():
    frames=static_frames()
    frames[4][54:73,54:73]=245
    evidence=assess_observation_support(frames,4,observation(bbox=[54,54,19,19]))
    assert evidence['status']!='persistent_background'
    assert not evidence['confirms_identity'] and not evidence['confirms_contact']


@pytest.mark.parametrize('change',[{'observed':False},{'interpolated':True},{'bbox':[64,64,.25,.5]}])
def test_unmeasured_or_subpixel_appearance_is_neutral(change):
    evidence=assess_observation_support(static_frames(),4,observation(**change))
    assert evidence['status']=='unknown' and not evidence['evidence_valid']
    assert not evidence['reject_for_moving_association']


def test_sparse_actual_indices_do_not_invent_missing_temporal_images():
    frames=static_frames()
    evidence=assess_observation_support([frames[0],frames[4],frames[8]],4,observation(),frame_indices=[0,4,8])
    assert evidence['status']=='unknown' and evidence['window_frame_indices']==[0,4,8]
    with pytest.raises(ValueError):
        assess_observation_support(frames[:3],4,observation(),frame_indices=[0,4,4])


def test_sparse_native_window_can_use_actual_plus_minus_three_frames():
    frames=static_frames();indices=[1,3,4,5,7]
    evidence=assess_observation_support([frames[i] for i in indices],4,observation(),frame_indices=indices)
    assert evidence['window_frame_indices']==indices and evidence['status']=='persistent_background'


def test_unreliable_temporal_background_is_unknown():
    rng=np.random.default_rng(8)
    frames=[rng.integers(0,256,(128,128,3),dtype=np.uint8) for _ in range(9)]
    cv2.circle(frames[4],(64,64),2,(255,255,255),-1)
    evidence=assess_observation_support(frames,4,observation())
    assert evidence['status']=='unknown' and not evidence['evidence_valid']


def test_prealigned_frames_skip_registration_but_still_check_background(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Registration must be amortized by the caller')
    monkeypatch.setattr(cv2,'findTransformECC',forbidden)
    evidence=assess_observation_support(static_frames(),4,observation(),config={'prealigned':True})
    assert evidence['status']=='persistent_background'
    assert all(row['method']=='caller_registered_residual_check' for row in evidence['alignment'])


@pytest.mark.parametrize('field,value',[('x',65),('frame_index',5),('coordinate_size',[256,256]),('observed',False)])
def test_negative_evidence_cannot_be_reused_after_candidate_changes(field,value):
    candidate=observation()
    evidence=assess_observation_support(static_frames(),4,candidate)
    assert not rejects_moving_association({**candidate,'observation_support':evidence,field:value})


def test_unknown_or_incomplete_negative_metadata_is_neutral():
    for evidence in ({'status':'persistent_background'},
                     {'schema_version':1,'status':'unknown','evidence_valid':True,'reject_for_moving_association':True}):
        assert not rejects_moving_association({**observation(),'observation_support':evidence})


def test_broad_background_support_cannot_reject_a_model_localization():
    frame=texture();frame[61:67,:, :]=15
    evidence=assess_observation_support([frame.copy() for _ in range(9)],4,
        observation(bbox=[60,61,9,6],appearance_polarity='dark'))
    assert evidence['evidence_valid'] and evidence['persistence_fraction']==1
    assert evidence['status']=='unknown' and not evidence['reject_for_moving_association']
    assert 'PERSISTENT_BROAD_STRUCTURE_AMBIGUOUS_LOCALIZATION' in evidence['reason_codes']


def test_transient_reader_and_malformed_geometry_are_neutral_when_stale():
    frames=static_frames();frames[4][62:67,62:67]=250
    candidate=observation();evidence=assess_observation_support(frames,4,candidate)
    evidence.update(status='transient',evidence_valid=True,reject_for_moving_association=False)
    assert observation_support_status({**candidate,'observation_support':evidence})=='transient'
    assert observation_support_status({**candidate,'x':65,'observation_support':evidence})=='unknown'
    assert observation_support_status({**candidate,'coordinate_size':123,'observation_support':evidence})=='unknown'
    assert observation_support_status({**candidate,'observation_support':{**evidence,'status':[]}})=='unknown'


@pytest.mark.parametrize('config',[{'patch_radius':True},{'patch_radius':500},{'prealigned':'yes'}, {'minimum_contrast':0}])
def test_compute_bounds_and_fixed_thresholds_are_validated(config):
    with pytest.raises(ValueError):
        assess_observation_support(static_frames(),4,observation(),config=config)
    with pytest.raises(ValueError):
        annotate_observations([{'frame_index':4,'detections':[observation()]}],static_frames(),config=config)


def test_batch_preserves_raw_records_reports_budget_and_reuses_duplicate_patch_evidence():
    frames=static_frames();candidate=observation();candidate.pop('frame_index')
    record={'frame_index':4,'time':.12,'custom_source_field':'kept',
            'detections':[{**candidate,'confidence':.9},{**candidate,'confidence':.8},
                          {**candidate,'confidence':.7,'source_bbox':[1,2,3,4]}]}
    original=deepcopy(record);progress=[]
    output, report=annotate_observations([record],frames,
        config={'max_observations_per_frame':2},progress_callback=progress.append)
    assert record==original and len(output[0]['detections'])==3
    assert output[0]['detections'][2]==record['detections'][2]
    assert output[0]['custom_source_field']=='kept'
    assert report['assessed_observations']==2 and report['unique_assessments']==1
    assert report['unassessed_due_to_budget']==1 and report['rejected_moving_observations']==2
    assert report['elapsed_seconds']>=0 and progress[-1]['processed_frames']==1
    assert all(rejects_moving_association(row) for row in output[0]['detections'][:2])


def test_native_mapping_preserves_native_pixel_provenance_and_cannot_move_the_point():
    native=assess_observation_support(static_frames(),4,observation())
    candidate=observation(x=91.,y=66.,bbox=[90,65,2,2])
    mapped=remap_native_support(candidate,native,source_roi=[300,200,128,128],scale_xy=[4,4],coordinate_size=[128,128])
    assert mapped['point_px']==[91,66] and mapped['native_pixel_evidence']==native
    assert mapped['native_mapping']['native_evidence_units']=='actual_source_patch_pixels'
    assert rejects_moving_association({**candidate,'observation_support':mapped})
    assert not rejects_moving_association({**candidate,'observation_support':native})
    with pytest.raises(ValueError):
        remap_native_support({**candidate,'x':92},native,source_roi=[300,200,128,128],scale_xy=[4,4],coordinate_size=[128,128])


def test_batch_invalid_candidate_geometry_does_not_retain_stale_negative_evidence():
    frames=static_frames();candidate={**observation(),'x':float('nan'),'observation_support':{'status':'persistent_background'}}
    output, report=annotate_observations([{'frame_index':4,'detections':[candidate]}],frames)
    assert 'observation_support' not in output[0]['detections'][0]
    assert report['invalid_geometry_observations']==1
    assert 'observation_support' in candidate


def test_budget_excluded_candidates_do_not_keep_old_source_pixel_annotations():
    frames=static_frames();evidence=assess_observation_support(frames,4,observation())
    candidate={**observation(),'observation_support':evidence}
    records=[{'frame_index':4,'detections':[deepcopy(candidate),deepcopy(candidate)]}]
    output, report=annotate_observations(records,frames,config={'max_observations_per_frame':1})
    assert report['unassessed_due_to_budget']==1
    assert 'observation_support' not in output[0]['detections'][1]
    assert 'observation_support' in records[0]['detections'][1]
