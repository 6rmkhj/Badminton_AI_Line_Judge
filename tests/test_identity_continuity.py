from copy import deepcopy
import json

from badminton.identity_continuity import assess_identity_continuity


def path(*, flip=True, resize=True, occluded=True):
    rows=[]
    for i in range(16):
        after=i>=7
        radius=(.9 if i==10 else 4.2) if after and resize else 2.
        rows.append({'frame_index':i,'time':i/30,'x':100+5*i,'y':40+3*i,
                     'radius':radius,'appearance_polarity':'dark' if after and flip else 'bright',
                     'possible_occluder_overlap':after and occluded,'observed':True})
    return rows


def test_combined_occluder_entry_and_appearance_break_vetoes_only_flight_support():
    observations=path()
    before=deepcopy(observations)
    result=assess_identity_continuity(observations,10)
    assert result['flight_support_veto'] and not result['flight_support_eligible']
    assert result['reason_codes']==['OCCLUDER_ENTRY_AND_APPEARANCE_DISCONTINUITY']
    assert not result['identity_confirmed'] and not result['physical_contact_confirmed']
    assert result['evidence'][0]['entry_frame']==7
    assert result['evidence'][0]['polarity_change'] and result['evidence'][0]['size_discontinuity']
    assert observations==before
    json.dumps(result,allow_nan=False)


def test_real_shuttle_can_cross_player_with_stable_size_despite_polarity_change():
    result=assess_identity_continuity(path(resize=False),10)
    assert not result['flight_support_veto']
    assert result['evidence'][0]['polarity_change']
    assert not result['evidence'][0]['size_discontinuity']


def test_blur_size_change_and_player_overlap_without_polarity_flip_is_retained():
    result=assess_identity_continuity(path(flip=False),10)
    assert not result['flight_support_veto']
    assert result['evidence'][0]['size_discontinuity']
    assert not result['evidence'][0]['polarity_change']


def test_contrast_and_size_change_without_occluder_entry_is_retained():
    result=assess_identity_continuity(path(occluded=False),10)
    assert not result['flight_support_veto']
    assert result['evidence']==[]


def test_unknown_appearance_or_interpolated_support_cannot_create_veto():
    rows=path()
    for observation in rows:
        observation.pop('appearance_polarity')
    result=assess_identity_continuity(rows,10)
    assert not result['flight_support_veto'] and result['status']=='insufficient_appearance_evidence'
    rows=path()
    for observation in rows[:7]:
        observation['observed']=False
    assert not assess_identity_continuity(rows,10)['flight_support_veto']


def test_prior_ambiguous_appearance_does_not_become_identity_confirmation():
    rows=path()
    for i,observation in enumerate(rows[:7]):
        observation['appearance_polarity']='dark' if i%2 else 'bright'
    result=assess_identity_continuity(rows,10)
    assert not result['flight_support_veto']
    assert not result['identity_confirmed']


def collapsed_path():
    rows=path(flip=False,resize=False)
    for observation in rows:
        observation['radius']=3.0 if observation['frame_index']<7 else .6
    return rows


def test_sustained_size_collapse_inside_occluder_does_not_need_polarity_flip():
    rows=collapsed_path()
    before=deepcopy(rows)
    result=assess_identity_continuity(rows,10)
    assert result['flight_support_veto']
    assert result['reason_codes']==['OCCLUDER_ENTRY_AND_SUSTAINED_SIZE_COLLAPSE']
    evidence=result['evidence'][0]
    assert not evidence['polarity_change'] and evidence['size_collapse_veto']
    assert evidence['post_collapsed_radius_count']==5
    assert evidence['post_wholly_overlaps_possible_occluder']
    assert rows==before
    assert not result['identity_confirmed'] and not result['physical_contact_confirmed']
    json.dumps(result,allow_nan=False)


def test_collapsed_flight_emerging_from_occluder_is_retained():
    rows=collapsed_path()
    rows[12]['possible_occluder_overlap']=False
    assert not assess_identity_continuity(rows,10)['flight_support_veto']
    rows[12].pop('possible_occluder_overlap')
    assert not assess_identity_continuity(rows,10)['flight_support_veto']


def test_transient_or_moderate_size_change_does_not_trigger_collapse_veto():
    rows=collapsed_path()
    for observation in rows[11:]:
        observation['radius']=3.0
    assert not assess_identity_continuity(rows,10)['flight_support_veto']
    rows=collapsed_path()
    for observation in rows[7:]:
        observation['radius']=1.5
    assert not assess_identity_continuity(rows,10)['flight_support_veto']


def test_collapse_needs_three_distinct_measured_post_event_samples():
    rows=collapsed_path()[:13]
    rows.extend([deepcopy(rows[-1]),deepcopy(rows[-1])])
    assert not assess_identity_continuity(rows,10)['flight_support_veto']
    rows=collapsed_path()
    for observation in rows[11:14]:
        observation['observed']=False
    assert not assess_identity_continuity(rows,10)['flight_support_veto']
    rows=collapsed_path()
    for observation in rows[11:]:
        observation.pop('radius')
    assert not assess_identity_continuity(rows,10)['flight_support_veto']


def test_extreme_event_collapse_with_partial_but_small_post_fragments_is_withheld():
    rows=collapsed_path()
    # Segmentation can recover some pixels while the suspected association
    # stays on the occluder. It has not regained the incoming component size.
    for observation,radius in zip(rows[11:],(1.2,1.3,1.8,1.4,1.1)):
        observation['radius']=radius
    result=assess_identity_continuity(rows,10)
    assert result['flight_support_veto']
    assert result['evidence'][0]['post_collapsed_radius_count']==4
    rows[13]['possible_occluder_overlap']=False
    assert not assess_identity_continuity(rows,10)['flight_support_veto']


def test_one_small_post_fragment_does_not_establish_sustained_collapse():
    rows=collapsed_path()
    for observation,radius in zip(rows[11:],(.6,2.7,3.,3.1,3.)):
        observation['radius']=radius
    assert not assess_identity_continuity(rows,10)['flight_support_veto']


def test_three_sample_post_median_tolerates_one_segmentation_outlier():
    rows=collapsed_path()[:14]
    for observation,radius in zip(rows[11:],(1.,.5,1.7)):
        observation['radius']=radius
    result=assess_identity_continuity(rows,10)
    assert result['flight_support_veto']
    assert result['evidence'][0]['post_distinct_observation_count']==3


def test_post_gap_fragments_cannot_establish_sustained_local_size_collapse():
    rows=collapsed_path()
    rows.append({**rows[-1], 'frame_index':16, 'time':16/30})
    for observation in rows[11:14]:
        observation['radius']=1.8
    for observation in rows[14:]:
        observation['frame_index']+=15
        observation['time']+=.5
    result=assess_identity_continuity(rows,10)
    assert not result['flight_support_veto']
    assert result['evidence'][0]['post_distinct_observation_count']==3
    assert result['evidence'][0]['post_distinct_median_radius_ratio']==.6
    # Timestamp gaps must also be respected when frame labels are adjacent.
    for index,observation in enumerate(rows):
        observation['frame_index']=index
    result=assess_identity_continuity(rows,10,nominal_step=1/30)
    assert not result['flight_support_veto']
    assert result['evidence'][0]['post_distinct_observation_count']==3
