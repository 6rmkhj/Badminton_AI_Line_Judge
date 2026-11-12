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
