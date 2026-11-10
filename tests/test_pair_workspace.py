from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import threading
import time

import pytest

from badminton.pair_workspace import (PairWorkspace, PairWorkspaceError, combine_views, MAX_UPLOAD_BYTES,
                                      MAX_FRAMES, RULES, _write)


def wait_job(workspace, job_id):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        job = workspace.get_job(job_id)
        if job['status'] in {'completed', 'completed_with_errors', 'failed', 'interrupted'}:
            return job
        time.sleep(.01)
    pytest.fail('Pair job did not terminate')


def source_view(label, digest='a', **kwargs):
    return {'status': 'completed', 'source_sha256': digest * 64,
            'direct_decision': label, 'assisted_decision': label, 'qualified': label != 'REVIEW',
            'automatic_decision': 'REVIEW', 'rules': deepcopy(RULES), **kwargs}


@pytest.mark.parametrize('left,right,expected', [('IN','REVIEW','IN'), ('REVIEW','OUT','OUT'),
                                              ('IN','IN','IN'), ('OUT','OUT','OUT'),
                                              ('IN','OUT','REVIEW'), ('REVIEW','REVIEW','REVIEW')])
def test_pair_final_uses_available_direct_evidence_or_conflict(left, right, expected):
    views = {'A': source_view(left), 'B': source_view(right, 'b')}
    result = combine_views(views, same_event_confirmed=True, event_source='operator_pair_selection')
    assert result['decision'] == expected
    assert result['automatic_benchmark_eligible'] is False and result['ground_truth_created'] is False
    assert result['court_position_m'] is None
    assert views['B']['direct_decision'] == right
    if expected != 'REVIEW':
        assert result['source_camera_ids']
        assert all(v['assisted_decision'] == expected for v in result['views'].values())


def test_transferred_label_cannot_be_used_as_a_direct_donor():
    views = {'A': source_view('REVIEW', assisted_decision='IN'),
             'B': source_view('REVIEW', 'b')}
    assert combine_views(views, same_event_confirmed=True)['decision'] == 'REVIEW'


@pytest.mark.parametrize('adjust,reason', [
    ({'same_event_confirmed': False}, 'SAME_EVENT_UNCONFIRMED'),
    ({'different_event': True}, 'KNOWN_DIFFERENT_RALLY'),
])
def test_event_identity_is_a_required_condition(adjust, reason):
    options = {'same_event_confirmed': True, **adjust}
    result = combine_views({'A': source_view('IN'), 'B': source_view('IN','b')}, **options)
    assert result['decision'] == 'REVIEW' and reason in result['reason_codes']


def test_duplicate_and_failed_other_view_veto_a_positive():
    result = combine_views({'A': source_view('IN'), 'B': source_view('REVIEW')}, same_event_confirmed=True)
    assert 'DUPLICATE_SOURCE' in result['reason_codes']
    result = combine_views({'A': source_view('IN'), 'B': source_view('REVIEW','b',status='failed')}, same_event_confirmed=True)
    assert result['decision'] == 'REVIEW' and 'VIEW_ANALYSIS_FAILED' in result['reason_codes']


@pytest.fixture
def pair_fixture(tmp_path):
    root = tmp_path/'project'
    root.mkdir()
    report_root = root/'runs/ai-reviewed-poc'
    rows = []
    for name, payload, label in [('a',b'first-source','IN'),('b',b'second-source','REVIEW')]:
        digest = hashlib.sha256(payload).hexdigest()
        identifier = 'v_' + name
        source = root/'video'/f'{name}.mp4'
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(payload)
        annotation = {
            'schema_version':1,'video_id':identifier,'source_sha256':digest,'file':source.name,
            'source':'ai_visual_review','status':'reviewed_by_ai','human_verified':False,
            'label': label if label!='REVIEW' else 'UNKNOWN','rules':deepcopy(RULES),
            'analysis_size':{'width':960,'height':540},'source_size':{'width':3840,'height':2160},
            'contact_observed':label!='REVIEW','target_contact_identity_confirmed':label!='REVIEW',
            'contact_frame_index':3,'time':.3,'contact_px':[100,100],'rationale':'Reviewed native contact.',
            'inspected_frame_indices':list(range(20)),
            'boundary_evidence':{'kind':'court_interior','interior_confirmed':True,'polygon_px':[[50,50],[200,50],[200,200],[50,200]],
                'contact_uncertainty_px':1,'boundary_uncertainty_px':1,
                'explanation':'All four edges lie safely inside the valid court.'},
        }
        _write(root/f'annotations/ai-visual/{identifier}.json', annotation)
        folder = report_root/f'assets/{identifier}'
        folder.mkdir(parents=True)
        (folder/'preview.mp4').write_bytes(b'preview')
        (folder/'contact.jpg').write_bytes(b'image')
        row = {'video_id':identifier,'file':source.name,'source_sha256':digest,
               'automatic_decision':'REVIEW','direct_assisted_decision':label,'assisted_decision':label,
               'annotation':annotation,'source_integrity':{'verified':True,'sha256':digest,'path':f'video/{source.name}'},
               'review':{'valid':True,'qualified':label!='REVIEW','errors':[],'reason_codes':[]},
               'assets':{'preview':f'assets/{identifier}/preview.mp4','evidence':[{'path':f'assets/{identifier}/contact.jpg'}]}}
        rows.append(row)
    rows[1]['annotation']['cross_view_match'] = {
        'other_video_id':'v_a','same_event_confirmed_by_ai':True,
        'matched_actions':[{'frame_index':1,'other_frame_index':2,'description':'first stroke'},
                           {'frame_index':6,'other_frame_index':7,'description':'second stroke'}]}
    rows[1]['cross_view_review'] = {'qualified':True,'originating_video_id':'v_a'}
    rows[1]['assisted_decision'] = 'IN'
    _write(root/'annotations/ai-visual/v_b.json', rows[1]['annotation'])
    report = {'status':'completed','review_source':'ai_visual_review','ground_truth_eligible':False,'videos':rows}
    _write(report_root/'visual-review.json', report)
    workspace = PairWorkspace(root,root/'runs/pairs')
    yield workspace, root, report
    workspace.close()


def submit_fixture(workspace, **kwargs):
    return workspace.submit({'views':{'A':{'video_id':'v_a'},'B':{'video_id':'v_b'}}, **kwargs})['job_id']


def test_stored_pair_resolves_from_one_direct_view_and_persists(pair_fixture):
    workspace, root, report = pair_fixture
    job = wait_job(workspace, submit_fixture(workspace))
    assert job['status'] == 'completed'
    result = job['result']
    assert result['decision'] == 'IN'
    assert result['event_correspondence_source'] == 'verified_ai_action_correspondence'
    assert result['source_camera_ids'] == ['A']
    assert result['views']['A']['processing_mode'] == 'stored_ai_review'
    assert result['views']['B']['direct_decision'] == 'REVIEW'
    assert result['views']['B']['assisted_decision'] == 'IN'
    assert result['views']['B']['contact_time_seconds'] is None
    assert result['views']['A']['contact_time_seconds'] == .3
    assert (root/f'runs/pairs/jobs/{job["job_id"]}/job.json').is_file()
    assert workspace.list_jobs()['jobs'][0]['decision'] == 'IN'


def test_no_filename_only_propagation_and_explicit_operator_selection(pair_fixture):
    workspace, root, report = pair_fixture
    report['videos'][1].pop('cross_view_review')
    report['videos'][1]['annotation'].pop('cross_view_match')
    _write(root/'annotations/ai-visual/v_b.json', report['videos'][1]['annotation'])
    _write(workspace.report_path, report)
    unresolved = wait_job(workspace, submit_fixture(workspace))['result']
    assert unresolved['decision'] == 'REVIEW'
    accepted = wait_job(workspace, submit_fixture(workspace,same_event_confirmed=True))['result']
    assert accepted['decision'] == 'IN'
    assert accepted['event_correspondence_source'] == 'operator_pair_selection'


def test_known_mismatch_overrides_operator_confirmation(pair_fixture):
    workspace, root, report = pair_fixture
    report['videos'][0]['annotation']['cross_view_candidates'] = [
        {'other_video_id':'v_b','status':'different_rally','same_event_confirmed_by_ai':False}]
    _write(root/'annotations/ai-visual/v_a.json', report['videos'][0]['annotation'])
    _write(workspace.report_path, report)
    result = wait_job(workspace, submit_fixture(workspace,same_event_confirmed=True))['result']
    assert result['decision'] == 'REVIEW' and 'KNOWN_DIFFERENT_RALLY' in result['reason_codes']


@pytest.mark.parametrize('damage', ['annotation','original','asset','forged_direct'])
def test_stale_source_or_review_cannot_supply_positive_verdict(pair_fixture, damage):
    workspace, root, report = pair_fixture
    if damage == 'annotation':
        annotation = deepcopy(report['videos'][0]['annotation'])
        annotation['rationale'] = 'Changed after report build.'
        _write(root/'annotations/ai-visual/v_a.json', annotation)
    elif damage == 'original':
        (root/'video/a.mp4').write_bytes(b'changed-source')
    elif damage == 'asset':
        (workspace.review_root/'assets/v_a/contact.jpg').unlink()
    else:
        report['videos'][0]['direct_assisted_decision'] = 'OUT'
        _write(workspace.report_path, report)
    result = wait_job(workspace, submit_fixture(workspace,same_event_confirmed=True))['result']
    assert result['decision'] == 'REVIEW'


def test_portable_review_does_not_require_excluded_originals(pair_fixture):
    workspace, root, _ = pair_fixture
    for source in (root/'video').iterdir():
        source.unlink()
    result = wait_job(workspace, submit_fixture(workspace))['result']
    assert result['decision'] == 'IN'
    assert result['views']['A']['source_verification'] == 'packaged_source_bound_review'
    assert len(workspace.library()['videos']) == 2


def test_upload_matches_original_hash_even_when_file_is_renamed(pair_fixture,monkeypatch):
    workspace, root, _ = pair_fixture
    monkeypatch.setattr('badminton.media.probe',lambda path:{'frame_count':10,'width':3840,'height':2160,'duration':1,'timestamps':[0]})
    staged = workspace.stage_upload(io.BytesIO(b'first-source'),len(b'first-source'),'renamed.mp4')
    job_id = workspace.submit({'views':{'A':{'upload_id':staged['upload_id']},'B':{'video_id':'v_b'}}})['job_id']
    result = wait_job(workspace,job_id)['result']
    assert result['decision'] == 'IN'
    assert result['views']['A']['file'] == 'renamed.mp4'
    assert result['views']['A']['stored_video_id'] == 'v_a'


def test_upload_invalid_length_paths_and_decode_failure_cleanup(pair_fixture,monkeypatch):
    workspace, _, _ = pair_fixture
    for filename in ('../a.mp4',r'..\a.mp4','/a.mp4','C:\\a.mp4','a.txt'):
        with pytest.raises(PairWorkspaceError):
            workspace.stage_upload(io.BytesIO(b'123'),3,filename)
    with pytest.raises(PairWorkspaceError) as error:
        workspace.stage_upload(io.BytesIO(),MAX_UPLOAD_BYTES+1,'a.mp4')
    assert error.value.status == 413
    with pytest.raises(PairWorkspaceError):
        workspace.stage_upload(io.BytesIO(b'1'),3,'a.mp4')
    monkeypatch.setattr('badminton.media.probe',lambda path: (_ for _ in ()).throw(ValueError('bad video')))
    with pytest.raises(PairWorkspaceError):
        workspace.stage_upload(io.BytesIO(b'123'),3,'a.mp4')
    assert list((workspace.root/'uploads').iterdir()) == []


def test_upload_frame_limit_is_rejected_without_truncation(pair_fixture,monkeypatch):
    workspace, _, _ = pair_fixture
    monkeypatch.setattr('badminton.media.probe',lambda path:{'frame_count':MAX_FRAMES+1,'width':960,'height':540})
    with pytest.raises(PairWorkspaceError) as error:
        workspace.stage_upload(io.BytesIO(b'123'),3,'a.mp4')
    assert error.value.status == 413


def test_asset_endpoints_cannot_expose_uploads_or_escape_workspace(pair_fixture):
    workspace, root, _ = pair_fixture
    for path in ('../secret.mp4','uploads/abc/source.mp4','jobs/a/job.json','jobs/a/views/../../secret.mp4'):
        with pytest.raises(PairWorkspaceError):
            workspace.asset_path(path)
    (workspace.root/'jobs').mkdir(exist_ok=True)
    (workspace.root/'jobs/linked').symlink_to(root)
    with pytest.raises(PairWorkspaceError):
        workspace.asset_path('jobs/linked/views/a.mp4')
    with pytest.raises(PairWorkspaceError):
        workspace.review_asset_path('../visual-review.json')


def test_job_survives_restart_and_inflight_work_becomes_interrupted(pair_fixture):
    workspace, root, _ = pair_fixture
    completed = wait_job(workspace,submit_fixture(workspace))
    workspace.close()
    interrupted = deepcopy(completed)
    interrupted.update(job_id='unfinished',status='running',result=None)
    _write(workspace.root/'jobs/unfinished/job.json',interrupted)
    restored = PairWorkspace(root,workspace.root)
    try:
        assert restored.get_job(completed['job_id'])['result']['decision'] == 'IN'
        assert restored.get_job('unfinished')['status'] == 'interrupted'
        assert restored.get_job('unfinished')['result'] is None
    finally:
        restored.close()


def test_new_sources_run_both_views_and_keep_automatic_review(pair_fixture,monkeypatch):
    workspace, root, _ = pair_fixture
    monkeypatch.setattr('badminton.media.probe',lambda path:{'frame_count':3,'width':960,'height':540,'timestamps':[0,.1,.2]})
    barrier = threading.Barrier(2,timeout=4)
    calls = []
    def analyze(source,output,config):
        barrier.wait()
        output.mkdir(parents=True)
        (output/'preview.mp4').write_bytes(b'preview')
        calls.append(source)
        return {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'decision':{'decision':'REVIEW','reason_codes':['NO_CONTACT']},
                'artifacts':{'preview':'preview.mp4','evidence':[]},'timings':{'total_seconds':.1}}
    monkeypatch.setattr('badminton.pipeline.analyze_video',analyze)
    a=workspace.stage_upload(io.BytesIO(b'new-a'),5,'new-a.mp4')
    b=workspace.stage_upload(io.BytesIO(b'new-b'),5,'new-b.mp4')
    job = workspace.submit({'views':{'A':{'upload_id':a['upload_id']},'B':{'upload_id':b['upload_id']}},'same_event_confirmed':True})
    result = wait_job(workspace,job['job_id'])['result']
    assert len(calls)==2 and result['decision']=='REVIEW'
    assert all(v['processing_mode']=='fresh_analysis' for v in result['views'].values())
    assert all(v['preview_url'].startswith('/pair-media/jobs/') for v in result['views'].values())


def test_duplicate_hash_upload_does_not_choose_arbitrary_review_identity(pair_fixture,monkeypatch):
    workspace, root, report = pair_fixture
    duplicate = deepcopy(report['videos'][1])
    duplicate['video_id']='v_b_duplicate'
    duplicate['annotation']['video_id']='v_b_duplicate'
    duplicate['file']='duplicate.mp4'
    duplicate['annotation'].pop('cross_view_match')
    duplicate.pop('cross_view_review')
    report['videos'].append(duplicate)
    _write(root/'annotations/ai-visual/v_b_duplicate.json',duplicate['annotation'])
    _write(workspace.report_path,report)
    monkeypatch.setattr('badminton.media.probe',lambda path:{'frame_count':3,'width':960,'height':540,'timestamps':[0,.1,.2]})
    staged=workspace.stage_upload(io.BytesIO(b'second-source'),len(b'second-source'),'some-name.mp4')
    _, _, candidates, entry=workspace._resolve({'upload_id':staged['upload_id']},workspace._catalog())
    assert set(candidates)=={'v_b','v_b_duplicate'} and entry is None


def test_fresh_mode_does_not_reuse_verified_ai_event_link(pair_fixture,monkeypatch):
    workspace, _, _ = pair_fixture
    def fresh(identifier,camera,selected,reuse,catalog,refine=False):
        assert reuse is False
        assert refine is False
        row = catalog[selected['video_id']]['row']
        return source_view('REVIEW', _candidate_ids=[selected['video_id']],
                           source_sha256=row['source_sha256'])
    monkeypatch.setattr(workspace,'_view',fresh)
    result=wait_job(workspace,submit_fixture(workspace,reuse_review=False))['result']
    assert result['same_event_confirmed'] is False
    assert result['event_correspondence_source'] is None
    assert result['matched_actions'] == []


def test_event_match_can_be_valid_while_both_contacts_remain_unresolved(pair_fixture):
    workspace, root, report = pair_fixture
    report['videos'][0].update(direct_assisted_decision='REVIEW', assisted_decision='REVIEW')
    report['videos'][1]['cross_view_review'].update(qualified=False, event_correspondence_validated=True)
    _write(workspace.report_path, report)
    result = wait_job(workspace, submit_fixture(workspace))['result']
    assert result['same_event_confirmed'] is True
    assert result['decision'] == 'REVIEW'
    assert result['reason_codes'] == ['NO_QUALIFIED_CONTACT']
    assert result['source_camera_ids'] == []


@pytest.mark.parametrize('options', [{'reuse_review':True,'refine_review_regions':True},
                                     {'reuse_review':False,'refine_review_regions':'yes'}])
def test_refinement_requires_explicit_boolean_and_fresh_processing(pair_fixture, options):
    workspace, _, _ = pair_fixture
    with pytest.raises(PairWorkspaceError) as error:
        submit_fixture(workspace, **options)
    assert error.value.status == 400


def test_refinement_passes_hints_not_labels_and_marks_assisted_view(pair_fixture, monkeypatch):
    workspace, _, _ = pair_fixture
    configs = []
    monkeypatch.setattr('badminton.media.probe', lambda _: {'frame_count':20,'width':3840,'height':2160})
    def analyze(source, output, config):
        configs.append(config)
        return {'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                'decision':{'decision':'REVIEW','reason_codes':['FIRST_CONTACT_NOT_OBSERVED']},
                'artifacts':{},'timings':{}}
    monkeypatch.setattr('badminton.pipeline.analyze_video', analyze)
    result = wait_job(workspace, submit_fixture(workspace,reuse_review=False,refine_review_regions=True))['result']
    assert result['decision'] == 'REVIEW'
    guided = [c for c in configs if c.get('detector',{}).get('refinement_regions')]
    assert len(configs) == 2 and len(guided) == 1
    assert guided[0]['detector']['include_dark_components']
    assert all('contact_observed' not in c and 'target_track_id' not in c for c in configs)
    assert result['views']['A']['processing_mode']=='review_guided_reanalysis'
    assert result['views']['B']['processing_mode']=='fresh_analysis'
    assert not any(v['qualified'] for v in result['views'].values())


def test_analysis_only_library_has_independent_asset_scope(pair_fixture):
    workspace, root, _ = pair_fixture
    folder=workspace.library_root/'videos/v_extra'
    folder.mkdir(parents=True)
    (folder/'preview.mp4').write_bytes(b'video')
    _write(workspace.library_root/'index.json',{'videos':[{'id':'v_extra','file':'extra.mp4',
           'preview_path':'videos/v_extra/preview.mp4','decision':'REVIEW'}]})
    row=next(v for v in workspace.library()['videos'] if v['id']=='v_extra')
    assert row['preview_url']=='/pair-library-assets/videos/v_extra/preview.mp4'
    assert workspace.library_asset_path('videos/v_extra/preview.mp4')==folder/'preview.mp4'
    with pytest.raises(PairWorkspaceError):
        workspace.library_asset_path('index.json')


def test_analysis_original_fallback_cannot_follow_external_symlink(pair_fixture,tmp_path):
    workspace, root, _ = pair_fixture
    outside=tmp_path/'elsewhere.mp4'
    outside.write_bytes(b'not-a-project-source')
    (root/'video/linked.mp4').symlink_to(outside)
    entry={'kind':'analysis','row':{'id':'v_linked','file':'linked.mp4'}}
    with pytest.raises(PairWorkspaceError):
        workspace._original(entry)
