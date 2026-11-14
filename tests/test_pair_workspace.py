from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import threading
import time

import pytest

from badminton.pair_workspace import (PairWorkspace, PairWorkspaceError, combine_views, MAX_UPLOAD_BYTES,
                                      MAX_FRAMES, RULES, _write, _landing_candidate, _trajectory_replay)


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
    return workspace.submit({'views':{'A':{'video_id':'v_a'},'B':{'video_id':'v_b'}}, 'reuse_review':True, **kwargs})['job_id']


def test_stored_pair_resolves_from_one_direct_view_and_persists(pair_fixture):
    workspace, root, report = pair_fixture
    job = wait_job(workspace, submit_fixture(workspace))
    assert job['status'] == 'completed'
    result = job['result']
    assert result['decision'] == 'IN'
    assert result['event_correspondence_source'] == 'verified_ai_action_correspondence'
    assert result['source_camera_ids'] == ['A']
    assert result['views']['A']['processing_mode'] == 'stored_ai_review'
    assert result['views']['A']['trajectory_replay_url'] is None
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
    job_id = workspace.submit({'views':{'A':{'upload_id':staged['upload_id']},'B':{'video_id':'v_b'}},'reuse_review':True})['job_id']
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
    def analyze(source,output,config,progress_callback=None):
        barrier.wait()
        output.mkdir(parents=True)
        (output/'preview.mp4').write_bytes(b'preview')
        (output/'evidence').mkdir()
        (output/'evidence/trajectory-context.jpg').write_bytes(b'context-image')
        (output/'evidence/annotated.mp4').write_bytes(b'candidate-replay')
        (output/'evidence/calibration.jpg').write_bytes(b'line-diagnostic')
        calls.append(source)
        return {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'decision':{'decision':'REVIEW','reason_codes':['NO_CONTACT'],'contact_time_seconds':.1},
                'assessment':{'schema_version':1,'line_verdict':'REVIEW','changes_decision':False,'stages':[]},
                'calibration':{'status':'review','line_segments_px':[[10,20,100,20]],'reason_codes':['PARTIAL_COURT_VISIBLE']},
                'analysis':{'width':960,'height':540,'frame_count':3},
                'selected_landing':{'candidate_px':[361,444],'frame_index':1,'time':.1,'contact_observed':False,
                    'image_motion_event':{'kind':'descent_and_rebound','inference_only':True,
                        'frame_interval':[0,2],'time_interval':[0,.2]}},
                'artifacts':{'preview':'preview.mp4','evidence':[{'trajectory_context':'evidence/trajectory-context.jpg'}],
                    'calibration_overlay':'evidence/calibration.jpg',
                    'annotated_clip':{'path':'evidence/annotated.mp4','source_time_origin':.05,
                        'source_start_frame':1,'source_end_frame':2,'frame_count':2}},
                'timings':{'total_seconds':.1}}
    monkeypatch.setattr('badminton.pipeline.analyze_video',analyze)
    a=workspace.stage_upload(io.BytesIO(b'new-a'),5,'new-a.mp4')
    b=workspace.stage_upload(io.BytesIO(b'new-b'),5,'new-b.mp4')
    job = workspace.submit({'views':{'A':{'upload_id':a['upload_id']},'B':{'upload_id':b['upload_id']}},'same_event_confirmed':True})
    result = wait_job(workspace,job['job_id'])['result']
    assert len(calls)==2 and result['decision']=='REVIEW'
    assert all(v['processing_mode']=='fresh_analysis' for v in result['views'].values())
    assert all(v['preview_url'].startswith('/pair-media/jobs/') for v in result['views'].values())
    for view in result['views'].values():
        assert view['qualified'] is False and view['contact_time_seconds'] is None
        assert view['assessment']['line_verdict']=='REVIEW' and view['assessment']['changes_decision'] is False
        assert view['landing_candidate']['status']=='approximate'
        assert view['landing_candidate']['contact_observed'] is False
        assert view['landing_candidate']['point_kind']=='object_center'
        assert view['landing_candidate']['point_px']==[361,444]
        assert view['landing_candidate']['frame_interval']==[0,2]
        assert view['landing_candidate']['time_interval']==[0,.2]
        assert view['trajectory_context_url'].endswith('/evidence/trajectory-context.jpg')
        assert view['trajectory_replay_url'].endswith('/evidence/annotated.mp4')
        assert view['trajectory_replay_info']['source_time_origin']==.05
        assert view['preview_url'] != view['trajectory_replay_url']
        assert view['calibration_summary']['detected_line_count'] == 1
        assert view['calibration_summary']['court_geometry_status'] == 'unconfirmed'
        assert view['calibration_preview_url'].endswith('/evidence/calibration.jpg')


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


@pytest.mark.parametrize('options', [{}, {'reuse_review':False}])
def test_fresh_mode_does_not_reuse_verified_ai_event_link(pair_fixture,monkeypatch,options):
    workspace, _, _ = pair_fixture
    def fresh(identifier,camera,selected,reuse,catalog,refine=False):
        assert reuse is False
        assert refine is False
        row = catalog[selected['video_id']]['row']
        return source_view('REVIEW', _candidate_ids=[selected['video_id']],
                           source_sha256=row['source_sha256'])
    monkeypatch.setattr(workspace,'_view',fresh)
    identifier=workspace.submit({'views':{'A':{'video_id':'v_a'},'B':{'video_id':'v_b'}},**options})['job_id']
    job=wait_job(workspace,identifier)
    assert job['request']['reuse_review'] is False
    result=job['result']
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
    def analyze(source, output, config, progress_callback=None):
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


def test_live_progress_reports_both_views_and_freezes_completed_durations(pair_fixture, monkeypatch):
    workspace, root, _ = pair_fixture
    clock = [100.]
    monkeypatch.setattr('badminton.pair_workspace.time.perf_counter', lambda: clock[0])
    monkeypatch.setattr('badminton.media.probe', lambda _: {'frame_count':3,'width':960,'height':540})
    reached = threading.Barrier(3, timeout=4)
    release = threading.Event()

    def analyze(source, output, config, progress_callback=None):
        progress_callback({'stage':'detection','completed_units':1,'total_units':3,'unit':'frames'})
        reached.wait()
        assert release.wait(4)
        progress_callback({'stage':'detection','completed_units':3,'total_units':3,'unit':'frames'})
        progress_callback({'stage':'save'})
        return {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'decision':{'decision':'REVIEW'},'artifacts':{},'timings':{}}

    monkeypatch.setattr('badminton.pipeline.analyze_video', analyze)
    identifier = submit_fixture(workspace, reuse_review=False)
    try:
        reached.wait()
        clock[0] = 107.
        running = workspace.get_job(identifier)
        assert running['phase'] == 'analyzing'
        assert running['elapsed_seconds'] == 7.
        assert running['server_time']
        for view in running['views'].values():
            assert view['status'] == 'running' and view['elapsed_seconds'] == 7.
            progress = view['progress']
            assert progress['stage'] == 'detection' and progress['stage_index'] == 3
            assert progress['completed_units'] == 1 and progress['total_units'] == 3
            assert progress['stage_elapsed_seconds'] == 7.
        # Frame updates change memory for polling, but do not write per frame.
        persisted = json.loads((workspace.root/f'jobs/{identifier}/job.json').read_text())
        assert persisted['views']['A']['progress']['stage'] == 'detection'
        clock[0] = 109.
        assert workspace.get_job(identifier)['elapsed_seconds'] == 9.
    finally:
        release.set()
    finished = wait_job(workspace, identifier)
    assert finished['status'] == finished['phase'] == 'completed'
    assert finished['finished_at'] and finished['elapsed_seconds'] == 9.
    for camera, view in finished['views'].items():
        assert view['progress']['stage'] == 'completed'
        assert view['stage_history'][-1]['stage'] == 'save'
        assert all(stage['status'] == 'completed' for stage in view['stage_history'])
        assert finished['result']['views'][camera]['stage_history'] == view['stage_history']
    clock[0] = 1000.
    assert workspace.get_job(identifier)['elapsed_seconds'] == 9.
    assert workspace.list_jobs()['jobs'][0]['elapsed_seconds'] == 9.


def test_failure_preserves_the_actual_stage_and_partial_frame_count(pair_fixture, monkeypatch):
    workspace, _, _ = pair_fixture
    monkeypatch.setattr('badminton.media.probe', lambda _: {'frame_count':7,'width':960,'height':540})

    def analyze(source, output, config, progress_callback=None):
        progress_callback({'stage':'detection','completed_units':2,'total_units':7,'unit':'frames'})
        if source.name == 'a.mp4':
            raise RuntimeError('decoder stopped')
        progress_callback({'stage':'save'})
        return {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'decision':{'decision':'REVIEW'},'artifacts':{},'timings':{}}

    monkeypatch.setattr('badminton.pipeline.analyze_video', analyze)
    job = wait_job(workspace, submit_fixture(workspace, reuse_review=False))
    assert job['status'] == 'completed_with_errors' and job['phase'] == 'failed'
    assert job['last_phase'] == 'analyzing'
    failed = job['views']['A']
    assert failed['status'] == 'failed'
    assert failed['progress']['stage'] == 'detection'
    assert failed['progress']['stage_status'] == 'failed'
    assert failed['progress']['completed_units'] == 2
    assert failed['stage_history'][-1]['status'] == 'failed'
    assert job['result']['views']['A']['progress'] == failed['progress']
    assert not any(stage['stage'] == 'save' for stage in failed['stage_history'])


def test_queued_time_is_distinct_from_analysis_and_stored_review_has_own_stage(pair_fixture):
    workspace, _, _ = pair_fixture
    blocker = threading.Event()
    workspace.executor.submit(blocker.wait)
    identifier = submit_fixture(workspace)
    try:
        queued = workspace.get_job(identifier)
        assert queued['status'] == queued['phase'] == 'queued'
        assert queued['started_at'] is None and queued['elapsed_seconds'] == 0.
        assert queued['queue_seconds'] >= 0.
        assert all(v['progress']['stage'] == 'queued' for v in queued['views'].values())
    finally:
        blocker.set()
    job = wait_job(workspace, identifier)
    assert job['queue_seconds'] >= queued['queue_seconds']
    for view in job['views'].values():
        assert view['processing_mode'] == 'stored_ai_review'
        assert [stage['stage'] for stage in view['stage_history']] == ['resolve', 'stored_review']
        assert view['progress']['last_stage'] == 'stored_review'


def test_restart_freezes_at_last_persisted_activity_and_keeps_legacy_jobs_compatible(pair_fixture):
    workspace, root, _ = pair_fixture
    workspace.close()
    started = '2026-01-01T00:00:00+00:00'
    stage_started = '2026-01-01T00:00:02+00:00'
    stopped = '2026-01-01T00:00:07+00:00'
    _write(workspace.root/'jobs/live/job.json', {
        'job_id':'live','status':'running','phase':'analyzing','created_at':started,
        'started_at':started,'updated_at':stopped,'result':None,
        'views':{'A':{'status':'running','started_at':started,
            'progress':{'stage':'detection','stage_started_at':stage_started,'completed_units':12,'total_units':30},
            'stage_history':[{'stage':'detection','started_at':stage_started,'status':'running'}]},
                 'B':{'status':'queued'}}})
    _write(workspace.root/'jobs/legacy/job.json', {'job_id':'legacy','status':'completed','created_at':started,
        'updated_at':stopped,'views':{'A':{'status':'completed'},'B':{'status':'completed'}},'result':{'decision':'IN'}})
    _write(workspace.root/'jobs/linking/job.json', {'job_id':'linking','status':'running','phase':'linking',
        'created_at':started,'started_at':started,'updated_at':stopped,'result':None,
        'views':{'A':{'status':'completed'},'B':{'status':'completed'}}})
    restored = PairWorkspace(root, workspace.root)
    try:
        job = restored.get_job('live')
        assert job['status'] == job['phase'] == 'interrupted'
        assert job['last_phase'] == 'analyzing'
        assert job['finished_at'] == stopped and job['elapsed_seconds'] == 7.
        view = job['views']['A']
        assert view['status'] == 'interrupted'
        assert view['elapsed_seconds'] == 7.
        assert view['progress']['stage'] == 'detection'
        assert view['progress']['stage_status'] == 'interrupted'
        assert view['progress']['stage_elapsed_seconds'] == 5.
        assert view['progress']['completed_units'] == 12
        assert view['stage_history'][-1]['elapsed_seconds'] == 5.
        assert restored.get_job('live')['elapsed_seconds'] == 7.
        linking = restored.get_job('linking')
        assert linking['status'] == 'interrupted' and linking['last_phase'] == 'linking'
        assert linking['elapsed_seconds'] == 7.
        assert all(view['status'] == 'completed' for view in linking['views'].values())
        legacy = restored.get_job('legacy')
        assert legacy['result']['decision'] == 'IN'
        assert 'progress' not in legacy['views']['A']
    finally:
        restored.close()


def test_progress_throttles_frame_writes_and_clears_previous_subtask_counters(pair_fixture, monkeypatch):
    workspace, _, _ = pair_fixture
    monkeypatch.setattr('badminton.pair_workspace.time.perf_counter', lambda: 100.)
    workspace.jobs['manual'] = {'job_id':'manual','status':'running','views':{
        'A':{'status':'running','stage_history':[]}}}
    workspace._view_clocks[('manual','A')] = 100.
    writes = []
    persist = workspace._persist

    def record(job):
        writes.append(deepcopy(job))
        persist(job)

    monkeypatch.setattr(workspace, '_persist', record)
    for count in range(1, 31):
        workspace._analysis_progress('manual', 'A', {
            'stage':'prepare','detail':'미리보기 생성','completed_units':count,'total_units':30,'unit':'frames'})
    assert len(writes) == 2  # Stage start and actual final frame, not 30 disk writes.
    workspace._analysis_progress('manual', 'A', {
        'stage':'prepare','detail':'미리보기 생성','completed_units':30,'total_units':30,'unit':'frames'})
    assert len(writes) == 2
    workspace._analysis_progress('manual', 'A', {'stage':'prepare','detail':'미리보기 프레임·시각 검증'})
    assert len(writes) == 3
    progress = workspace.get_job('manual')['views']['A']['progress']
    assert progress['detail'] == '미리보기 프레임·시각 검증'
    assert not {'completed_units','total_units','unit'}.intersection(progress)
    assert len(workspace.get_job('manual')['views']['A']['stage_history']) == 1
    saved = json.loads((workspace.root/'jobs/manual/job.json').read_text())
    assert 'completed_units' not in saved['views']['A']['progress']


def test_linking_failure_preserves_stopped_phase_completed_views_and_frozen_timer(pair_fixture, monkeypatch):
    workspace, root, _ = pair_fixture
    clock = [100.]
    monkeypatch.setattr('badminton.pair_workspace.time.perf_counter', lambda: clock[0])

    def fail_correspondence(views, catalog):
        assert all(view['status'] == 'completed' for view in views.values())
        clock[0] = 123.
        raise RuntimeError('correspondence unavailable')

    monkeypatch.setattr(workspace, '_correspondence', fail_correspondence)
    job = wait_job(workspace, submit_fixture(workspace))
    assert job['status'] == job['phase'] == 'failed'
    assert job['last_phase'] == 'linking'
    assert job['result'] is None
    assert job['finished_at'] and job['elapsed_seconds'] == 23.
    assert all(view['status'] == 'completed' for view in job['views'].values())
    assert all(view['progress']['stage'] == 'completed' for view in job['views'].values())
    clock[0] = 1000.
    assert workspace.get_job(job['job_id'])['elapsed_seconds'] == 23.
    assert workspace.list_jobs()['jobs'][0]['last_phase'] == 'linking'
    workspace.close()
    restored = PairWorkspace(root, workspace.root)
    try:
        saved = restored.get_job(job['job_id'])
        assert saved['last_phase'] == 'linking' and saved['elapsed_seconds'] == 23.
        assert all(view['status'] == 'completed' for view in saved['views'].values())
    finally:
        restored.close()


def test_landing_candidate_without_rebound_stays_a_single_approximate_position():
    candidate = _landing_candidate({'selected_landing':{'candidate_px':[123,234],'time':.5,
        'frame_index':5,'contact_observed':False},'analysis':{'width':960,'height':540,'frame_count':20}})
    assert candidate['kind']=='motion_candidate' and candidate['time_seconds']==.5
    assert candidate['inference_only'] is True and candidate['contact_observed'] is False
    assert 'time_interval' not in candidate and 'frame_interval' not in candidate


@pytest.mark.parametrize('change', [
    {'candidate_px':[float('nan'),100]}, {'candidate_px':[961,100]},
    {'frame_index':True}, {'frame_index':20}, {'time':float('inf')}, {'contact_observed':True},
])
def test_invalid_or_observed_contact_is_not_exposed_as_an_approximation(change):
    assert _landing_candidate({'selected_landing':{'candidate_px':[123,234],'time':.5,
        'frame_index':5,'contact_observed':False,**change},
        'analysis':{'width':960,'height':540,'frame_count':20}}) is None


def test_occluded_landing_candidate_preserves_inferred_location_and_time_interval():
    candidate = _landing_candidate({'selected_landing':{'candidate_px':[397,202],'time':1.8,
        'frame_index':43,'contact_observed':False,
        'image_motion_event':{'kind':'occluded_descent_and_rebound','inference_only':True,
            'point_kind':'inferred_object_center','frame_interval':[41,47],'time_interval':[1.71,1.97]}},
        'analysis':{'width':960,'height':540,'frame_count':82}})
    assert candidate['kind']=='occluded_descent_and_rebound'
    assert candidate['point_kind']=='inferred_object_center'
    assert candidate['frame_interval']==[41,47] and candidate['time_interval']==[1.71,1.97]
    assert candidate['contact_observed'] is False and candidate['inference_only'] is True


@pytest.mark.parametrize('relative', ['../escape.mp4','/private/escape.mp4',
    'evidence/../escape.mp4','evidence/missing.mp4','evidence/result.json'])
def test_optional_replay_never_exposes_unsafe_or_missing_asset(tmp_path, relative):
    (tmp_path/'evidence').mkdir()
    (tmp_path/'evidence/result.json').write_text('{}')
    assert _trajectory_replay(tmp_path, {'annotated_clip':{'path':relative}}, '/pair-media/test/')==(None,None)


def test_optional_replay_rejects_symlinks_and_nonfinite_timeline(tmp_path):
    (tmp_path/'evidence').mkdir()
    (tmp_path/'target.mp4').write_bytes(b'replay')
    (tmp_path/'evidence/link.mp4').symlink_to(tmp_path/'target.mp4')
    assert _trajectory_replay(tmp_path, {'annotated_clip':{'path':'evidence/link.mp4'}}, '/pair-media/test/')==(None,None)
    (tmp_path/'evidence/actual.mp4').write_bytes(b'replay')
    url, info = _trajectory_replay(tmp_path, {'annotated_clip':{'path':'evidence/actual.mp4',
        'source_time_origin':float('nan'),'source_start_frame':True}}, '/pair-media/test/')
    assert url.endswith('/evidence/actual.mp4') and info=={}


@pytest.mark.parametrize('region', [
    {}, [], True, {'x':0,'y':0,'width':1},
    {'x':0,'y':0,'width':1,'height':1,'contact_observed':True},
    {'x':False,'y':0,'width':1,'height':1},
    {'x':0,'y':0,'width':'1','height':1},
    {'x':float('nan'),'y':0,'width':1,'height':1},
    {'x':0,'y':0,'width':float('inf'),'height':1},
    {'x':0,'y':0,'width':10**400,'height':1},
    {'x':-.01,'y':0,'width':1,'height':1},
    {'x':0,'y':1,'width':1,'height':.1},
    {'x':0,'y':0,'width':0,'height':1},
    {'x':0,'y':0,'width':1,'height':-1},
    {'x':.9,'y':0,'width':.2,'height':1},
    {'x':0,'y':.5,'width':1,'height':.6},
])
def test_user_region_rejects_invalid_second_view_before_creating_any_job(pair_fixture, region):
    workspace, _, _ = pair_fixture
    before = set(workspace.root.rglob('*'))
    with pytest.raises(PairWorkspaceError) as error:
        workspace.submit({'views': {'A': {'video_id':'v_a', 'review_region':{'x':0,'y':0,'width':1,'height':1}},
                                   'B': {'video_id':'v_b', 'review_region':region}}})
    assert error.value.status == 400
    assert not workspace.jobs
    assert set(workspace.root.rglob('*')) == before


def test_user_region_requires_new_processing_before_job_creation(pair_fixture):
    workspace, _, _ = pair_fixture
    with pytest.raises(PairWorkspaceError) as error:
        workspace.submit({'views': {'A': {'video_id':'v_a','review_region':{'x':0,'y':0,'width':.5,'height':1}},
                                   'B': {'video_id':'v_b'}}, 'reuse_review':True})
    assert error.value.status == 400 and not workspace.jobs
    assert not (workspace.root/'jobs').exists()


@pytest.mark.parametrize('metadata,target,region,expected', [
    ({'width':3840,'height':2160,'frame_count':24}, 960,
     {'x':0,'y':0,'width':.5,'height':.5}, [0,0,480,270]),
    ({'width':3840,'height':2160,'frame_count':24}, 960,
     {'x':200/960,'y':30/540,'width':500/960,'height':480/540}, [200,30,500,480]),
    # probe has already applied display rotation: coded dimensions must not be used.
    ({'width':1080,'height':1920,'coded_width':1920,'coded_height':1080,
      'rotation_degrees':90,'frame_count':24}, 960,
     {'x':.5,'y':.5,'width':.5,'height':.5}, [480,853,480,853]),
    ({'width':961,'height':541,'frame_count':24}, 959,
     {'x':0,'y':0,'width':1,'height':1}, [0,0,958,540]),
    ({'width':640,'height':360,'frame_count':24}, 960,
     {'x':.9999,'y':.9999,'width':.0001,'height':.0001}, [639,359,1,1]),
])
def test_user_region_maps_full_timeline_and_oriented_even_pixel_bounds(metadata, target, region, expected):
    from badminton.pair_workspace import _manual_review_regions
    from badminton.detection import _refinement_regions
    actual = _manual_review_regions(region, metadata, target)
    assert actual == [{'frame_start':0,'frame_end':23,'bbox':expected}]
    width = min(target, metadata['width'])
    width -= width % 2
    height = max(2, round(metadata['height']*width/metadata['width']/2)*2)
    assert _refinement_regions({'refinement_regions':actual}, (height,width,3), 24) == actual


def test_user_region_overrides_ai_hints_and_persists_assistance_without_contact_claim(pair_fixture, monkeypatch):
    workspace, root, _ = pair_fixture
    configs = {}
    monkeypatch.setattr('badminton.media.probe', lambda _: {'frame_count':20,'width':3840,'height':2160})
    # Existing candidate/runtime budgets must survive manual guidance unchanged.
    detector = {'max_candidates':80,'highres_max_rois':16,'native_temporal_history':1,
                'native_motion_feedback':'legacy','guided_recovery_enabled':False}
    _write(root/'configs/poc.json', {'analysis_width':960,'detector':detector,'events':{'source_contact_max_candidates':6}})
    def analyze(source, output, config, progress_callback=None):
        configs[source.name] = deepcopy(config)
        return {'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                'decision':{'decision':'REVIEW','reason_codes':['FIRST_CONTACT_NOT_OBSERVED']},
                'analysis':{'width':960,'height':540,'frame_count':20},
                'selected_landing':{'candidate_px':[720,300],'frame_index':3,'time':.3,'contact_observed':False},
                'artifacts':{},'timings':{}}
    monkeypatch.setattr('badminton.pipeline.analyze_video', analyze)
    region = {'x':.5,'y':.25,'width':.5,'height':.5}
    identifier = workspace.submit({'views':{'A':{'video_id':'v_a','review_region':region},'B':{'video_id':'v_b'}},
                                   'refine_review_regions':True,'same_event_confirmed':True})['job_id']
    job = wait_job(workspace, identifier)
    manual = configs['a.mp4']
    expected = [{'frame_start':0,'frame_end':19,'bbox':[480,135,480,270]}]
    assert manual['detector']['refinement_regions'] == expected
    assert manual['events']['review_refinement_regions'] == expected
    assert all(manual['detector'][key] == value for key,value in detector.items())
    assert manual['events']['source_contact_max_candidates'] == 6
    assert 'court_hint' not in manual and 'identity' not in manual
    provenance = manual['request_context']['review_region_guidance']
    assert provenance['source']=='operator_review_region'
    assert provenance['normalized_region']==region and provenance['regions']==expected
    assert provenance['ai_visual_refinement_requested'] is True
    assert provenance['ai_visual_refinement_available'] is True
    assert provenance['ai_visual_refinement_applied'] is False
    assert provenance['priority']=='operator_region_over_ai_visual_review'
    assert provenance['assistance_used'] is True and provenance['evaluation_eligible'] is False
    result, view = job['result'], job['result']['views']['A']
    assert result['decision']=='REVIEW' and result['user_region_camera_ids']==['A']
    assert view['processing_mode']=='user_region_reanalysis'
    assert view['review_source']=='operator_region_guided_search'
    assert view['assistance_used'] is True and view['evaluation_eligible'] is False
    assert view['refinement']==provenance and view['qualified'] is False
    assert view['contact_time_seconds'] is None and view['landing_candidate']['contact_observed'] is False
    assert 'refinement_regions' not in configs['b.mp4']['detector']
    workspace.close()
    restored = PairWorkspace(root, workspace.root)
    try:
        saved = restored.get_job(identifier)
        assert saved['request']['views']['A']['review_region']==region
        assert saved['result']['views']['A']['refinement']==provenance
    finally:
        restored.close()


def test_absent_and_null_user_region_leave_automatic_config_unchanged(pair_fixture, monkeypatch):
    workspace, root, _ = pair_fixture
    monkeypatch.setattr('badminton.media.probe', lambda _: {'frame_count':20,'width':640,'height':360})
    base_config = {'analysis_width':640,'detector':{'max_candidates':80},'events':{'source_contact_max_candidates':6}}
    _write(root/'configs/poc.json', base_config)
    configs = []
    def analyze(source, output, config, progress_callback=None):
        configs.append(config)
        return {'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                'decision':{'decision':'REVIEW'},'artifacts':{}}
    monkeypatch.setattr('badminton.pipeline.analyze_video', analyze)
    identifier = workspace.submit({'views':{'A':{'video_id':'v_a','review_region':None},'B':{'video_id':'v_b'}}})['job_id']
    job = wait_job(workspace, identifier)
    assert len(configs)==2
    for config in configs:
        assert {key:config[key] for key in base_config} == base_config
        assert 'request_context' not in config
    assert all('review_region' not in view for view in job['request']['views'].values())
    assert all(view['processing_mode']=='fresh_analysis' for view in job['result']['views'].values())
    assert job['result']['event_correspondence_source'] is None


def test_uploaded_region_rerun_uses_fresh_output_and_updated_config(pair_fixture, monkeypatch):
    workspace, _, _ = pair_fixture
    monkeypatch.setattr('badminton.media.probe', lambda _: {'frame_count':20,'width':960,'height':540})
    uploaded = workspace.stage_upload(io.BytesIO(b'new-region-source'),17,'new-region.mp4')
    calls = []
    def analyze(source, output, config, progress_callback=None):
        if config.get('request_context'):
            calls.append((output, deepcopy(config)))
        return {'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                'decision':{'decision':'REVIEW'},'artifacts':{}}
    monkeypatch.setattr('badminton.pipeline.analyze_video', analyze)
    regions = [{'x':0,'y':0,'width':.5,'height':1}, {'x':.5,'y':0,'width':.5,'height':1}]
    jobs = []
    for region in regions:
        identifier = workspace.submit({'views':{'A':{'upload_id':uploaded['upload_id'],'review_region':region},
                                               'B':{'video_id':'v_b'}}})['job_id']
        jobs.append(wait_job(workspace, identifier))
    assert len(calls)==2 and calls[0][0]!=calls[1][0]
    assert calls[0][1]['detector']['refinement_regions']!=calls[1][1]['detector']['refinement_regions']
    assert calls[0][1]['request_context']!=calls[1][1]['request_context']
    assert all(job['request']['views']['A']['upload_id']==uploaded['upload_id'] for job in jobs)
    assert [job['request']['views']['A']['review_region'] for job in jobs]==regions
