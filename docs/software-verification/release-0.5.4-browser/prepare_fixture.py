"""Isolated completed-result display fixture; never runs or fabricates inference."""
import argparse, datetime, hashlib, json, shutil, subprocess, sys
from pathlib import Path
from urllib.parse import quote
REPO=Path('/workspace/badminton-release-0.5.4')
CORE='7306efe267e40f4f8539c0439711ec54c2afcd41'
EXPECTED_FP='ec9a727c87bdf7a264f492cbe2e286de14d8c8b5760c183075c320cf613bfe25'
sys.path.insert(0,str(REPO))
from badminton.pair_workspace import _landing_candidate, _trajectory_replay, combine_views, RULES
from badminton.pipeline import implementation_fingerprint

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

parser=argparse.ArgumentParser()
parser.add_argument('--job',required=True)
parser.add_argument('--source-a',required=True)
parser.add_argument('--source-b',required=True)
parser.add_argument('--validation')
parser.add_argument('--different-events',action='store_true')
args=parser.parse_args()
root=Path('/workspace/remediation/release054-browser/app')
jobroot=root/'paired/jobs'/args.job
jobroot.mkdir(parents=True,exist_ok=True)
project=root/'fixture-project'
(project/'configs').mkdir(parents=True,exist_ok=True)
shutil.copy2(REPO/'configs/poc.json',project/'configs/poc.json')
assert implementation_fingerprint()==EXPECTED_FP
assert not subprocess.check_output(['git','diff',CORE,'--','badminton','configs'],cwd=REPO,text=True)
validation=None
if args.validation:
    validation=json.loads(Path(args.validation).read_text())
    assert validation['status']=='passed' and validation['source_commit']==CORE
    assert validation['implementation_fingerprint']==EXPECTED_FP
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
views,sources={},{}
for camera,source in [('A',Path(args.source_a)),('B',Path(args.source_b))]:
    path=source/'result.json'
    result=json.loads(path.read_text())
    assert result['pipeline_version']=='0.5.4' and result['implementation_fingerprint']==EXPECTED_FP
    assert result['decision']['decision']=='REVIEW' and result['selected_landing']['contact_observed'] is False
    if validation:
        assert sha(path)==validation['views'][camera]['result_sha256']
        assert result['source_sha256']==validation['views'][camera]['source_sha256']
    output=jobroot/'views'/camera
    output.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,output/'result.json')
    artifacts=result['artifacts']
    relative_files={artifacts['preview']}
    for entry in artifacts.get('evidence',[]):
        relative_files.update(entry[name] for name in ['image','source_sequence','source_crop','trajectory_context'] if isinstance(entry.get(name),str))
    clip=artifacts.get('annotated_clip') or {}
    relative_files.update(clip[name] for name in ['path','timing_path'] if isinstance(clip.get(name),str))
    hashes={}
    for relative in sorted(relative_files):
        raw=source/relative
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts and raw.is_file() and raw.stat().st_size>0
        destination=output/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(raw,destination)
        hashes[relative]=sha(raw)
        assert sha(destination)==hashes[relative]
    prefix=f'/pair-media/jobs/{args.job}/views/{camera}/'
    evidence,context=[],None
    for entry in artifacts.get('evidence',[]):
        for field in ['image','source_sequence','source_crop']:
            if isinstance(entry.get(field),str):evidence.append(prefix+quote(entry[field],safe='/'))
        if isinstance(entry.get('trajectory_context'),str):context=prefix+quote(entry['trajectory_context'],safe='/')
    candidate=_landing_candidate(result)
    assert candidate and candidate['point_px']==result['selected_landing']['candidate_px']
    first=artifacts['evidence'][0]
    assert first['frame_index']==candidate['frame_index'] and abs(first['time']-candidate['time_seconds'])<1e-6
    assert clip['source_start_frame']<=candidate['frame_index']<=clip['source_end_frame']
    timing=json.loads((source/clip['timing_path']).read_text())
    stamped=next(row for row in timing['frames'] if row['source_frame_index']==candidate['frame_index'])
    assert abs(stamped['source_time']-candidate['time_seconds'])<1e-6
    assert abs(stamped['replay_time']-(candidate['time_seconds']-clip['source_time_origin']))<1e-6
    replay,replay_info=_trajectory_replay(output,artifacts,prefix)
    views[camera]={'camera_id':camera,'file':result['file'],'status':'completed','rules':RULES,
        'source_sha256':result['source_sha256'],'automatic_decision':'REVIEW','direct_decision':'REVIEW',
        'assisted_decision':'REVIEW','qualified':False,'preview_url':prefix+quote(artifacts['preview'],safe='/'),
        'evidence_urls':evidence,'contact_time_seconds':None,'reason_codes':result['decision']['reason_codes'],
        'rationale':'완료된 원본 CLI 분석을 불러온 표시 검증 자료입니다. 브라우저에서 새 분석을 실행한 기록은 아닙니다.',
        'refinement_available':False,'review_source':'automatic','processing_mode':'fresh_analysis',
        'landing_candidate':candidate,'trajectory_context_url':context,'trajectory_replay_url':replay,
        'trajectory_replay_info':replay_info,'analysis_timings':result['timings'],
        'source_verification':'current_original_sha256','elapsed_seconds':result['timings']['total_seconds'],
        'started_at':None,'finished_at':None,'stage_history':[],
        'progress':{'stage':'completed','stage_label':'시점 분석 완료','stage_index':7,'stage_count':7,
                    'stage_status':'completed','updated_at':now}}
    sources[camera]={'source_result_path':str(path),'result_sha256':sha(path),'source_sha256':result['source_sha256'],
        'pipeline_version':result['pipeline_version'],'implementation_fingerprint':result['implementation_fingerprint'],
        'selected':{key:result['selected_landing'].get(key) for key in ['track_id','frame_index','time','candidate_px','contact_observed']},
        'public_landing_candidate':candidate,'artifact_sha256':hashes,'clip_timing_at_candidate':stamped,
        'artifact_selected_candidate_agreement':True,'cli_timings':result['timings']}
    receipt=source/'execution-receipt.json'
    if receipt.is_file():sources[camera]['execution_receipt']={'path':str(receipt),'sha256':sha(receipt)}
combined=combine_views(views,same_event_confirmed=False,different_event=args.different_events)
assert combined['decision']=='REVIEW'
scope='Completed CLI-result replay fixture only. No browser upload/inference, live progress timing, pair job elapsed time, cross-view synchronization or accuracy benchmark is tested. No pair/stage timing history is fabricated.'
job={'job_id':args.job,'status':'completed','phase':'completed','created_at':now,'updated_at':now,
     'started_at':None,'finished_at':None,'elapsed_seconds':None,'queue_seconds':None,'views':views,'result':combined,
     'request':{'views':{},'same_event_confirmed':False,'reuse_review':False,'refine_review_regions':False,'rules':RULES},
     'verification_scope':scope}
(jobroot/'job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2)+'\n')
provenance={'scope':scope,'core_commit':CORE,'implementation_fingerprint':EXPECTED_FP,
    'fixture_job':args.job,'fixture_created_at':now,'different_source_events':args.different_events,
    'sources':sources,'profile_sha256':sha(REPO/'configs/poc.json'),
    'ui_sha256':{name:sha(REPO/'badminton/static'/name) for name in ['pair.html','pair.js','pair.css']},
    'server_sha256':{name:sha(REPO/'badminton'/name) for name in ['web.py','pair_workspace.py']},
    'new_holdout_inputs_or_references_opened':False}
if validation:provenance['control_validation']={'path':args.validation,'sha256':sha(args.validation)}
(root/(args.job+'-provenance.json')).write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
print('Prepared',args.job)
