"""HTTP smoke of extracted application on Linux; not a Windows executable test."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

parser=argparse.ArgumentParser()
parser.add_argument('--app',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
app=args.app.resolve(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
assert (app/'start.py').is_file()
assert not (app/'.venv').exists(), 'Use a pristine extracted package for this Linux check.'
record={'scope':'Linux HTTP smoke of freshly extracted application using Linux Python/FFmpeg; no Windows executable or GPU hardware execution','app':str(app),'python':sys.executable,'started_at':datetime.now(timezone.utc).isoformat(),'checks':{}}
log=output/'server.log'
process=None

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def request(path,body=None,headers=None,expected=200):
    if isinstance(body,dict):
        body=json.dumps(body).encode(); headers={'Content-Type':'application/json',**(headers or {})}
    req=Request(base+path,data=body,headers=headers or {})
    try:
        with urlopen(req,timeout=30) as response: status,data=response.status,response.read()
    except HTTPError as error: status,data=error.code,error.read()
    assert status==expected,(path,status,data[:1000])
    return data

def jrequest(*a,**kw):return json.loads(request(*a,**kw))

try:
    freeze=json.loads((app/'docs/landing-engine/accepted-freeze.json').read_text())
    for relative,expected in freeze['source_freeze']['source_files'].items():
        assert digest(app/relative)==expected,relative
    assert json.loads((app/'configs/poc.json').read_text())==freeze['config']
    assert (app/'scripts/setup_gpu.py').is_file() and (app/'setup-gpu-windows.bat').is_file()
    record['checks']['frozen_engine']={'python_files_matching':len(freeze['source_freeze']['source_files']),'source_tree_sha256':freeze['source_freeze']['source_tree_sha256'],'configuration_matches':True,'gpu_setup_included':True}
    for idx in (1,2):
        source=output/f'input-{idx}.mp4'
        command=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i','testsrc2=size=320x240:rate=24','-frames:v','12']
        if idx==2:command+=['-vf','hue=h=45']
        command+=['-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(source)]
        subprocess.run(command,check=True)
    env={**os.environ,'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1'};env.pop('PYTHONPATH',None)
    with log.open('w') as stdout:
        process=subprocess.Popen([sys.executable,'-u','-X','utf8',str(app/'start.py'),'--pair','--run',str(output/'run'),'--video-dir',str(output/'empty-video'),'--port','0','--no-browser'],cwd=app,stdout=stdout,stderr=subprocess.STDOUT,env=env)
        for _ in range(300):
            text=log.read_text()
            match=re.search(r'http://127\.0\.0\.1:(\d+)',text)
            if match:break
            assert process.poll() is None, text
            time.sleep(.1)
        else:raise AssertionError('Server did not announce URL: '+log.read_text())
        base='http://127.0.0.1:'+match.group(1)
        for path in ['/pair','/pair.js','/pair.css','/resources.js','/resources.css']:
            data=request(path);assert len(data)>100
            record['checks'][path]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        assert 'renderEvidenceAssessment' in request('/pair.js').decode()
        resources=jrequest('/api/resources')
        assert resources['confirmed'] is False
        record['checks']['resource_initial']={'confirmed':False,'settings':resources['settings'],'environment':resources['environment']}
        blocked=jrequest('/api/pair-jobs',{},expected=409)
        record['checks']['analysis_before_confirmation']={'http_status':409,'response':blocked}
        chosen={**resources['settings'],'device':'cpu','cpu_threads':1,'parallel_views':1,'frame_memory_mb':min(resources['settings']['frame_memory_mb'],512)}
        configured=jrequest('/api/resources',chosen)
        assert configured['confirmed'] is True and configured['settings']==chosen
        record['checks']['resource_confirmation']=configured['settings']
        uploads=[]
        for idx in (1,2):
            source=output/f'input-{idx}.mp4'
            upload=jrequest('/api/pair-uploads',source.read_bytes(),headers={'Content-Type':'application/octet-stream','X-Filename':quote(f'검증-{idx}.mp4')},expected=201)
            metadata=upload['metadata']
            assert upload['sha256']==digest(source)
            assert metadata['width']==320 and metadata['height']==240 and metadata['frame_count']==12
            assert metadata['extension']=='.mp4' and metadata['codec']=='h264'
            assert abs(metadata['average_fps']-24)<.001
            uploads.append(upload)
        record['checks']['video_metadata']=[x['metadata'] for x in uploads]
        job=jrequest('/api/pair-jobs',{'views':{'A':{'upload_id':uploads[0]['upload_id']},'B':{'upload_id':uploads[1]['upload_id']}},'same_event_confirmed':False,'reuse_review':False},expected=202)
        identifier=job['job_id']
        deadline=time.monotonic()+180
        while time.monotonic()<deadline:
            job=jrequest('/api/pair-jobs/'+identifier)
            if job['status'] in ['completed','completed_with_errors','failed','interrupted']:break
            time.sleep(.2)
        else:raise AssertionError('Analysis did not complete within 180 seconds')
        assert job['status']=='completed',job
        for camera,view in job['views'].items():
            assert view['status']=='completed',view
            assert view['processing_mode']=='fresh_analysis',view
            assert len(view['assessment']['stages'])==6,view
            assert view['automatic_decision']=='REVIEW',view
        record['checks']['fresh_analysis']={'status':job['status'],'job_id':identifier,'views':{camera:{field:view.get(field) for field in ['status','processing_mode','automatic_decision','assessment','execution']} for camera,view in job['views'].items()}}
        (output/'completed-job.json').write_text(json.dumps(job,ensure_ascii=False,indent=2)+'\n')
        record['passed']=True
finally:
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
    record['finished_at']=datetime.now(timezone.utc).isoformat()
    (output/'smoke.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'passed':record['passed'],'report':str(output/'smoke.json')},ensure_ascii=False))
