"""Predeclare this round's four additional same-project A/B pairs from metadata only."""
import hashlib,json,re,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path('/workspace/remediation');REPO=Path('/workspace/badminton-remediation');SOURCE='7e74b058ae95543bb9c7d121761426462e74f49c';BASE='868b1e38700ad3613f83835b71857ae36d898814'
ROOT.mkdir(exist_ok=True);inputdir=ROOT/'input';inputdir.mkdir(exist_ok=True)
if (ROOT/'cohort-selection.json').exists():raise SystemExit('Cohort already declared; do not overwrite')
old=json.loads(Path('/workspace/generalization/manifest.json').read_text());oldkeys={r['key'] for r in old['videos']};oldfiles={r['file'] for r in old['videos']};oldblobs={r['git_blob'] for r in old['videos']};oldhashes={r['sha256'] for r in old['videos']}
patterns=[('8k',re.compile(r'8k 30 4k 24 ([ab])시점 (\d+)\.mp4')),('ordinary',re.compile(r'([ab])시점 (\d+)\.mp4')),('uhd4k',re.compile(r'([ab])-(\d+)시점\(uhd30 4k30\)\.mp4')),('uhdhd',re.compile(r'([ab])시점\(uhd30 hd30\) (\d+)\.mp4'))]
# Identify known TrackNet experiment clips from artifact names, never predictions.
modelroot=Path('/workspace/generalization/next/model-research');modelkeys=set();model_inventory=[]
for p in modelroot.rglob('*'):
 if p.is_file():
  matches=re.findall(r'(?:8k|ordinary|uhd4k|uhdhd)-\d+-[ab]',str(p.relative_to(modelroot)))
  modelkeys.update(matches)
  if matches:model_inventory.append(str(p.relative_to(modelroot)))
raw=subprocess.check_output(['git','-C',str(REPO),'ls-tree','-r','-z','-l',SOURCE,'video'])
entries=[]
for line in raw.split(b'\0'):
 if not line:continue
 fields,path=line.split(b'\t',1);mode,kind,oid,size=fields.decode().split();path=path.decode();filename=Path(path).name
 for group,pattern in patterns:
  match=pattern.fullmatch(filename)
  if match:
   view,num=match.groups();num=int(num);key=f'{group}-{num}-{view}'
   entries.append({'key':key,'file':filename,'group':group,'number':num,'view':view.upper(),'git_path':path,'git_blob':oid,'bytes':int(size),'source_git_commit':SOURCE});break
originals={r['git_blob'] for r in entries if r['key'] in {'8k-1-a','8k-1-b'}}
pairs={}
for row in entries:pairs.setdefault((row['group'],row['number']),{})[row['view']]=row
eligible=[];excluded=[]
for (group,num),views in sorted(pairs.items()):
 reasons=[]
 if set(views)!={'A','B'}:reasons.append('incomplete_pair')
 for row in views.values():
  if row['key'] in oldkeys:reasons.append('prior24')
  if row['key'] in {'8k-1-a','8k-1-b'}:reasons.append('original_A1_B1_control')
  if row['key'] in modelkeys:reasons.append('known_TrackNet_experiment_clip')
  if row['git_blob'] in oldblobs and row['file'] not in oldfiles:reasons.append('content_duplicate_of_prior24')
  if row['git_blob'] in originals:reasons.append('content_of_original_A1_B1_control')
 if reasons:excluded.append({'group':group,'number':num,'keys':[r['key'] for r in views.values()],'reasons':sorted(set(reasons))})
 else:eligible.append({'group':group,'number':num,'views':views})
chosen=[]
for group in ['8k','ordinary','uhd4k','uhdhd']:
 candidates=sorted([p for p in eligible if p['group']==group],key=lambda p:p['number'])
 if not candidates:raise RuntimeError('No eligible complete pair for '+group)
 chosen.extend(candidates[0]['views'][v] for v in ['A','B'])
# Save the metadata-only selection before extracting or viewing source pixels.
declaration={'declared_at':datetime.now(timezone.utc).isoformat(),'round':'false_positive_remediation_after_0.5.3','base_commit':BASE,'source_inventory_commit':SOURCE,'selection_algorithm':'For each filename group in [8k,ordinary,uhd4k,uhdhd], choose the lowest numeric complete A/B pair after exclusions. No predictions or source pixels used for selection.','prior_prediction_inspection_for_selected_inputs_this_round':False,'pairing_scope':'Filename pairing only; actual same-rally identity and clock correspondence are not established by names.','historical_exposure_scope':'These are held out for this task round only. All74 source files previously appeared in project inventory/possible historical CV/AI workflows; globally unseen exposure is not claimed. Known TrackNet experiment clips and exact content duplicates of prior24/original controls are excluded.','known_model_experiment_keys':sorted(modelkeys),'model_artifact_name_inventory_sha256':hashlib.sha256('\n'.join(sorted(model_inventory)).encode()).hexdigest(),'excluded_pairs':excluded,'eligible_pairs':[{'group':p['group'],'number':p['number']} for p in eligible],'selected_keys':[r['key'] for r in chosen],'holdout_status':'sealed_for_current_round_until_new_code_config_freeze_and_explicit_root_release'}
(ROOT/'cohort-selection.json').write_text(json.dumps(declaration,ensure_ascii=False,indent=2)+'\n')
for row in chosen:
 destination=inputdir/row['file'];hasher=hashlib.sha256();size=0
 with destination.open('wb') as out:
  process=subprocess.Popen(['git','-C',str(REPO),'cat-file','blob',row['git_blob']],stdout=subprocess.PIPE)
  while chunk:=process.stdout.read(1024*1024):out.write(chunk);hasher.update(chunk);size+=len(chunk)
  assert process.wait()==0
 assert size==row['bytes'];digest=hasher.hexdigest();assert digest not in oldhashes,'Selected byte duplicate of prior24'
 row.update(path=str(destination),sha256=digest,source_available=True,split='holdout',round_holdout=True,reference_state='not_created_or_inspected_by_selector')
manifest={'schema_version':1,'round':declaration['round'],'source_commit':SOURCE,'baseline_code_commit':BASE,'selection_declaration_sha256':hashlib.sha256((ROOT/'cohort-selection.json').read_bytes()).hexdigest(),'scope':'Eight same-project/current-round holdout video files, four filename A/B pairs; not globally unseen data or independent human ground truth.','holdout_positions_visible_to_algorithm_agents':False,'videos':chosen}
(ROOT/'holdout-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
# Existing reference bytes are immutable even though all24 are now development/regression.
reference_root=ROOT/'regression_references';reference_root.mkdir(exist_ok=True);inventory=[]
regression=[]
for row in old['videos']:
 source=Path('/workspace/generalization/references')/(row['key']+'.json');target=reference_root/source.name;target.write_bytes(source.read_bytes());digest=hashlib.sha256(target.read_bytes()).hexdigest()
 inventory.append({'key':row['key'],'sha256':digest,'path':str(target),'original_split':row['split']})
 regression.append({**row,'historical_split':row['split'],'split':'development'})
(ROOT/'regression-manifest.json').write_text(json.dumps({'source_commit':SOURCE,'baseline_code_commit':BASE,'baseline_run':'/workspace/generalization/release','scope':'All prior24 now development/regression; original frozen reference bytes and strict13-visible criteria retained.','reference_policy':old['reference_policy'],'selection':old['selection'],'videos':regression},ensure_ascii=False,indent=2)+'\n')
(ROOT/'regression-reference-inventory.json').write_text(json.dumps({'strict_visible_count':13,'baseline_strict_matches':5,'reference_bytes_unchanged':True,'references':inventory},indent=2)+'\n')
print('Selected',[(r['key'],r['bytes'],r['sha256']) for r in chosen]);print('Known model keys',sorted(modelkeys))
