"""Exercise re-execution choice without executing Windows binaries."""
from pathlib import Path
import argparse, ast, hashlib, importlib.util, json, os, sys, tempfile, types
from unittest.mock import patch

parser=argparse.ArgumentParser();parser.add_argument('--builder',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
spec=importlib.util.spec_from_file_location('package_builder',args.builder);builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
source=builder.portable_start();syntax=ast.parse(source)
syntax.body=[statement for statement in syntax.body if not(isinstance(statement,ast.Import) and any(alias.name=='sys' for alias in statement.names))]
class Reexecuted(BaseException):pass
calls=[];imports=[]
with tempfile.TemporaryDirectory(prefix='badminton launcher 검증 ') as temporary:
 root=Path(temporary);app=root/'app';environment=app/'.venv';(environment/'Scripts').mkdir(parents=True)
 executable=environment/'Scripts/python.exe';executable.write_bytes(b'not executed');(environment/'pyvenv.cfg').write_text('home = expected-base')
 launcher=root/'portable_start.py';launcher.write_bytes(source)
 fake=types.SimpleNamespace(platform='win32',maxsize=2**63-1,prefix='/different/base',argv=[str(launcher),'--no-browser'])
 def reexecute(path,argv):calls.append((path,argv));raise Reexecuted()
 def dependency_import(name):imports.append(name);raise AssertionError('Imported dependency before prepared environment')
 with patch.object(os,'execv',reexecute),patch('importlib.import_module',dependency_import):
  try:exec(compile(syntax,str(launcher),'exec'),{'__file__':str(launcher),'sys':fake})
  except Reexecuted:pass
 assert len(calls)==1 and not imports
 assert calls[0]==(str(executable),[str(executable),'-X','utf8',str(launcher),'--no-browser']),calls
 report={'scope':'Controlled launcher re-execution test with a fake Windows sys object and os.execv spy; no Windows binary executed','passed':True,'launcher_sha256':hashlib.sha256(source).hexdigest(),'prepared_venv_selected_before_dependency_import':True,'paths_with_spaces_and_korean_handled':True,'utf8_and_arguments_preserved':True}
 args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(report,ensure_ascii=False))
