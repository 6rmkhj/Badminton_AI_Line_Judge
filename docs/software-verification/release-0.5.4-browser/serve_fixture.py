from pathlib import Path
import sys
sys.path.insert(0,'/workspace/badminton-release-0.5.4')
from badminton.web import create_server
from badminton.pair_workspace import PairWorkspace
root=Path('/workspace/remediation/release054-browser/app')
server=create_server(root,port=8784,video_dir=root/'no-library')
server.pair_workspace=PairWorkspace(project_root=root/'fixture-project',workspace_root=root/'paired',library_root=root/'no-library')
print('Serving isolated completed replay on127.0.0.1:8784',flush=True)
try:server.serve_forever()
finally:server.server_close()
