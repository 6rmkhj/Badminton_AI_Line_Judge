"""Reassemble the verified, prebuilt Windows ZIP without changing its bytes."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / 'release'
receipt = json.loads((RELEASE / 'receipt.json').read_text())
layout = json.loads((RELEASE / 'archive-layout.json').read_text())
output = ROOT / 'dist'
output.mkdir(exist_ok=True)
archive_path = output / 'badminton-windows-0.5.11.zip'
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_STORED) as archive:
    archive.comment = bytes.fromhex(layout['comment'])
    for entry in layout['entries']:
        info = zipfile.ZipInfo(entry['filename'], tuple(entry['date_time']))
        for key, value in entry.items():
            if key in ('filename', 'date_time'):
                continue
            setattr(info, key, bytes.fromhex(value) if key in ('extra', 'comment') else value)
        relative = Path(entry['filename']).relative_to('badminton-windows-0.5.11')
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid package path')
        archive.writestr(info, (ROOT / relative).read_bytes())
digest = hashlib.file_digest(archive_path.open('rb'), 'sha256').hexdigest()
if digest != receipt['sha256'] or archive_path.stat().st_size != receipt['bytes']:
    raise ValueError('Reassembled ZIP differs from the verified package')
with zipfile.ZipFile(archive_path) as archive:
    if archive.testzip() is not None:
        raise ValueError('ZIP CRC verification failed')
checksum = RELEASE / (archive_path.name + '.sha256')
if checksum.read_text().split()[0] != digest:
    raise ValueError('Published checksum differs from the package')
(output / checksum.name).write_bytes(checksum.read_bytes())
print(json.dumps({'sha256': digest, 'bytes': archive_path.stat().st_size,
                  'source_commit': receipt['source_commit'], 'verified': True}))
